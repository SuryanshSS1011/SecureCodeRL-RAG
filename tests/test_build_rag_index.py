"""Tests for scripts/build_rag_index.py.

The script reads exemplar_pairs.jsonl, embeds e_pos via an embedder, and
writes pairs.jsonl + bm25.pickle + faiss.index + manifest.json under the
output dir.

For tests we use a hand-crafted MockEmbedder (selected via --embedder
mock) so we don't pull sentence-transformers. The real embedder is the
default but only exercised on ROAR.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_rag_index.py"


def _run_script(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _write_exemplar_pairs(path: Path, n: int = 3) -> None:
    """Write n exemplar pairs in the corpus build's output format."""
    records = []
    for i in range(n):
        records.append(
            {
                "cwe": f"CWE-{89 if i % 2 == 0 else 79}",
                "task_signature": f"def task_{i}():",
                "e_pos": f"# secure code {i}\nreturn None\n",
                "e_neg": f"# vuln code {i}\nreturn None\n",
                "language": "python",
                "source": "test",
                "cve_id": None,
                "fix_commit": None,
                "metadata": {},
            }
        )
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_build_rag_index_writes_outputs(tmp_path: Path):
    pairs_path = tmp_path / "exemplar_pairs.jsonl"
    _write_exemplar_pairs(pairs_path)
    output_dir = tmp_path / "rag_index"

    result = _run_script(
        [
            "--exemplar-pairs", str(pairs_path),
            "--output", str(output_dir),
            "--embedder", "mock",
            "--mock-dim", "4",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

    for name in ("pairs.jsonl", "bm25.pickle", "faiss.index", "manifest.json"):
        assert (output_dir / name).exists(), f"missing {name}"


def test_build_rag_index_manifest_carries_hashes(tmp_path: Path):
    pairs_path = tmp_path / "exemplar_pairs.jsonl"
    _write_exemplar_pairs(pairs_path)
    output_dir = tmp_path / "rag_index"

    _run_script(
        [
            "--exemplar-pairs", str(pairs_path),
            "--output", str(output_dir),
            "--embedder", "mock",
            "--mock-dim", "4",
        ],
        cwd=tmp_path,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "hashes" in manifest
    for name in ("pairs.jsonl", "bm25.pickle", "faiss.index"):
        assert name in manifest["hashes"]
        actual = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        assert manifest["hashes"][name] == f"sha256:{actual}"

    assert manifest["n_pairs"] == 3
    assert manifest["embedding_dim"] == 4


def test_build_rag_index_bm25_pickle_is_loadable(tmp_path: Path):
    pairs_path = tmp_path / "exemplar_pairs.jsonl"
    _write_exemplar_pairs(pairs_path)
    output_dir = tmp_path / "rag_index"

    _run_script(
        [
            "--exemplar-pairs", str(pairs_path),
            "--output", str(output_dir),
            "--embedder", "mock",
            "--mock-dim", "4",
        ],
        cwd=tmp_path,
    )
    # The pickle contains a Bm25Backend ready to .search() (with rank_bm25
    # available at unpickle time). For the unit test we just verify
    # unpickling doesn't raise and returns a non-None object.
    with open(output_dir / "bm25.pickle", "rb") as fh:
        try:
            obj = pickle.load(fh)
        except ImportError:
            pytest.skip("rank_bm25 not installed locally")
    assert obj is not None


def test_build_rag_index_refuses_overwrite_without_force(tmp_path: Path):
    pairs_path = tmp_path / "exemplar_pairs.jsonl"
    _write_exemplar_pairs(pairs_path)
    output_dir = tmp_path / "rag_index"

    r1 = _run_script(
        [
            "--exemplar-pairs", str(pairs_path),
            "--output", str(output_dir),
            "--embedder", "mock",
            "--mock-dim", "4",
        ],
        cwd=tmp_path,
    )
    assert r1.returncode == 0

    r2 = _run_script(
        [
            "--exemplar-pairs", str(pairs_path),
            "--output", str(output_dir),
            "--embedder", "mock",
            "--mock-dim", "4",
        ],
        cwd=tmp_path,
    )
    assert r2.returncode != 0
    assert "exists" in r2.stderr.lower() or "force" in r2.stderr.lower()

    r3 = _run_script(
        [
            "--exemplar-pairs", str(pairs_path),
            "--output", str(output_dir),
            "--embedder", "mock",
            "--mock-dim", "4",
            "--force",
        ],
        cwd=tmp_path,
    )
    assert r3.returncode == 0


def test_build_rag_index_rejects_missing_pairs(tmp_path: Path):
    result = _run_script(
        [
            "--exemplar-pairs", str(tmp_path / "nope.jsonl"),
            "--output", str(tmp_path / "out"),
            "--embedder", "mock",
            "--mock-dim", "4",
        ],
        cwd=tmp_path,
    )
    assert result.returncode != 0
