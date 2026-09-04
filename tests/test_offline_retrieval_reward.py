"""Tests for scripts/offline_retrieval_reward.py (Table VI variants)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "offline_retrieval_reward.py"
_spec = importlib.util.spec_from_file_location("offline_retrieval_reward", _SCRIPT)
orr = importlib.util.module_from_spec(_spec)
sys.modules["offline_retrieval_reward"] = orr
_spec.loader.exec_module(orr)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _corpus():
    pairs = [
        orr.Pair("CWE-89", f"q = db.execute(sql, params_{i})", f"q = db.execute(sql % user_{i})")
        for i in range(6)
    ]
    rng = np.random.default_rng(0)
    fix = _unit(rng.normal(size=8))
    pos = np.stack([_unit(rng.normal(size=8) * 0.2 + fix) for _ in pairs])
    neg = np.stack([_unit(rng.normal(size=8) * 0.2 - fix) for _ in pairs])
    return pairs, pos, neg


def test_pointwise_variants_match_their_formulas():
    pairs, pos, neg = _corpus()
    sc = orr.Scorer(pairs, pos, neg)
    y = pos[1]
    assert sc.score("pos", y, pairs[1].e_pos, 0, 1) == pytest.approx(y @ pos[0])
    assert sc.score("con", y, pairs[1].e_pos, 0, 1) == pytest.approx(y @ pos[0] - y @ neg[0])
    assert sc.score("gap", y, pairs[1].e_pos, 0, 1) == pytest.approx(y @ _unit(pos[0] - neg[0]))


def test_twin_reward_signs():
    """Cosine pays the vulnerable twin its similarity to e+; contrastive pays cos - 1."""
    pairs, pos, neg = _corpus()
    sc = orr.Scorer(pairs, pos, neg)
    twin_pos = sc.score("pos", neg[0], pairs[0].e_neg, 0, 0)
    twin_con = sc.score("con", neg[0], pairs[0].e_neg, 0, 0)
    assert twin_pos == pytest.approx(neg[0] @ pos[0])
    assert twin_con == pytest.approx(neg[0] @ pos[0] - 1.0)
    assert sc.score("lex", neg[0], pairs[0].e_neg, 0, 0) < 0


def test_lexical_overlap_counts_added_and_deleted_identifiers():
    pairs = [orr.Pair("CWE-89", "execute(sql, params)", "execute(sql % user)")] * 2
    sc = orr.Scorer(pairs, np.eye(2), np.eye(2))
    # A = {params}, D = {user}
    assert sc.score("lex", None, "execute(sql, params)", 0, 1) == pytest.approx(0.5)
    assert sc.score("lex", None, "execute(sql % user)", 0, 1) == pytest.approx(-0.5)


def test_auc_counts_ties_as_half():
    assert orr.auc([1.0, 2.0], [0.0, 0.5]) == 1.0
    assert orr.auc([1.0], [1.0]) == 0.5
    assert orr.auc([0.0], [1.0]) == 0.0


def test_evaluate_separates_secure_from_vulnerable_on_a_clean_fix_direction():
    pairs, pos, neg = _corpus()
    out = orr.evaluate(pairs, pos, neg, group_size=4)
    assert set(out) == set(orr.VARIANTS)
    assert out["con"]["auc_cross"] > 0.9
    assert out["gap"]["auc_cross"] > 0.9
    assert out["pos"]["twin"] > out["con"]["twin"]
    assert all(v["std"] >= 0 for v in out.values())


def test_load_design_pairs_drops_small_cwes(tmp_path: Path):
    rows = [
        {
            "target_cwe": "CWE-862" if i < 5 else "CWE-306",
            "metadata": {"secure_completion": f"ok{i}", "vulnerable_completion": f"bad{i}"},
        }
        for i in range(6)
    ]
    path = tmp_path / "design.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pairs = orr.load_pairs(path, "design")
    assert len(pairs) == 5
    assert {p.cwe for p in pairs} == {"CWE-862"}
