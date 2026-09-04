#!/usr/bin/env python3
"""Offline comparison of retrieval-reward variants (paper Section VI-F, Table VI).

Scores eight members of the retrieval-reward family on matched secure and
vulnerable pairs (cwe, e+, e-), without any RL training. For a candidate
completion y and a retrieved exemplar pair (e+, e-), with phi the encoder
and all embeddings L2-normalized:

    pos       cos(y, e+)                               (CARGO, Eq. 2)
    con       cos(y, e+) - cos(y, e-)                  (contrastive)
    gap       <y, (e+ - e-) / ||e+ - e-||>             (fix-direction projection)
    disp      cos(y - e-, e+ - e-)
    top5      mean con over the 5 same-CWE pairs nearest to y
    proto     <y, normalized mean fix direction of the CWE>
    lex       (|T(y) & A| - |T(y) & D|) / (|A| + |D|)  (embedding-free)
    lexproto  sum over T(y) of w(t) / sqrt(|T(y)|), where w(t) is the mean
              over the CWE's pairs of 1[t added] - 1[t deleted]

T(y) is the identifier-token set of y, and A and D are the identifier
tokens the fix added and deleted. top5, proto, and lexproto leave the
candidate's own pair out.

Reported per corpus and encoder:
    - cross-task AUC: each pair's e+ (label 1) and e- (label 0) are scored
      against the nearest other pair of the same CWE;
    - in-task AUC: each pair is scored against itself;
    - within-group std: standard deviation over groups of G same-CWE
      candidates, the variance available to a group-relative update
      before lambda_rag scales it;
    - twin: the mean reward paid to the retrieved exemplar's own
      vulnerable version e-.

Input pairs are JSONL. Three formats are accepted:
    pairs   {"cwe", "e_pos", "e_neg"} per line (e.g. a RAG index's
            pairs.jsonl, filtered to one source)
    design  the authored design-pair file (target_cwe plus
            metadata.secure_completion / metadata.vulnerable_completion)
    primevul  PrimeVul paired rows ({"cwe": [...], "func", "target",
            "commit_id", "file_name"}), matched into (fixed, vulnerable)
            pairs per commit and file

Usage:
    PYTHONPATH=src python scripts/offline_retrieval_reward.py \\
        --corpus juliet=build/v0.1.7/juliet_pairs.jsonl:pairs \\
        --corpus design=data/v0.1.5.1/design_pair_patterns.jsonl:design \\
        --corpus primevul=raw/primevul/primevul_test_paired.jsonl:primevul \\
        --encoder bge=BAAI/bge-base-en-v1.5 \\
        --encoder unixcoder=microsoft/unixcoder-base \\
        --encoder codebert=microsoft/codebert-base \\
        --output results/offline_retrieval_reward.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VARIANTS = ("pos", "con", "gap", "disp", "top5", "proto", "lex", "lexproto")
TABLE_VARIANTS = ("pos", "con", "gap", "lex")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MIN_PAIRS_PER_CWE = 5


@dataclass
class Pair:
    cwe: str
    e_pos: str
    e_neg: str


def identifiers(code: str) -> set[str]:
    return set(_IDENT.findall(code))


# ---------------------------------------------------------------- loading


def load_pairs(path: Path, fmt: str) -> list[Pair]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if fmt == "pairs":
        pairs = [Pair(r["cwe"], r["e_pos"], r["e_neg"]) for r in rows]
    elif fmt == "design":
        pairs = [
            Pair(
                r["target_cwe"],
                r["metadata"]["secure_completion"],
                r["metadata"]["vulnerable_completion"],
            )
            for r in rows
        ]
    elif fmt == "primevul":
        by_commit: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for r in rows:
            cwes = r.get("cwe") or []
            if not cwes:
                continue
            key = (r["commit_id"], r["file_name"])
            by_commit[key][int(r["target"])] = r
        pairs = [
            Pair(d[1]["cwe"][0], d[0]["func"], d[1]["func"])
            for d in by_commit.values()
            if 0 in d and 1 in d and d[0]["func"] != d[1]["func"]
        ]
    else:
        raise ValueError(f"unknown format {fmt!r}")
    counts = defaultdict(int)
    for p in pairs:
        counts[p.cwe] += 1
    return [p for p in pairs if counts[p.cwe] >= MIN_PAIRS_PER_CWE]


def hf_encoder(model_id: str, device: str) -> Callable[[list[str]], np.ndarray]:
    """bge through sentence-transformers; other models mean-pooled."""
    if "bge" in model_id:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_id, device=device)
        return lambda texts: np.asarray(
            st.encode(texts, batch_size=32, normalize_embeddings=True), dtype=float
        )

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = tok(
                    texts[i : i + 32],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(float)

    return encode


# ---------------------------------------------------------------- scoring


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Scorer:
    """Scores candidates against exemplar pairs of one corpus under one encoder."""

    def __init__(self, pairs: list[Pair], pos: np.ndarray, neg: np.ndarray) -> None:
        self.pairs = pairs
        self.pos = pos
        self.neg = neg
        self.by_cwe: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(pairs):
            self.by_cwe[p.cwe].append(i)
        self.added = [identifiers(p.e_pos) - identifiers(p.e_neg) for p in pairs]
        self.removed = [identifiers(p.e_neg) - identifiers(p.e_pos) for p in pairs]

    def nearest_other(self, i: int) -> int:
        others = [j for j in self.by_cwe[self.pairs[i].cwe] if j != i]
        return max(others, key=lambda j: float(self.pos[i] @ self.pos[j]))

    def score(self, variant: str, y_vec: np.ndarray, y_text: str, ex: int, own: int) -> float:
        e_pos, e_neg = self.pos[ex], self.neg[ex]
        if variant == "pos":
            return float(y_vec @ e_pos)
        if variant == "con":
            return float(y_vec @ e_pos - y_vec @ e_neg)
        if variant == "gap":
            return float(y_vec @ _unit(e_pos - e_neg))
        if variant == "disp":
            return float(_unit(y_vec - e_neg) @ _unit(e_pos - e_neg))
        if variant == "lex":
            added, removed = self.added[ex], self.removed[ex]
            denom = len(added) + len(removed)
            toks = identifiers(y_text)
            return (len(toks & added) - len(toks & removed)) / denom if denom else 0.0
        others = [j for j in self.by_cwe[self.pairs[ex].cwe] if j != own]
        if variant == "top5":
            near = sorted(others, key=lambda j: -float(y_vec @ self.pos[j]))[:5]
            return float(np.mean([y_vec @ self.pos[j] - y_vec @ self.neg[j] for j in near]))
        if variant == "proto":
            direction = _unit(np.mean([self.pos[j] - self.neg[j] for j in others], axis=0))
            return float(y_vec @ direction)
        if variant == "lexproto":
            weight: dict[str, float] = defaultdict(float)
            for j in others:
                for t in self.added[j]:
                    weight[t] += 1.0 / len(others)
                for t in self.removed[j]:
                    weight[t] -= 1.0 / len(others)
            toks = identifiers(y_text)
            return float(sum(weight.get(t, 0.0) for t in toks)) / max(1.0, len(toks)) ** 0.5
        raise ValueError(f"unknown variant {variant!r}")


def auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Probability a secure candidate outscores a vulnerable one (ties count 1/2)."""
    p = np.asarray(pos_scores)[:, None]
    n = np.asarray(neg_scores)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def evaluate(
    pairs: list[Pair], pos: np.ndarray, neg: np.ndarray, group_size: int = 16, seed: int = 0
) -> dict[str, dict[str, float]]:
    sc = Scorer(pairs, pos, neg)
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for v in VARIANTS:
        cross_p, cross_n, in_p, in_n, twin, stds = [], [], [], [], [], []
        for i, p in enumerate(pairs):
            ex = sc.nearest_other(i)
            cross_p.append(sc.score(v, pos[i], p.e_pos, ex, i))
            cross_n.append(sc.score(v, neg[i], p.e_neg, ex, i))
            in_p.append(sc.score(v, pos[i], p.e_pos, i, i))
            in_n.append(sc.score(v, neg[i], p.e_neg, i, i))
            twin.append(sc.score(v, neg[ex], pairs[ex].e_neg, ex, ex))
            pool = [(k, side) for k in sc.by_cwe[p.cwe] if k != i for side in (0, 1)]
            pick = rng.choice(len(pool), size=min(group_size, len(pool)), replace=False)
            group = [
                sc.score(
                    v,
                    (pos if side == 0 else neg)[k],
                    pairs[k].e_pos if side == 0 else pairs[k].e_neg,
                    i,
                    k,
                )
                for k, side in (pool[j] for j in pick)
            ]
            stds.append(float(np.std(group)))
        out[v] = {
            "auc_cross": auc(cross_p, cross_n),
            "auc_in": auc(in_p, in_n),
            "std": float(np.mean(stds)),
            "twin": float(np.mean(twin)),
        }
    return out


# ---------------------------------------------------------------- report


def render_table(results: dict[str, dict[str, dict[str, dict[str, float]]]]) -> str:
    """Markdown in the layout of Table VI: Juliet AUC per encoder, PrimeVul AUC,
    Juliet std, and twin score under the first encoder."""
    encoders = list(next(iter(results.values())))
    first = encoders[0]
    lines = [
        "| Variant | "
        + " | ".join(f"Juliet AUC {e}" for e in encoders)
        + " | PrimeVul AUC | Juliet std | Twin |",
        "|---" * (len(encoders) + 4) + "|",
    ]
    for v in TABLE_VARIANTS:
        row = [f"{results['juliet'][e][v]['auc_cross']:.2f}" for e in encoders]
        pv = results.get("primevul", {}).get(first, {}).get(v, {}).get("auc_cross")
        row += [
            f"{pv:.2f}" if pv is not None else "-",
            f"{results['juliet'][first][v]['std']:.3f}",
            f"{results['juliet'][first][v]['twin']:+.2f}",
        ]
        lines.append(f"| {v} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        help="name=path:format, format in {pairs, design, primevul}",
    )
    ap.add_argument("--encoder", action="append", required=True, help="name=hf_model_id")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    corpora = {}
    for spec in args.corpus:
        name, rest = spec.split("=", 1)
        path, fmt = rest.rsplit(":", 1)
        corpora[name] = load_pairs(Path(path), fmt)
        print(f"[offline] {name}: {len(corpora[name])} pairs", file=sys.stderr)

    results: dict = defaultdict(dict)
    for spec in args.encoder:
        enc_name, model_id = spec.split("=", 1)
        encode = hf_encoder(model_id, args.device)
        for name, pairs in corpora.items():
            pos = encode([p.e_pos for p in pairs])
            neg = encode([p.e_neg for p in pairs])
            results[name][enc_name] = evaluate(pairs, pos, neg, args.group_size, args.seed)
            print(f"[offline] scored {name} / {enc_name}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    if "juliet" in results:
        table = render_table(results)
        args.output.with_suffix(".md").write_text(table + "\n")
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
