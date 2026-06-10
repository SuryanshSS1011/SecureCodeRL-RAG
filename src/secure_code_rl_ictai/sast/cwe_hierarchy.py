"""CWE parent/child hierarchy.

Loads data/cwe_hierarchy.json and exposes:
  - `CweHierarchy.parents_of(cwe) -> set[str]`
  - `CweHierarchy.ancestors_of(cwe) -> set[str]` (transitive)
  - `is_related(a, b, h) -> bool` (a is ancestor of b OR b is ancestor of a
    OR a == b)

The normalizer uses `is_related` during merge() to corroborate parent /
child finding pairs across tools (see docs/sast_pipeline_spec.md and the
'parent/child CWE hierarchy resolution' bullet in plan §5.4).

The hierarchy is hand-curated from MITRE CWE data for our ICTAI 19-CWE
taxonomy. Expanding it is a static-file change; no code change required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "cwe_hierarchy.json"


class CweHierarchy:
    """In-memory CWE parent index with transitive-ancestor lookup."""

    def __init__(self, parents_map: dict[str, list[str] | set[str]]) -> None:
        # Normalize values to sets for fast membership tests.
        self._parents: dict[str, frozenset[str]] = {
            k: frozenset(v) for k, v in parents_map.items()
        }

    def parents_of(self, cwe: str) -> set[str]:
        """Direct parents of `cwe` (one hop)."""
        return set(self._parents.get(cwe, frozenset()))

    def ancestors_of(self, cwe: str) -> set[str]:
        """Transitive ancestors of `cwe` (parents, grandparents, ...).

        Cycle-safe: tracks visited nodes during BFS.
        """
        seen: set[str] = set()
        frontier: list[str] = list(self.parents_of(cwe))
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            for parent in self.parents_of(node):
                if parent not in seen:
                    frontier.append(parent)
        return seen


def is_related(a: str, b: str, hierarchy: CweHierarchy) -> bool:
    """True iff `a` and `b` are the same OR one is a transitive ancestor of the other."""
    if a == b:
        return True
    if b in hierarchy.ancestors_of(a):
        return True
    if a in hierarchy.ancestors_of(b):
        return True
    return False


def load_default_hierarchy(path: Optional[Path] = None) -> CweHierarchy:
    """Load the hierarchy from data/cwe_hierarchy.json."""
    p = path or _DEFAULT_PATH
    raw = json.loads(p.read_text())
    return CweHierarchy(raw.get("parents", {}))
