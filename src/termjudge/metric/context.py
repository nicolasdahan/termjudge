"""Load preceding source/target segments for richer LLM prompts.

Source segments are read from ``<source_docs_dir>/<doc_id>.txt`` and target
segments from ``<target_docs_dir>/<system>/<doc_id>.txt``, one segment per
line (line ``i`` is segment ``i``). The loader is doc-scoped: each file is
read once, then sliced ``[max(0, idx-N):idx]`` per occurrence using the
first source (resp. target) segment id of the occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DocSegments:
    src_lines: List[str]
    tgt_lines: List[str]


class ContextLoader:
    """Load and cache document-level segment lists.

    Both directories are optional: a missing directory (or file) simply
    yields an empty list of lines, and callers degrade accordingly.
    """

    def __init__(self, source_docs_dir: Optional[Path], target_docs_dir: Optional[Path] = None):
        self.source_docs_dir = Path(source_docs_dir) if source_docs_dir else None
        self.target_docs_dir = Path(target_docs_dir) if target_docs_dir else None
        self._cache: dict = {}

    def source_path(self, doc_id: str) -> Optional[Path]:
        if self.source_docs_dir is None:
            return None
        return self.source_docs_dir / f"{doc_id}.txt"

    def target_path(self, doc_id: str, system: str) -> Optional[Path]:
        if self.target_docs_dir is None:
            return None
        return self.target_docs_dir / system / f"{doc_id}.txt"

    def get(self, doc_id: str, system: str) -> DocSegments:
        key = (doc_id, system)
        if key in self._cache:
            return self._cache[key]

        src_lines: List[str] = []
        sp = self.source_path(doc_id)
        if sp is not None and sp.is_file():
            src_lines = _read_lines(sp)

        tgt_lines: List[str] = []
        tp = self.target_path(doc_id, system)
        if tp is not None and tp.is_file():
            tgt_lines = _read_lines(tp)

        ds = DocSegments(src_lines=src_lines, tgt_lines=tgt_lines)
        self._cache[key] = ds
        return ds

    def context_window(
        self,
        doc_id: str,
        system: str,
        src_sent_ids: List[int],
        tgt_sent_ids: List[int],
        n: int,
    ) -> str:
        """Preceding segments as two monolingual blocks (EN lines, then FR)."""
        if n <= 0:
            return ""
        ds = self.get(doc_id, system)
        src_idx = src_sent_ids[0] if src_sent_ids else 0
        tgt_idx = tgt_sent_ids[0] if tgt_sent_ids else 0
        src_window = ds.src_lines[max(0, src_idx - n) : src_idx]
        tgt_window = ds.tgt_lines[max(0, tgt_idx - n) : tgt_idx]
        if not src_window and not tgt_window:
            return ""
        lines = []
        for i, s in enumerate(src_window):
            offset = src_idx - len(src_window) + i
            lines.append(f"  EN[seg {offset}]: {s.strip()}")
        for i, t in enumerate(tgt_window):
            offset = tgt_idx - len(tgt_window) + i
            lines.append(f"  FR[seg {offset}]: {t.strip()}")
        return "\n".join(lines)

    def context_window_pairs(
        self,
        doc_id: str,
        system: str,
        src_sent_ids: List[int],
        tgt_sent_ids: List[int],
        n: int,
    ) -> str:
        """Like ``context_window``, but each SOURCE segment is paired with its
        translation instead of two monolingual blocks::

            [seg 3] EN: ...
                    FR: ...

        Pairing removes the realignment the model would otherwise have to do
        itself. Chronological order ("most recent last"): the segment closest
        to the current one comes last, adjacent to the rest of the prompt.
        """
        if n <= 0:
            return ""
        ds = self.get(doc_id, system)
        src_idx = src_sent_ids[0] if src_sent_ids else 0
        tgt_idx = tgt_sent_ids[0] if tgt_sent_ids else 0
        src_window = ds.src_lines[max(0, src_idx - n) : src_idx]
        tgt_window = ds.tgt_lines[max(0, tgt_idx - n) : tgt_idx]
        if not src_window and not tgt_window:
            return ""
        lines = []
        for i in range(max(len(src_window), len(tgt_window))):
            s = src_window[i].strip() if i < len(src_window) else ""
            t = tgt_window[i].strip() if i < len(tgt_window) else ""
            if not s and not t:
                continue
            offset = src_idx - len(src_window) + i if src_window else tgt_idx - len(tgt_window) + i
            lines.append(f"[seg {offset}] EN: {s}")
            lines.append(f"        FR: {t}")
        return "\n".join(lines)


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_discourse_context(items) -> str:
    """Assemble an optional 'Discourse context:' section from (label, value)
    pairs, dropping any pair whose value is empty. Returns '' when nothing
    remains, so the whole section (header included) disappears instead of
    printing empty '(none)' fields. Each kept value goes on its own line under
    its label; ends with a blank line so the next section follows cleanly.
    """
    kept = [(label, value) for label, value in items if value and str(value).strip()]
    if not kept:
        return ""
    lines = ["Discourse context:"]
    for label, value in kept:
        lines.append(f"- {label}:\n{value}")
    return "\n".join(lines) + "\n\n"
