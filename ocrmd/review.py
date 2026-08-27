"""Human review support: flag sites in a page body, applying decisions, and edit preservation.

Flags are written by `reconcile` as `==[llm reading]==[^ocrmd-n]` with a footnote
`[^ocrmd-n]: ocr="…" llm="…" — unresolved`. `ocrmd review` walks them; this module is the pure part.

Edit preservation: the pipeline overwrites page files on every `verify`/`reprocess`. So that a human
correction is not silently lost, the human-editable parts of each page (context, page label, related
links, reviewer, body) are fingerprinted when the pipeline writes the file. If the file's fingerprint
no longer matches on the next write, the file was edited by hand and `merge_human_edits` keeps the
human values: context/page_label/related/reviewer always; the body only when a reviewer signed off
(otherwise a better transcription should still replace it).
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass

MARK_RE = re.compile(r"==\[(?P<llm>.*?)\]==\[\^ocrmd-(?P<n>\d+)\]", re.S)
NOTE_RE = re.compile(r'^\[\^ocrmd-(?P<n>\d+)\]: ocr="(?P<ocr>.*?)" llm="(?P<llm>.*?)" — unresolved\s*$', re.M)


@dataclass
class FlagSite:
    n: int
    llm: str
    ocr: str
    start: int      # span of the ==[…]==[^ocrmd-n] mark in the body
    end: int

    def context(self, body: str, width: int = 70) -> str:
        pre = body[max(0, self.start - width):self.start].replace("\n", " ")
        post = body[self.end:self.end + width].replace("\n", " ")
        return f"…{pre}[{self.llm}]{post}…"


def parse_flags(body: str) -> list[FlagSite]:
    notes = {int(m["n"]): (m["ocr"], m["llm"]) for m in NOTE_RE.finditer(body)}
    out = []
    for m in MARK_RE.finditer(body):
        n = int(m["n"])
        ocr = notes.get(n, ("", ""))[0]
        out.append(FlagSite(n=n, llm=m["llm"], ocr=ocr, start=m.start(), end=m.end()))
    return out


def resolve(body: str, n: int, chosen: str) -> str:
    """Replace flag `n` with `chosen` text and drop its footnote."""
    body = re.sub(r"==\[.*?\]==\[\^ocrmd-%d\]" % n, lambda _: chosen, body, count=1, flags=re.S)
    body = re.sub(r'^\[\^ocrmd-%d\]: .*?— unresolved\s*\n?' % n, "", body, flags=re.M)
    return body.rstrip() + "\n"


def unresolved(body: str) -> int:
    return len(MARK_RE.findall(body))


# --- edit preservation --------------------------------------------------------------------------
HUMAN_FIELDS = ("context", "page_label")


def fingerprint(fm: dict, body: str) -> str:
    """Hash of everything a human is allowed to edit; provenance and nav are excluded so that a
    `build` (which rewrites nav) or a pipeline re-render never looks like a human edit."""
    payload = {
        "context": fm.get("context"),
        "page_label": fm.get("page_label"),
        "related": (fm.get("references") or {}).get("related"),
        "reviewed_by": (fm.get("verification") or {}).get("reviewed_by"),
        "body": body.strip(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def merge_human_edits(new_fm: dict, new_body: str, old_fm: dict, old_body: str) -> tuple[dict, str, list[str]]:
    """Apply human values from the existing file onto the pipeline's fresh output.
    Returns (fm, body, what was preserved)."""
    kept: list[str] = []
    ctx_new, ctx_old = dict(new_fm.get("context") or {}), dict(old_fm.get("context") or {})
    for k, v in ctx_old.items():
        if v not in (None, [], "") and v != ctx_new.get(k):
            ctx_new[k] = v
            kept.append(f"context.{k}")
    new_fm["context"] = ctx_new
    if old_fm.get("page_label") and old_fm.get("page_label") != new_fm.get("page_label"):
        new_fm["page_label"] = old_fm["page_label"]
        kept.append("page_label")
    related = (old_fm.get("references") or {}).get("related") or []
    if related:
        new_fm.setdefault("references", {})["related"] = related
        kept.append("references.related")
    reviewer = (old_fm.get("verification") or {}).get("reviewed_by")
    body = new_body
    if reviewer:
        new_fm.setdefault("verification", {})["reviewed_by"] = reviewer
        new_fm["verification"]["status"] = "reviewed"
        body = old_body
        kept += ["verification.reviewed_by", "body"]
    return new_fm, body, kept
