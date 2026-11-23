"""A person's writes to a page: edit the body or context, resolve a disagreement, sign off, reopen.

These lived inside the web routes until a second client needed them: the browser UI, the phone app's
`/api/v1` and `api.Library` all write through here, so the rule that makes an edit *human* is stated
once. That rule: an edit or a resolution rewrites the page file **without** updating the pipeline's
fingerprint (`Store.set_fingerprint`), so the next `verify` sees a hand-edited file and merges the
person's values forward (`review.merge_human_edits`). A sign-off is the one write that does record
the fingerprint — the page is `reviewed`, and its body is kept on later runs by its status.

Nothing here knows about HTTP. A missing page file is `EditConflict` (the page exists but has not been
rendered: run `verify`), an unknown flag is `FlagNotFound`, and a sign-off over open disagreements is
`EditConflict` too; each driver maps them onto its own vocabulary.
"""
from __future__ import annotations
import time
from typing import Any

from .config import Settings
from .store import Store
from . import render, review

CONTEXT_FIELDS = ("title", "author", "chapter", "section", "running_head", "citation_key", "tags", "project")


class EditConflict(Exception):
    """The page cannot take this write in its current state."""


class FlagNotFound(LookupError):
    """No open disagreement with that number on the page."""


def _read(s: Settings, slug: str, row) -> tuple[dict, str]:
    path = render.page_path(s.documents / slug, row["sequence"])
    if not path.exists():
        raise EditConflict("page has no file yet; run verify")
    return render.read_page(path)


def edit_page(s: Settings, st: Store, slug: str, row, *, body: str | None = None, page_label: str | None = None,
              context: dict[str, Any] | None = None) -> None:
    """Replace the body, the printed page label and/or `context` fields. A human edit: the fingerprint
    is deliberately left alone so a later `verify` keeps these values."""
    fm, cur = _read(s, slug, row)
    if body is not None:
        cur = body
    if page_label is not None:
        fm["page_label"] = page_label or None
        st.set_status(row["id"], row["status"], fm["page_label"])
    if context:
        ctx = fm.setdefault("context", {})
        for k, v in context.items():
            if k in CONTEXT_FIELDS:
                ctx[k] = v
    fm.setdefault("verification", {})["flagged_spans"] = review.unresolved(cur)
    render.write_page(s.documents / slug, row["sequence"], fm, cur)


def resolve_flag(s: Settings, st: Store, slug: str, row, n: int, chosen: str) -> None:
    """Settle disagreement `n` with `chosen` (the OCR reading, the model's, or the person's own)."""
    fm, body = _read(s, slug, row)
    if not any(f.n == n for f in review.parse_flags(body)):
        raise FlagNotFound(f"flag #{n} not found")
    body = review.resolve(body, n, chosen)
    fm.setdefault("verification", {})["flagged_spans"] = review.unresolved(body)
    render.write_page(s.documents / slug, row["sequence"], fm, body)


def sign_off(s: Settings, st: Store, slug: str, row, who: str) -> None:
    """Mark the page `reviewed` by `who`. Refused while any disagreement is still open."""
    fm, body = _read(s, slug, row)
    left = review.unresolved(body)
    if left:
        raise EditConflict(f"{left} disagreement(s) still unresolved")
    v = fm.setdefault("verification", {})
    v["reviewed_by"] = {"who": who, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    v["status"] = "reviewed"
    v["flagged_spans"] = 0
    st.set_reviewed(row["id"], who, "reviewed")
    render.write_page(s.documents / slug, row["sequence"], fm, body)
    st.set_fingerprint(row["id"], review.fingerprint(fm, body))


def reopen(s: Settings, st: Store, slug: str, row) -> None:
    """Undo a sign-off: the page goes back to flagged or verified by its own numbers."""
    fm, body = _read(s, slug, row)
    v = fm.setdefault("verification", {})
    v["reviewed_by"] = None
    v["status"] = "flagged" if review.unresolved(body) else "verified"
    st.set_reviewed(row["id"], None, v["status"])
    render.write_page(s.documents / slug, row["sequence"], fm, body)
