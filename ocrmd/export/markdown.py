"""The built document as an export: `render.build_document` regenerated, then copied to the export tree.

The page files stay the source of truth; the built file is derived from them, so exporting it is
"build, then copy" and never a third copy of the text that could drift."""
from __future__ import annotations
import shutil
from pathlib import Path

from ..config import Settings
from ..store import Store
from .. import render
from . import default_path

FORMAT = "md"


def export_document(s: Settings, st: Store, slug: str, out: Path | None = None) -> Path:
    render.ensure_document_meta(s.documents / slug, slug)
    built = render.build_document(s.documents / slug, slug)
    out = out or default_path(s, slug, FORMAT)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, out)
    return out
