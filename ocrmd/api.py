"""ocrmd as a library: the embedding surface for another program.

    from ocrmd.api import Library
    lib = Library("~/scans")                                   # any directory; created on first use
    report = lib.ingest(["~/photos/book"], doc="my-book", title="My Book")
    for page in lib.pages("my-book"):
        print(page.sequence, page.status, page.text[:80])
    lib.export("my-book", "jsonl")
    lib.search("ridership")
    lib.send("my-book", 12, "law-ai", {"hint": "question 15"})

The CLI and the web UI are two drivers over the same modules (`pipeline`, `render`, `library`,
`export`, `sinks`); this is the third, for a host application. It holds one `Settings` and one
`Store` of its own — never the module-level `config.settings` singleton the CLI uses — so a host can
point it at its own root, and it takes the LLM provider as an argument: `"auto"` (the configured one,
when credentials exist), `None` (OCR only: nothing leaves the machine), or any object implementing
`llm.Provider`, which is how a host with its own model-routing layer keeps every call inside it.
Every method returns plain dataclasses and nothing here prints; progress lines go to `emit` if given.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2, numpy as np

from .config import Settings, load_settings
from .store import Store, DONE_STATUSES
from . import library as library_mod, pipeline, render, review, sinks as sinks_mod
from .export import FORMATS, export_document
from .session import LLMHealth, make_provider, process_page

log = logging.getLogger("ocrmd.api")
Emit = Callable[[str, str], None]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
__all__ = ["Library", "PageInfo", "IngestReport", "FORMATS"]


def _noop(_line: str, _level: str = "info") -> None:
    pass


@dataclass
class PageInfo:
    """One page as a host program sees it: identity, verdict, text, and where the pictures are."""
    document: str
    sequence: int
    id: str
    status: str
    page_label: str | None = None
    part: str | None = None
    captured_at: str | None = None
    path: Path | None = None                  # the page file, when one has been written
    text: str = ""                            # the reconciled Markdown body
    frontmatter: dict = field(default_factory=dict)
    agreement: float | None = None
    overlap: float | None = None
    flagged_spans: int = 0
    unresolved: int = 0
    llm_error: str | None = None
    images: dict[str, Path] = field(default_factory=dict)   # flat | processed | bin | raw -> file

    @property
    def done(self) -> bool:
        return self.status in DONE_STATUSES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path) if self.path else None
        d["images"] = {k: str(v) for k, v in self.images.items()}
        return d


@dataclass
class IngestReport:
    """What became of the files handed to `ingest`."""
    document: str
    pages: list[PageInfo] = field(default_factory=list)
    files: int = 0
    duplicates: int = 0                      # frames that re-captured a page already in the document
    rejected: int = 0                        # crops refused as unreadable (automatic captures only)
    unreadable: list[str] = field(default_factory=list)   # files no decoder could open

    def summary(self) -> str:
        s = f"{self.document}: {len(self.pages)} page(s) from {self.files} file(s)"
        if self.duplicates:
            s += f", {self.duplicates} duplicate(s) skipped"
        if self.rejected:
            s += f", {self.rejected} refused as unreadable"
        if self.unreadable:
            s += f", {len(self.unreadable)} unreadable file(s)"
        return s


class Library:
    def __init__(self, root: Path | str | None = None, *, settings: Settings | None = None,
                 provider: Any = "auto", emit: Emit | None = None):
        self.s = settings or load_settings(root=Path(root).expanduser() if root else None)
        self.s.ensure_dirs()
        self.st = Store(self.s.db_path)
        self.emit: Emit = emit or _noop
        self._provider_arg = provider
        self._provider: Any = None
        self._provider_resolved = False

    # --- lifecycle -------------------------------------------------------------------------------
    def close(self) -> None:
        self.st.close()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def provider(self):
        """The LLM provider this library transcribes with, or None for OCR-only. Resolved once."""
        if not self._provider_resolved:
            arg = self._provider_arg
            if arg == "auto":
                self._provider = make_provider(self.s, False, self.s.provider, self.s.model, self.emit)
            else:
                self._provider = arg
            self._provider_resolved = True
        return self._provider

    def _prov(self, llm: bool | None):
        if llm is False:
            return None
        return self.provider

    # --- inventory ---------------------------------------------------------------------------------
    def documents(self) -> list[library_mod.DocumentInfo]:
        return library_mod.catalog(self.s, self.st)

    def document(self, slug: str) -> library_mod.DocumentInfo:
        return library_mod.info(self.s, self.st, slug)

    def _row(self, slug: str, sequence: int):
        row = self.st.page_by_sequence(slug, sequence)
        if row is None:
            raise KeyError(f"{slug} has no page {sequence}")
        return row

    def _page_info(self, row) -> PageInfo:
        slug, pid = row["document"], row["id"]
        path = render.page_path(self.s.documents / slug, row["sequence"])
        fm: dict = {}
        body = ""
        if path.exists():
            try:
                fm, body = render.read_page(path)
            except Exception as e:                 # a hand-edited page with broken YAML is still a page
                log.warning("page %s unreadable: %s", path, e)
        v = fm.get("verification") or {}
        images: dict[str, Path] = {}
        if (self.s.processed / f"{pid}.png").exists():
            images["processed"] = self.s.processed / f"{pid}.png"
            flat = pipeline.ensure_flat(self.s, pid)
            if flat is not None:
                images["flat"] = flat
        if (self.s.processed / f"{pid}.bin.png").exists():
            images["bin"] = self.s.processed / f"{pid}.bin.png"
        if (self.s.raw / f"{row['raw_sha256']}.jpg").exists():
            images["raw"] = self.s.raw / f"{row['raw_sha256']}.jpg"
        return PageInfo(document=slug, sequence=row["sequence"], id=pid, status=row["status"],
                        page_label=fm.get("page_label") or row["page_label"], part=row["part"], captured_at=row["captured_at"],
                        path=path if path.exists() else None, text=body, frontmatter=fm,
                        agreement=v.get("agreement"), overlap=v.get("overlap"), flagged_spans=int(v.get("flagged_spans") or 0),
                        unresolved=review.unresolved(body), llm_error=self.st.stage_error(pid, "llm"), images=images)

    def pages(self, slug: str, status: str | None = None) -> list[PageInfo]:
        return [self._page_info(r) for r in self.st.pages(slug, status)]

    def page(self, slug: str, sequence: int) -> PageInfo:
        return self._page_info(self._row(slug, sequence))

    def page_by_id(self, page_id: str) -> PageInfo:
        row = self.st.page(page_id)
        if row is None:
            raise KeyError(f"no page {page_id}")
        return self._page_info(row)

    # --- in ------------------------------------------------------------------------------------------
    @staticmethod
    def input_files(paths: Iterable[Path | str]) -> list[Path]:
        """Images and PDFs, a directory's worth at a time, in name order."""
        out: list[Path] = []
        for p in paths:
            p = Path(p).expanduser()
            if p.is_dir():
                out += [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES + (".pdf",)]
            elif p.exists():
                out.append(p)
        return out

    def _landed(self, report: IngestReport, res: pipeline.Ingested, prov, health: LLMHealth, on_page) -> None:
        if res.duplicate_of:
            report.duplicates += 1
        report.rejected += len(res.rejected)
        for pid in res.pages:
            process_page(self.s, self.st, pid, None if (prov is None or health.degraded) else prov, report.document,
                         health=health, emit=self.emit)
            info = self.page_by_id(pid)
            report.pages.append(info)
            if on_page:
                on_page(info)

    def _finish(self, slug: str) -> None:
        render.build_document(self.s.documents / slug, slug)
        library_mod.sync_document(self.s, self.st, slug)

    def ingest(self, paths: Iterable[Path | str], doc: str, *, title: str | None = None, tags: Iterable[str] = (),
               project: str | None = None, llm: bool | None = None,
               on_page: Callable[[PageInfo], None] | None = None) -> IngestReport:
        """Image files, PDFs or directories of them → pages of `doc`, each read by the local engines and
        (unless `llm=False`, or the library has no provider) by the model, then the document is built.

        `title`/`tags`/`project` seed document.yaml the first time a document is seen and never
        overwrite what a person wrote there. A file the shutter would have refused is *not* refused
        here: an operator handing over a file is a decision, not a guess."""
        files = self.input_files(paths)
        self.st.ensure_document(doc)
        render.ensure_document_meta(self.s.documents / doc, doc, title=title, tags=list(tags), project=project)
        prov = self._prov(llm)
        health = LLMHealth(self.s.llm_failure_limit)
        report = IngestReport(document=doc, files=len(files))
        for f in files:
            if f.suffix.lower() == ".pdf":
                try:
                    pipeline.ingest_pdf(self.s, self.st, f, doc, on_page=lambda _n, res: self._landed(report, res, prov, health, on_page))
                except RuntimeError as e:                 # pymupdf missing, or an unreadable file
                    self.emit(f"{f.name}: {e}", "error")
                    report.unreadable.append(f.name)
                continue
            img = cv2.imread(str(f))
            if img is None:
                report.unreadable.append(f.name)
                continue
            self._landed(report, pipeline.ingest_frame(self.s, self.st, img, doc, f"file:{f.name}"), prov, health, on_page)
        self._finish(doc)
        return report

    def ingest_image(self, image: np.ndarray | bytes | Path | str, doc: str, *, source: str = "api",
                     llm: bool | None = None, variants: list[np.ndarray] | None = None) -> IngestReport:
        """One frame — an array, encoded bytes, or a path — as a capture of `doc`."""
        if isinstance(image, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image, (str, Path)):
            img = cv2.imread(str(Path(image).expanduser()))
        else:
            img = image
        report = IngestReport(document=doc, files=1)
        if img is None:
            report.unreadable.append(str(image) if isinstance(image, (str, Path)) else "<bytes>")
            return report
        self.st.ensure_document(doc)
        render.ensure_document_meta(self.s.documents / doc, doc)
        health = LLMHealth(self.s.llm_failure_limit)
        self._landed(report, pipeline.ingest_frame(self.s, self.st, img, doc, source, variants), self._prov(llm), health, None)
        self._finish(doc)
        return report

    def verify(self, doc: str | None = None, *, refresh: bool = False, flagged_only: bool = False,
               llm: bool | None = None) -> list[PageInfo]:
        """(Re)run the stages that are missing or stale — the LLM pass for pages made OCR-only, the OCR
        stage for pages read by an older engine set. Without `refresh` a completed LLM pass is reused
        whatever prompt made it: a version bump must never silently spend tokens on a whole book."""
        s, st = self.s, self.st
        prov = self._prov(llm)
        pages = st.pages(doc, "flagged" if flagged_only else None)
        ocr_ver = pipeline.ocr_version(s)
        stale = lambda p: pipeline.ocr_stale(s, st, p, ocr_ver)
        if refresh:
            current = pipeline.llm_version(s, prov.model if prov else None)
            targets = [p for p in pages if flagged_only or stale(p) or not st.stage_done(p["id"], "llm", current)]
        else:
            targets = [p for p in pages if flagged_only or stale(p) or p["status"] not in DONE_STATUSES]
        health = LLMHealth(s.llm_failure_limit)
        out = []
        for page in targets:
            if prov is not None and health.degraded:
                self.emit("LLM unavailable; finishing the rest OCR-only", "error")
                prov = None
            process_page(s, st, page["id"], prov, page["document"], refresh=refresh, health=health, emit=self.emit)
            out.append(self.page_by_id(page["id"]))
        for slug in {p["document"] for p in targets}:
            self._finish(slug)
        return out

    def build(self, doc: str) -> Path:
        render.ensure_document_meta(self.s.documents / doc, doc)
        out = render.build_document(self.s.documents / doc, doc)
        library_mod.sync_document(self.s, self.st, doc)
        return out

    # --- out -----------------------------------------------------------------------------------------
    def export(self, doc: str, fmt: str = "md", out: Path | str | None = None) -> Path:
        """The document in one of `FORMATS` (md | txt | json | jsonl), written under `<root>/export/`
        unless `out` says where."""
        return export_document(self.s, self.st, doc, fmt, Path(out) if out else None)

    def envelope(self, doc: str, sequence: int, *, text: str | None = None, bbox: list[float] | None = None) -> dict:
        """The page as a destination would receive it (see `sinks.envelope`) — also the handiest
        single object for a host that wants text plus provenance in one go."""
        return sinks_mod.envelope(self.s, doc, self._row(doc, sequence), text=text, bbox=bbox)

    def search(self, query: str, doc: str | None = None, limit: int = 50) -> list[library_mod.Hit]:
        return library_mod.search(self.s, self.st, query, doc, limit)

    def reindex(self, doc: str | None = None, force: bool = False) -> int:
        return library_mod.reindex(self.s, self.st, doc, force)

    def read_region(self, doc: str, sequence: int, bbox: list[float], *, llm: bool = False,
                    question: str | None = None) -> pipeline.RegionRead:
        """One rectangle of a page read on its own terms; nothing is written (see `pipeline.read_region`)."""
        return pipeline.read_region(self.s, self.st, self._row(doc, sequence), bbox, self.provider if llm else None, question)

    # --- destinations --------------------------------------------------------------------------------
    def sinks(self) -> list:
        return sinks_mod.registry(self.s)

    def send(self, doc: str, sequence: int, sink: Any, options: dict | None = None, *, text: str | None = None,
             bbox: list[float] | None = None, dry_run: bool = False) -> sinks_mod.Delivery:
        """Hand a page (or a reading of part of one) to a destination, by name or as an object."""
        target = sinks_mod.get(self.s, sink) if isinstance(sink, str) else sink
        return sinks_mod.send_page(self.s, doc, self._row(doc, sequence), target, options, text=text, bbox=bbox, dry_run=dry_run)

    # --- housekeeping --------------------------------------------------------------------------------
    def delete_pages(self, doc: str, sequences: Iterable[int], *, purge_raw: bool = True,
                     whole_capture: bool = False) -> pipeline.Deleted:
        ids = [r["id"] for seq in sequences if (r := self.st.page_by_sequence(doc, seq)) is not None]
        out = pipeline.delete_pages(self.s, self.st, doc, ids, purge_raw=purge_raw, whole_capture=whole_capture)
        if doc in self.st.documents():
            self._finish(doc)
        return out

    def delete_document(self, doc: str, *, purge_raw: bool = True) -> pipeline.Deleted:
        return pipeline.delete_document(self.s, self.st, doc, purge_raw=purge_raw)

    def doctor(self) -> dict:
        from .web.health import doctor_report
        return doctor_report(self.s, self.st, probe_cameras=False)
