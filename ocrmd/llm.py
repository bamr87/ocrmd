"""LLM providers: Anthropic (Claude, structured outputs, batches) and xAI (OpenAI-compatible JSON schema).

Errors are split into two families so a scan can react correctly:
- `LLMUnavailable` — the *service* is unusable for this session (no network, bad credentials, model
  not served to this token). A scan drops to OCR-only after a few of these; `ocrmd verify` fills in later.
- `LLMPageError` — this *page* failed (refusal, content filter, schema mismatch). Recorded, skipped, the
  scan carries on.
"""
from __future__ import annotations
import base64, json, logging, os, time
from dataclasses import dataclass, field
from typing import Protocol
import cv2, numpy as np
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("ocrmd.llm")

DIALECT = """You transcribe scanned pages into a strict Markdown dialect:
- ATX headings (#, ##, ...) reflecting the document hierarchy, not font size.
- Rejoin paragraphs across line breaks; de-hyphenate words broken at line ends.
- Numbered and bulleted lists stay lists; boxed tips, sidebars and callouts become blockquotes.
- GFM pipe tables for anything tabular — a table is never flattened into prose. Receipts: | Item | Qty | Unit | Total | then a totals block.
- Mathematics as LaTeX: inline `$...$`, displayed `$$...$$` on its own lines. Never spell a formula out in words.
- Source footnotes as [^n]. No inline HTML.
- A figure, photograph, diagram or chart is a placeholder line `![figure n](figure-n)` at the place it sits in the
  reading order, followed by its printed caption as a paragraph if it has one; describe each in `figures`.
  Never describe a picture's contents in the body — only the caption that is printed.
- Report the column count of the body text in `columns` and whether the page carries equations in `has_math`.
- Transcribe exactly what is printed; never summarise, never add content that is not on the page.
- Running heads, running feet and the printed page number go in the structured fields, not the body.
- If a word is unreadable, transcribe your best guess and list it in `uncertain`."""

ANTI_FABRICATION = """CRITICAL — do not invent text. These pages are photographed by a webcam and are often
low-resolution, rotated, out of focus, or partly covered by something. If you cannot
actually read a region, you must NOT fill it in with plausible-sounding content: no invented chapter titles,
no invented table-of-contents entries, no invented numbers, no completing a sentence a finger covers.
Transcribe only glyphs you can genuinely see.
If most of the page is illegible, set `legible: false`, explain why in `illegible_reason`, and put only the
text you could truly read in `markdown` (an empty string is a valid, useful answer). In `illegible_reason`
and `notes` describe only what is visible (blur, small type, cropping, glare, an object over the text);
never guess at causes you cannot see.
A short honest transcription is always better than a long invented one.
`markdown` holds page text only — never a description of the page ("this appears to be a contents page"),
never an apology, never a note about image quality. Observations about the image go in `notes`."""

SYSTEM = f"""You are the transcription stage of ocrmd, a scanning tool.
{DIALECT}

{ANTI_FABRICATION}
You receive the page image, a draft from a local OCR engine, and context about where the page sits in the
document. The image is the truth; the draft is only a hint for spelling and line order; the context helps
you keep heading levels and running heads consistent from page to page.
Sometimes you also receive further images: additional exposures of THE SAME page, taken moments apart at
different camera focus settings, together with a list of the regions local OCR could not read at any of
them. They are one page seen several times, never several pages — transcribe the page once, using
whichever exposure lets you read a given region. If a region is unreadable in all of them, it stays
unreadable: report it in `uncertain` (or `legible: false`) rather than choosing between guesses.
Return the structured result."""


class Span(BaseModel):
    text: str
    reason: str = ""


class PageContext(BaseModel):
    chapter: str | None = None
    section: str | None = None
    running_head: str | None = None
    cites: list[str] = []
    figures: list[str] = []


class FigureNote(BaseModel):
    """One picture on the page: which placeholder it is, what it is, what the page calls it."""
    n: int = Field(description="the placeholder number, 1-based, in reading order")
    kind: str = Field("figure", description="photo | diagram | chart | table_image | equation | figure")
    caption: str = Field("", description="the printed caption, exactly; empty when there is none")


class Transcription(BaseModel):
    kind: str = Field(description="book_page | receipt | letter | note | other")
    markdown: str
    title: str | None = None
    page_number: str | None = Field(None, description="page label as printed, e.g. 'xii' or '203'")
    language: str = "en"
    legible: bool = Field(True, description="false when most of the page could not genuinely be read")
    illegible_reason: str | None = Field(None, description="why, when legible is false (blurry, too small, rotated, covered...)")
    uncertain: list[Span] = []
    context: PageContext = PageContext()
    figures: list[FigureNote] = []
    columns: int = Field(1, description="columns of body text on the page, 1 for ordinary prose")
    has_math: bool = Field(False, description="the page carries equations or mathematical notation")
    tables: int = Field(0, description="how many tables the page carries")
    notes: str | None = None


def strict_schema(schema: dict) -> dict:
    """Structured outputs require additionalProperties:false on every object (incl. $defs)."""
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                n.setdefault("additionalProperties", False)
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(schema)
    return schema


TRANSCRIPTION_SCHEMA = strict_schema(Transcription.model_json_schema())


class LLMResult(BaseModel):
    transcription: Transcription
    provider: str
    model: str
    request_id: str | None
    usage: dict
    latency_ms: int


class JsonResult(BaseModel):
    """One structured, text-only answer: what `etl` runs on, where `transcribe` runs on an image."""
    data: dict
    provider: str
    model: str
    request_id: str | None = None
    usage: dict = {}
    latency_ms: int = 0


@dataclass
class PageHint:
    """What the pipeline knows before the model looks: where this page sits in the book."""
    part: str | None = None               # left | right (half of a spread) | None
    document_title: str | None = None
    prev_page_label: str | None = None
    prev_chapter: str | None = None
    prev_section: str | None = None
    prev_running_head: str | None = None
    kind: str | None = None               # e.g. "textbook"

    def render(self) -> str:
        lines = []
        if self.document_title: lines.append(f"Document: {self.document_title}")
        if self.kind: lines.append(f"Document kind: {self.kind}")
        if self.part: lines.append(f"This image is the {self.part} page of a two-page spread.")
        prev = {k: v for k, v in (("page label", self.prev_page_label), ("chapter", self.prev_chapter),
                                  ("section", self.prev_section), ("running head", self.prev_running_head)) if v}
        if prev:
            lines.append("Previous page: " + "; ".join(f"{k}={v}" for k, v in prev.items())
                         + ". Carry the chapter/section forward unless this page starts a new one.")
        return "\n".join(lines)


NOTE_SYSTEM = """You read handwriting off a photographed book page.
The image is a small crop of a page: someone has written a note on the paper by hand, in a margin or
between the printed lines. Transcribe ONLY the handwriting, exactly as written.
Do not transcribe the printed text around it, do not explain the note, do not tidy the grammar.
If you cannot read the handwriting, return an empty string — a wrong guess at someone's own note is
worse than no note at all. Return only the transcription, with no preamble."""


REGION_SYSTEM = """You read one small region of a photographed book page, cut out by a human who has
pointed at it because the rest of the pipeline got it wrong or could not read it at all.

Transcribe exactly what is printed or written inside the crop, and nothing outside it. Keep the
original spelling, capitalisation, hyphenation and line breaks; keep symbols, diacritics and
mathematics as they appear. Do not continue the sentence beyond the crop, do not tidy the grammar,
and do not explain what you see.

If part of it is genuinely unreadable, write that part as [illegible] rather than guessing: this
region has been singled out precisely because a guess would go unnoticed. If none of it can be read,
say only: cannot read this region. Return the transcription alone, with no preamble."""


class LLMError(RuntimeError):
    """Base for provider failures."""


class LLMUnavailable(LLMError):
    """The service cannot be used right now: auth, network, quota, model not served to this token."""


class LLMPageError(LLMError):
    """This page failed; others may succeed."""


class Provider(Protocol):
    name: str
    model: str
    def transcribe(self, image_bgr: np.ndarray, ocr_hint: str, hint: PageHint | None = None,
                   focus: FocusViews | None = None) -> LLMResult: ...


def _image_block(bgr: np.ndarray) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(bgr)}}


def _jpeg_b64(bgr: np.ndarray, max_side: int = 1568) -> str:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])   # small type suffers at 90
    return base64.b64encode(buf.tobytes()).decode()


@dataclass
class FocusViews:
    """Extra exposures of one page from the focus bracket, plus the analysis of what stayed unreadable.

    Attached only when a region survived `multifocus.resolve` unresolved: the images cost roughly 1.5 k
    input tokens each, and a page local OCR read confidently has nothing for them to add.
    """
    images: list[np.ndarray] = field(default_factory=list)
    note: str = ""

    def __bool__(self) -> bool:
        return bool(self.images or self.note)


def clean_hint(text: str) -> str:
    """Drop OCR lines that are mostly non-Latin/garbage; they mislead the model and can trip filters."""
    import re
    keep = []
    for line in text.splitlines():
        letters = re.findall(r"[A-Za-z]", line)
        if line.strip() and len(letters) >= 0.6 * len(re.sub(r"\s", "", line)):
            keep.append(line)
    return "\n".join(keep)


def user_text(ocr_hint: str, hint: PageHint | None, focus: "FocusViews | None" = None) -> str:
    parts = []
    if hint and hint.render():
        parts.append(hint.render())
    if focus is not None and focus.note:
        parts.append(focus.note)
    cleaned = clean_hint(ocr_hint) if ocr_hint else ""
    parts.append(f"Local OCR draft (hint only):\n\n{cleaned}" if cleaned else "Transcribe the page.")
    return "\n\n".join(parts)


def image_block(bgr: np.ndarray) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(bgr)}}


def content_blocks(image_bgr: np.ndarray, ocr_hint: str, hint: PageHint | None,
                   focus: "FocusViews | None" = None) -> list[dict]:
    """The page image, then any extra exposures each behind a label saying what it is, then the text.

    The labels are not decoration: an unlabelled second image of the same page reads as the next page
    of the book, and a model that believes that will merge two readings of one page into one
    transcription — the exact fabrication this pipeline exists to catch.
    """
    content: list[dict] = [image_block(image_bgr)]
    for i, extra in enumerate((focus.images if focus else []), 1):
        content.append({"type": "text", "text": f"Alternative exposure {i} of the SAME page (different focus):"})
        content.append(image_block(extra))
    content.append({"type": "text", "text": user_text(ocr_hint, hint, focus)})
    return content


def parse_transcription(text: str) -> Transcription:
    try:
        return Transcription.model_validate_json(text)
    except ValidationError as e:
        raise LLMPageError(f"model returned an object that does not match the schema: {str(e)[:200]}") from e


# --- Anthropic -----------------------------------------------------------------------------------
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", timeout: float = 180.0, max_retries: int = 3,
                 effort: str = "medium"):
        import anthropic
        # OAuth bearer tokens (ANTHROPIC_AUTH_TOKEN) require the oauth beta header; API keys do not.
        headers = {"anthropic-beta": "oauth-2025-04-20"} if (
            os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY")) else None
        self.client = anthropic.Anthropic(default_headers=headers, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.effort = effort

    @property
    def _modern(self) -> bool:
        """Fable 5 / Opus 5 / Sonnet 5 / 4.6+ family: adaptive thinking, effort, refusal fallbacks."""
        m = self.model
        return any(m.startswith(p) for p in ("claude-fable-5", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-6",
                                              "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6"))

    def _params(self, image_bgr: np.ndarray, ocr_hint: str, hint: PageHint | None, batch: bool = False,
                focus: FocusViews | None = None) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": TRANSCRIPTION_SCHEMA}}
        params: dict = dict(
            model=self.model, max_tokens=16000, output_config=output_config,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content_blocks(image_bgr, ocr_hint, hint, focus)}],
        )
        if self._modern:
            output_config["effort"] = self.effort
            if not batch:   # the fallbacks parameter is rejected by the Batches API
                params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return params

    def read_handwriting(self, image_bgr: np.ndarray) -> str:
        """Transcribe a handwritten note from a crop of a page. "" when it cannot be read.

        Deliberately small and unstructured: this is one short string, not a page, and wrapping it in
        the transcription schema would invite the model to describe the printed text around it.
        """
        import anthropic
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=400, system=NOTE_SYSTEM,
                messages=[{"role": "user", "content": [image_block(image_bgr),
                                                       {"type": "text", "text": "Transcribe the handwriting."}]}])
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMUnavailable(f"anthropic credentials rejected: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {e}") from e
        except anthropic.APIError as e:
            raise LLMPageError(str(e)[:300]) from e
        if msg.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    def read_region(self, image_bgr: np.ndarray, ocr_hint: str = "", question: str | None = None) -> str:
        """Transcribe one region of a page that someone selected by hand. "" when it cannot be read.

        Deliberately outside the page pipeline: no schema, no page context, nothing written to the
        store. This answers "what does *that* say", and the answer is handed to a human who is looking
        at the same crop — the one place in ocrmd where a single unreconciled reading is enough.
        """
        import anthropic
        text = question or "Transcribe this region."
        if ocr_hint.strip():
            text += f"\n\nLocal OCR read it as (hint only, it may be wrong):\n\n{clean_hint(ocr_hint)}"
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=1200, system=REGION_SYSTEM,
                messages=[{"role": "user", "content": [image_block(image_bgr), {"type": "text", "text": text}]}])
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMUnavailable(f"anthropic credentials rejected: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {e}") from e
        except anthropic.APIError as e:
            raise LLMPageError(str(e)[:300]) from e
        if msg.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 4000,
                      image_bgr: np.ndarray | None = None) -> JsonResult:
        """One structured answer to one question. No page, no transcription schema.

        This is what `etl` is built on: the operators ask about text that has *already* been read twice
        and reconciled, so there is nothing to cross-check here and usually no reason to send a
        picture. `image_bgr` is the exception — a figure crop, when the question is about the picture
        itself (what kind of chart, what the axes say) and no transcription can answer it.
        """
        import anthropic
        t0 = time.monotonic()
        content: list | str = user if image_bgr is None else [_image_block(image_bgr), {"type": "text", "text": user}]
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}])
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMUnavailable(f"anthropic credentials rejected: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMUnavailable(f"rate limited: {e}") from e
        except anthropic.APIError as e:
            raise LLMPageError(str(e)[:300]) from e
        if msg.stop_reason == "refusal":
            raise LLMPageError("refusal")
        if msg.stop_reason == "max_tokens":
            raise LLMPageError(f"answer truncated at max_tokens ({max_tokens})")
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            data = json.loads(text)
        except ValueError as e:
            raise LLMPageError(f"model returned text that is not JSON: {text[:200]!r}") from e
        return JsonResult(data=data, provider=self.name, model=msg.model, request_id=msg.id,
                          usage=msg.usage.model_dump(), latency_ms=int((time.monotonic() - t0) * 1000))

    def transcribe(self, image_bgr: np.ndarray, ocr_hint: str, hint: PageHint | None = None,
                   focus: FocusViews | None = None) -> LLMResult:
        import anthropic
        t0 = time.monotonic()
        try:
            try:
                return self._call(image_bgr, ocr_hint, hint, t0, focus)
            except anthropic.APIError as e:  # mid-stream errors surface as APIError, not BadRequestError
                if "content filtering" in str(e) and ocr_hint:
                    # the noisy OCR draft can trip the output filter; retry from the image alone
                    log.info("content filter tripped with OCR hint; retrying from the image alone")
                    return self._call(image_bgr, "", hint, t0, focus)
                raise
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMUnavailable(f"anthropic credentials rejected: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMUnavailable(f"model {self.model} not found: {e.message}") from e
        except anthropic.RateLimitError as e:
            body = e.body if isinstance(e.body, dict) else {}
            if os.environ.get("ANTHROPIC_AUTH_TOKEN") and body.get("error", {}).get("message") == "Error":
                # subscription OAuth tokens reach only some models via the API; this bare 429 is that case
                raise LLMUnavailable(f"{self.model} is not available to this OAuth token via the API; "
                                     "set OCRMD_MODEL=claude-haiku-4-5 (or use an API key)") from e
            raise LLMUnavailable(f"rate limited after retries: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMUnavailable(f"Anthropic API error {e.status_code} after retries") from e
            raise LLMPageError(f"API rejected the request ({e.status_code}): {e.message}") from e
        except anthropic.APIError as e:
            raise LLMPageError(str(e)[:300]) from e

    def _call(self, image_bgr, ocr_hint, hint, t0, focus=None) -> LLMResult:
        with self.client.beta.messages.stream(**self._params(image_bgr, ocr_hint, hint, focus=focus)) as stream:
            msg = stream.get_final_message()
        return self._result(msg, t0)

    def _result(self, msg, t0: float) -> LLMResult:
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise LLMPageError(f"refusal: {getattr(details, 'category', None)}")
        if msg.stop_reason == "max_tokens":
            raise LLMPageError("transcription truncated at max_tokens")
        text = "".join(b.text for b in msg.content if b.type == "text")
        tr = parse_transcription(text)
        return LLMResult(transcription=tr, provider=self.name, model=msg.model, request_id=msg.id,
                         usage=msg.usage.model_dump(), latency_ms=int((time.monotonic() - t0) * 1000))

    # --- batches: half price, hours of latency; for `ocrmd verify --batch` backlogs -------------------
    def batch_submit(self, items: list[tuple[str, np.ndarray, str, PageHint | None, "FocusViews | None"]]) -> str:
        import anthropic
        try:
            batch = self.client.messages.batches.create(requests=[
                {"custom_id": pid, "params": self._params(img, ocr_hint, hint, batch=True, focus=focus)}
                for pid, img, ocr_hint, hint, focus in items])
        except anthropic.APIStatusError as e:
            raise LLMUnavailable(f"batch submit failed ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {e}") from e
        return batch.id

    def batch_status(self, batch_id: str) -> tuple[str, dict]:
        b = self.client.messages.batches.retrieve(batch_id)
        counts = b.request_counts.model_dump() if hasattr(b.request_counts, "model_dump") else dict(b.request_counts)
        return b.processing_status, counts

    def batch_collect(self, batch_id: str) -> dict[str, LLMResult | Exception]:
        out: dict[str, LLMResult | Exception] = {}
        for r in self.client.messages.batches.results(batch_id):
            kind = r.result.type
            if kind == "succeeded":
                try:
                    out[r.custom_id] = self._result(r.result.message, time.monotonic())
                except LLMError as e:
                    out[r.custom_id] = e
            elif kind == "errored":
                out[r.custom_id] = LLMPageError(f"batch item errored: {getattr(r.result.error, 'type', r.result.error)}")
            else:
                out[r.custom_id] = LLMUnavailable(f"batch item {kind}; resubmit")
        return out


# --- xAI (OpenAI-compatible chat completions with a JSON schema response) ---------------------------
class XAIProvider:
    """Grok vision models through xAI's OpenAI-compatible endpoint. API key only (XAI_API_KEY); xAI does
    not offer third-party OAuth. Implemented against the documented request shape; verify against a
    live key before relying on it — it is not exercised by the test suite beyond the request builder."""
    name = "xai"
    endpoint = "https://api.x.ai/v1/chat/completions"

    def __init__(self, model: str = "grok-4", timeout: float = 180.0, max_retries: int = 3, effort: str = "medium"):
        self.model, self.timeout, self.max_retries = model, timeout, max_retries
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise LLMUnavailable("XAI_API_KEY is not set")

    def request_body(self, image_bgr: np.ndarray, ocr_hint: str, hint: PageHint | None,
                     focus: FocusViews | None = None) -> dict:
        content: list[dict] = [{"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(image_bgr)}", "detail": "high"}}]
        for i, extra in enumerate((focus.images if focus else []), 1):
            content.append({"type": "text", "text": f"Alternative exposure {i} of the SAME page (different focus):"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(extra)}", "detail": "high"}})
        content.append({"type": "text", "text": user_text(ocr_hint, hint, focus)})
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "transcription", "schema": TRANSCRIPTION_SCHEMA, "strict": True}},
            "max_tokens": 16000,
        }

    def _post(self, body: dict) -> dict:
        """One request with the retry/backoff policy, shared by every call this provider makes."""
        import urllib.request, urllib.error
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise LLMUnavailable(f"xAI credentials rejected ({e.code})") from e
                if e.code in (408, 409, 429) or e.code >= 500:
                    if attempt < self.max_retries:
                        time.sleep(delay); delay *= 2; continue
                    raise LLMUnavailable(f"xAI API error {e.code} after retries") from e
                raise LLMPageError(f"xAI rejected the request ({e.code}): {e.read()[:200]!r}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.max_retries:
                    time.sleep(delay); delay *= 2; continue
                raise LLMUnavailable(f"cannot reach the xAI API: {e}") from e
        raise LLMUnavailable("xAI request failed")           # unreachable; keeps the type honest

    def json_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        return {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "result", "schema": schema, "strict": True}},
                "max_tokens": max_tokens}

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 4000,
                      image_bgr: np.ndarray | None = None) -> JsonResult:
        t0 = time.monotonic()
        body = self.json_body(system, user, schema, max_tokens)
        if image_bgr is not None:
            body["messages"][1]["content"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(image_bgr)}", "detail": "high"}},
                {"type": "text", "text": user}]
        data = self._post(body)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "content_filter":
            raise LLMPageError("xAI content filter")
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        try:
            out = json.loads(text)
        except ValueError as e:
            raise LLMPageError(f"model returned text that is not JSON: {text[:200]!r}") from e
        return JsonResult(data=out, provider=self.name, model=data.get("model", self.model),
                          request_id=data.get("id"),
                          usage={"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
                          latency_ms=int((time.monotonic() - t0) * 1000))

    def transcribe(self, image_bgr: np.ndarray, ocr_hint: str, hint: PageHint | None = None,
                   focus: FocusViews | None = None) -> LLMResult:
        t0 = time.monotonic()
        data = self._post(self.request_body(image_bgr, ocr_hint, hint, focus))
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "content_filter":
            raise LLMPageError("xAI content filter")
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResult(transcription=parse_transcription(text), provider=self.name, model=data.get("model", self.model),
                         request_id=data.get("id"),
                         usage={"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
                         latency_ms=int((time.monotonic() - t0) * 1000))


def get_provider(name: str, model: str, timeout: float = 180.0, max_retries: int = 3, effort: str = "medium") -> Provider:
    if name == "anthropic":
        return AnthropicProvider(model, timeout=timeout, max_retries=max_retries, effort=effort)
    if name == "xai":
        return XAIProvider(model, timeout=timeout, max_retries=max_retries, effort=effort)
    raise ValueError(f"unknown provider {name}")


def credentials_available(name: str) -> bool:
    if name == "anthropic":
        from . import config  # noqa: F401  (loads .env, maps CLAUDE_CODE_OAUTH_TOKEN)
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        from pathlib import Path
        prof = Path.home() / ".config" / "anthropic"
        return prof.exists() and any(prof.iterdir())
    return bool(os.environ.get("XAI_API_KEY"))
