"""Diff local OCR (A) against LLM transcription (B); decide, flag, score."""
from __future__ import annotations
import re, unicodedata
from difflib import SequenceMatcher
from pydantic import BaseModel

_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "­": ""})


class Flag(BaseModel):
    ocr: str
    llm: str
    chosen: str | None  # None => unresolved, marked in body


class Reconciled(BaseModel):
    markdown: str
    agreement: float      # order-sensitive: how much of the OCR text the LLM reproduced in sequence
    overlap: float        # order-insensitive: do the two sources describe the same page at all?
    cer: float
    flags: list[Flag]
    status: str  # verified | flagged | suspect | blank


def _content_tokens(words: list[str]) -> set[str]:
    """Long-ish alphabetic tokens only: robust to OCR noise, numbers and reading-order differences."""
    return {w for w in (re.sub(r"[^a-z]", "", x.lower()) for x in words) if len(w) >= 4}


def token_overlap(a: list[str], b: list[str]) -> float:
    """Containment of the smaller token set in the larger. 1.0 = one text's vocabulary is inside the other's.
    Order-insensitive on purpose: OCR engines read multi-column pages and tables of contents out of order,
    which makes a sequence diff say 0% for two transcriptions of the very same page."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_FOLD)


def _dehyphenate(s: str) -> str:
    """Join words broken at line ends: 'pos-\nsession' -> 'possession'."""
    return re.sub(r"(\w)-\s*\n\s*(?=[a-z])", r"\1", s)


def _words(s: str) -> list[str]:
    return re.findall(r"\S+", _dehyphenate(_norm(s)))


def _strip_md(s: str) -> str:
    s = re.sub(r"^#{1,6}\s+|^\s*[-*>]\s+|\[\^[^\]]+\]:?|[*_`|=]", "", s, flags=re.M)
    return s


def _is_numeric(w: str) -> bool:
    return bool(re.search(r"\d", w))


def reconcile(ocr_text: str, ocr_words: list[tuple[str, float]], llm_md: str,
              ocr_trust: float = 0.9, ocr_distrust: float = 0.6, flag_threshold: float = 0.95,
              suspect_threshold: float = 0.3) -> Reconciled:
    """Policy: the LLM read the image and owns structure; OCR is a spell-check with per-word confidence.
    - numeric mismatch (short span)            -> flag, unresolved
    - short span, OCR confident, near-identical -> take OCR
    - short span, OCR middling                 -> flag, unresolved
    - long span or OCR unconfident             -> trust LLM silently (OCR garbage, not evidence)
    Agreement = 1 - mismatched OCR chars / OCR chars, i.e. how much of what OCR saw the LLM reproduced.
    Status also uses an order-insensitive token overlap, because OCR reads columns and contents pages
    out of order and a pure sequence diff would call two readings of the same page completely different."""
    a = _words(ocr_text)
    b = _words(_strip_md(llm_md))
    conf = {w: c for w, c in ocr_words}
    sm = SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False)
    flags: list[Flag] = []
    edits: list[tuple[str, str, Flag]] = []   # (llm phrase, replacement, flag) in document order
    mismatched = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        pa, pb = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        mismatched += len(pa)
        if op != "replace" or not pb or (j2 - j1) > 3 or (i2 - i1) > 3:
            continue  # insert/delete or long span: LLM wins silently
        c = min((conf.get(w, 0.0) for w in a[i1:i2]), default=0.0)
        sim = SequenceMatcher(None, pa.lower(), pb.lower()).ratio()
        if any(_is_numeric(w) for w in a[i1:i2] + b[j1:j2]) and sim > 0.5:
            fl = Flag(ocr=pa, llm=pb, chosen=None)
        elif c >= ocr_trust and sim > 0.75:
            fl = Flag(ocr=pa, llm=pb, chosen=pa)
        elif c >= ocr_distrust and sim > 0.6:
            fl = Flag(ocr=pa, llm=pb, chosen=None)
        else:
            continue
        flags.append(fl)
        edits.append((pb, fl.chosen if fl.chosen else f"==[{pb}]==", fl))
    total = max(1, sum(len(w) for w in a))
    agreement = max(0.0, 1 - mismatched / total)
    # apply edits left-to-right with a moving cursor so replacements never re-match earlier output
    out, cursor, notes, n = [], 0, [], 0
    for src, dst, fl in edits:
        k = llm_md.find(src, cursor)
        if k < 0:
            continue
        if fl.chosen is None:
            n += 1
            dst = f"{dst}[^ocrmd-{n}]"
            notes.append(f"[^ocrmd-{n}]: ocr=\"{fl.ocr}\" llm=\"{fl.llm}\" — unresolved")
        out.append(llm_md[cursor:k]); out.append(dst); cursor = k + len(src)
    out.append(llm_md[cursor:])
    md = "".join(out)
    if notes:
        md = md.rstrip() + "\n\n" + "\n".join(notes) + "\n"
    unresolved = sum(1 for f in flags if f.chosen is None)
    overlap = token_overlap(a, b)
    if not _content_tokens(a) and not _content_tokens(b):
        status = "blank"          # both sources found no text: the page really is empty
        agreement = 1.0
    elif len(a) > 20 and overlap < suspect_threshold:
        # the two sources share almost no vocabulary: one of them is not describing this page
        # (typically a model inventing content for an image it could not read)
        status = "suspect"
    elif agreement >= flag_threshold and unresolved == 0:
        status = "verified"
    else:
        status = "flagged"
    return Reconciled(markdown=md, agreement=round(agreement, 4), overlap=round(overlap, 4),
                      cer=round(1 - agreement, 4), flags=flags, status=status)
