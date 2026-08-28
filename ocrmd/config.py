"""Settings: defaults -> ocrmd.toml -> environment (OCRMD_*), plus logging setup.

Resolution order, lowest to highest precedence:
  1. the defaults on `Settings`
  2. `~/.config/ocrmd/config.toml`
  3. `<root>/ocrmd.toml`               (root = OCRMD_ROOT or the project directory)
  4. environment variables `OCRMD_<FIELD>` (e.g. OCRMD_MODEL, OCRMD_STABLE_FRAMES, OCRMD_RESOLUTION=1280x960)
  5. CLI options, applied by the command itself

Credentials come from `.env` (cwd, project root, then ~/.config/ocrmd/.env; first found wins and never
overrides a variable already in the environment).
"""
from __future__ import annotations
import logging, logging.handlers, os, tomllib
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

for _p in (Path.cwd() / ".env", _PROJECT_ROOT / ".env", Path.home() / ".config" / "ocrmd" / ".env"):
    if _p.exists():
        load_dotenv(_p, override=False)
        break
# A Claude Code OAuth token is a bearer token for the API; expose it under the name the SDK reads.
if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") and not (os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")):
    os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]


class Settings(BaseModel):
    # data lives in the repo itself (raw/, processed/, ocr/, llm/, documents/, ocrmd.db); OCRMD_ROOT overrides
    root: Path = Path(os.environ.get("OCRMD_ROOT", _PROJECT_ROOT))

    # --- webcam auto-shutter ----------------------------------------------------------------
    camera_index: int = 0
    analysis_width: int = 320
    motion_threshold: float = 4.0     # mean abs diff (0-255) below which frame is "still"; C525 sensor noise ~2.7
    stable_frames: int = 8            # undisturbed frames before capture, after ordinary noise
    # ...and after the scene was *handled* — a hand, or the page outline moving. A page turn has to
    # prove it has finished: the old gate fired during flips because every per-frame test can be true
    # halfway through one (see scene.py).
    settle_frames: int = 20
    geometry_frames: int = 3          # page-outline detections that must agree before the page counts as still
    geometry_tolerance: float = 0.03  # how far it may move between them, as a fraction of the frame
    novelty_threshold: int = 12       # dHash hamming distance vs last capture
    blur_threshold: float = 60.0      # absolute floor for page sharpness (Laplacian variance)
    sharp_ratio: float = 0.55         # ...and at least this fraction of the best this scene has shown
    sharp_decay: float = 0.995        # how fast that best fades, so the bar drops when the rig changes
    resolution: tuple[int, int] | None = None   # None = the camera's largest mode
    focus: str = "continuous"                   # continuous | locked | settle | off
    settle_after: int = 1                       # focus=settle: lock after this many captures
    # hand gate: fraction of the reference paper area that must vanish before the frame is "covered",
    # and how many stuck frames before a persistent intrusion is accepted as part of the scene
    occlusion_cover: float = 0.05
    occlusion_rebaseline: int = 30
    burst_frames: int = 3                       # frames read at the shutter; the sharpest is kept (1 = off)
    # focus bracket: shoot the same still page at several lens positions and keep the extra frames as
    # variants, so a region the sharpest frame cannot resolve can be re-read from a different focus.
    # Needs a camera that supports locking focus at a custom lens position (`ocrmd doctor` reports it).
    focus_bracket: str = "auto"        # auto (bracket only when the frame is not the scene's best) | always | off
    focus_bracket_shots: int = 3       # lens positions in the bracket, including the one the lens is at
    focus_bracket_step: float = 0.05   # lens-position offset between shots, on AVFoundation's 0-1 scale
    focus_bracket_settle: float = 0.25  # seconds to let the lens travel before the frame is read
    focus_bracket_ratio: float = 0.9   # auto: bracket when sharpness is below this share of the scene's best
    focus_keep: int = 2                # variant images kept per capture (0 = capture the bracket, keep none)
    capture_cooldown: float = 1.0               # seconds between auto captures
    camera_reopen_after: int = 40               # consecutive failed reads before the camera is reopened

    # --- admission (should a capture become a page at all?) ---------------------------------
    # Judged on what local OCR made of the crop, because image-quality signals measurably do not
    # separate legible from illegible pages (see legibility.py). Rejection keeps the raw frame and
    # creates nothing; it never applies to file ingest or to a forced (space-bar) capture.
    admit_captures: bool = True       # false: every auto capture becomes a page, as before 0.6
    admit_min_words: int = 60         # rejects 21 of the 120 scored pages, 20 of them truly illegible
    admit_min_confidence: float = 0.5

    # --- dedup ------------------------------------------------------------------------------
    near_duplicate_distance: int = 10       # dHash prefilter (bits), within the same document only...
    near_duplicate_similarity: float = 0.985  # ...confirmed by frame correlation; different pages measure <=0.98...
    near_duplicate_text: float = 0.85         # ...and by OCR token overlap, so two near-blank pages stay distinct...
    near_duplicate_detail: float = 0.5        # ...falling back to high-pass image correlation when there is no OCR text

    # --- annotation sync (camera on already-scanned pages) ----------------------------------
    annotate_match: float = 0.985   # correlation against a stored capture before it counts as the same page

    # --- preprocess -------------------------------------------------------------------------
    min_page_width: int = 900         # px across one page below which body text is unreliable
    # Decide which way up a page is by asking OCR at all four rotations. It is the only test that
    # works — every cheaper one measures at or near chance on real captures (see
    # preprocess.orientation_probe) — but it is OFF by default because on this machine the extra
    # Apple Vision traffic reliably segfaults the process inside RapidOCR's ONNX detector. That is a
    # third-party interaction, not an ocrmd bug: the same run with `ocr_engines = ["apple_vision"]`
    # is clean. Turn it on if you do not use rapidocr, or if your platform does not show the crash.
    orientation_probe: bool = False

    # --- ocr --------------------------------------------------------------------------------
    ocr_engines: list[str] = []       # [] = auto: the best installed engine (+ a second one when ocr_ensemble)
    ocr_ensemble: bool = True         # run two engines when installed and vote (see ocr.merge)
    ocr_languages: list[str] = ["en-US"]   # Vision recognition languages; [] = automatic
    ocr_min_side: int = 1400          # crops are upscaled (<=3x) until the short side reaches this
    focus_swap_margin: float = 0.10   # a variant focus must beat the primary's line confidence by this to replace it
    focus_llm_images: int = 2         # variant images attached to the LLM request when regions stay unresolved

    # --- reconcile --------------------------------------------------------------------------
    flag_threshold: float = 0.95
    ocr_trust: float = 0.90
    ocr_distrust: float = 0.60
    suspect_threshold: float = 0.35   # min token overlap; below it the sources aren't describing the same page

    # --- llm --------------------------------------------------------------------------------
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    xai_model: str = "grok-4"
    prompt_version: str = "transcribe@5"
    llm_workers: int = 2              # concurrent LLM calls during a scan (OCR stays serial)
    llm_timeout: float = 180.0        # seconds per request
    llm_max_retries: int = 3          # SDK-level retries on 429/5xx/connection errors
    llm_failure_limit: int = 3        # consecutive failures before a scan drops to OCR-only ("degraded")
    llm_effort: str = "medium"        # transcription is perception, not reasoning
    batch_poll_seconds: int = 30

    # --- logging ----------------------------------------------------------------------------
    log_file: str = "ocrmd.log"       # relative to root; "" disables
    log_level: str = "INFO"

    @field_validator("resolution", mode="before")
    @classmethod
    def _parse_resolution(cls, v: Any):
        if v in (None, "", "auto", "native"):
            return None
        if isinstance(v, str):
            w, h = v.lower().replace("×", "x").split("x")
            return (int(w), int(h))
        return tuple(v)

    @field_validator("ocr_engines", "ocr_languages", mode="before")
    @classmethod
    def _parse_list(cls, v: Any):
        if isinstance(v, str):
            return [x.strip() for x in v.replace("+", ",").split(",") if x.strip()]
        return v

    @field_validator("focus")
    @classmethod
    def _check_focus(cls, v: str) -> str:
        if v not in ("continuous", "locked", "settle", "off"):
            raise ValueError("focus must be continuous | locked | settle | off")
        return v

    @field_validator("focus_bracket")
    @classmethod
    def _check_bracket(cls, v: str) -> str:
        if v not in ("auto", "always", "off"):
            raise ValueError("focus_bracket must be auto | always | off")
        return v

    @property
    def raw(self) -> Path: return self.root / "raw"
    @property
    def processed(self) -> Path: return self.root / "processed"
    @property
    def ocr_dir(self) -> Path: return self.root / "ocr"
    @property
    def llm_dir(self) -> Path: return self.root / "llm"
    @property
    def layout_dir(self) -> Path: return self.root / "layout"
    @property
    def export_dir(self) -> Path: return self.root / "export"
    @property
    def documents(self) -> Path: return self.root / "documents"
    @property
    def db_path(self) -> Path: return self.root / "ocrmd.db"
    @property
    def log_path(self) -> Path | None: return (self.root / self.log_file) if self.log_file else None

    def ensure_dirs(self) -> None:
        for d in (self.raw, self.processed, self.ocr_dir, self.llm_dir, self.layout_dir, self.documents):
            d.mkdir(parents=True, exist_ok=True)


def _read_toml(p: Path) -> dict:
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise RuntimeError(f"cannot read {p}: {e}") from e
    # allow either flat keys or an [ocrmd] table
    return dict(data.get("ocrmd", data))


def env_overrides(env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ if env is None else env
    out = {}
    for name in Settings.model_fields:
        v = env.get(f"OCRMD_{name.upper()}")
        if v is not None and v != "":
            out[name] = v
    return out


def load_settings(root: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    root = Path(root or (env or os.environ).get("OCRMD_ROOT") or _PROJECT_ROOT)
    values: dict[str, Any] = {"root": root}
    for p in (Path.home() / ".config" / "ocrmd" / "config.toml", root / "ocrmd.toml"):
        if p.exists():
            values.update(_read_toml(p))
    values.update(env_overrides(env))
    values["root"] = root
    try:
        return Settings(**values)
    except ValidationError as e:
        raise RuntimeError(f"invalid ocrmd configuration: {e}") from e


settings = load_settings()


# --- logging -----------------------------------------------------------------------------------
def setup_logging(s: Settings, debug: bool = False, quiet: bool = False) -> logging.Logger:
    """Console shows warnings (everything with --debug); the log file keeps INFO+ with rotation.
    Third-party HTTP loggers stay at WARNING so tokens and request bodies never land in the file."""
    root = logging.getLogger("ocrmd")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else (logging.ERROR if quiet else logging.WARNING))
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    if s.log_path:
        try:
            s.log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(s.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setLevel(logging.DEBUG if debug else getattr(logging, s.log_level.upper(), logging.INFO))
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)
        except OSError as e:  # unwritable root: keep going with console only
            root.warning("log file unavailable: %s", e)
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root
