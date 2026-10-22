# Setup

> What a working install needs beyond `pip install`: macOS camera permission, Claude credentials and which models they reach, the local OCR engines, and where settings come from.

The install commands are in the README's [Quick start](../README.md#quick-start). This page covers what they assume.

## Camera, credentials and engines

**Camera permission:** macOS must authorize the app that runs `ocrmd` (Terminal, iTerm, VS Code…): System Settings → Privacy & Security → Camera. `ocrmd doctor` tells you if it's denied.

**Claude credentials** (put them in `.env` next to the project, or `~/.config/ocrmd/.env`):

| Variable | Notes |
|---|---|
| `ANTHROPIC_API_KEY` | Console API key. All models; default model is `claude-opus-5-5`. Needed for `verify --batch`. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code / subscription OAuth token (`sk-ant-oat…`). Works, but as of Aug 2026 the API only serves **`claude-haiku-5-5`** to it — set `OCRMD_MODEL=claude-haiku-5-5` too. Other models return a bare 429 `Error`; ocrmd reports this as "not available to this OAuth token". |
| `ANTHROPIC_AUTH_TOKEN` | Any other bearer token; same rules as above. |
| `XAI_API_KEY` | For `--provider xai` (Grok vision; implemented against xAI's OpenAI-compatible endpoint, not exercised live here). |

Without credentials ocrmd runs OCR-only and you can fill in the LLM pass later with `ocrmd verify`.

**Local OCR engines.** On macOS Apple Vision (through pyobjc, no extra install) is the primary engine; `pip install 'ocrmd[rapidocr]'` adds RapidOCR (PP-OCR on ONNX Runtime, models fetched on first use), which is the primary engine on Linux/Windows. When two engines are installed they both read every page and *vote*: words they agree on are trusted, words they disagree on are flagged for the LLM comparison, lines one missed are filled from the other (`ocr_ensemble = false` turns this off, or set `ocr_engines = ["apple_vision"]`). Tesseract (`ocrmd[tesseract]` + the binary) is supported but measured far behind on camera images. `ocrmd ocr-bench` measures whatever is installed against the LLM transcriptions already in your store, so you can see what a change buys before trusting it.

Page images are sent to the provider you configure; nothing else leaves the machine.

## Configuration

Every setting has a default, can be set in `ocrmd.toml` (project root) or `~/.config/ocrmd/config.toml`, and can be overridden by an `OCRMD_<KEY>` environment variable; CLI flags win over all of them. Every key, its `OCRMD_*` variable and its default is in the generated [settings reference](reference/settings.md); [ocrmd.example.toml](../ocrmd.example.toml) is a commented file to start from. Invalid values fail fast with the offending key named. `ocrmd doctor` prints the effective model/provider/focus/workers.
