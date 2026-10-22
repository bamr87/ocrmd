# Operations

> Running ocrmd day to day — degraded mode, costs, camera drop-outs, health checks, upgrades, logs, sharing one root — and debugging it in VS Code.

## Running it

- **Offline / service down.** After `llm_failure_limit` consecutive service failures (network, auth, quota — not page-level refusals) a scan drops to OCR-only, says so once, and keeps capturing. Every capture is on disk before anything else happens; `ocrmd verify --doc X` finishes the LLM pass later.
- **Costs.** `ocrmd doctor --costs` sums tokens per model. `verify --batch` sends a backlog through the Message Batches API at half price (`--no-wait` returns immediately; `verify --collect` fetches). A prompt or model change never re-runs pages by itself — `verify --refresh` is the explicit, paid step.
- **Camera drop-outs** are detected (no frames for a few seconds) and the device reopened; after five failures the scan ends cleanly and builds what it has.
- **Health.** `ocrmd doctor` checks the SQLite integrity, pages whose raw/processed files went missing, orphaned artifacts, open batches, camera focus state and the dictionary the orientation probe needs.
- **Upgrading.** Databases from 0.1 are migrated in place (additive columns). Bumped preprocessing (`preprocess@N`) is what `reprocess` is for; bumped prompts (`transcribe@N`) are what `verify --refresh` is for.
- **Logs.** `ocrmd.log` rotates at 2 MB × 3. HTTP client logs are kept at WARNING so tokens and request bodies never land in it.
- **Web UI and CLI share one database.** Run one `ocrmd serve` per root; `verify`/`build` from the CLI while the UI is open is fine (SQLite WAL + locking), but don't run two scans against one camera.

## Debugging in VS Code

[.vscode/launch.json](../.vscode/launch.json) has ready-made configurations: doctor, status, **serve (web UI)** — uvicorn runs in-process so breakpoints in `ocrmd/web/` and the pipeline fire from the browser — scan variants (C525 with settle focus, OCR-only, headless 60 s with debug logs, camera index 0), ingest, verify (missing / flagged only / `--refresh`), reprocess, review, build, pytest (all / current file) and the current file. They run the CLI in-process as a module so breakpoints work anywhere in the package, prompt for the document slug and camera, and load `.env` for credentials. [.vscode/tasks.json](../.vscode/tasks.json) adds lint + test (the default build task), serve, doctor, status and install; [.vscode/settings.json](../.vscode/settings.json) enables pytest in the Test Explorer and keeps the scan data directories out of search and file watching.
