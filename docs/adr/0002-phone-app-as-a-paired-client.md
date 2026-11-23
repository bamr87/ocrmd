# ADR 0002 — The phone app is a paired client of the operator's own ocrmd

- **Status**: proposed
- **Date**: 2026-11-23
- **Deciders**: the maintainer (on merge); proposed by an AI agent
- **Related**: [docs/mobile.md](../mobile.md), [ARCHITECTURE.md §13b](../ARCHITECTURE.md#13b-the-phone-app), [constitution §1](../constitution.md#1-missions), features OCRMD-012 and OCRMD-013, PRD FR7 and NFR4

## Context

The request was to redesign the repository so that ocrmd can have a mobile app for Android and iOS. A phone is the camera most people already own, and it is a better one than the webcam the auto-shutter was tuned on (a C525 at ~1080p; a current phone shoots 12–48 MP with real autofocus), so the app is mainly a capture device and a review screen you can hold.

The forces:

- **The cross-check is the product, and it is Python on a computer.** Every page is read by local OCR engines (Apple Vision through pyobjc, RapidOCR on ONNX Runtime) and by a vision model, then diffed; most constants in `preprocess.py`, `reconcile.py` and `focus.py` were measured on real captures. A second implementation on a phone would drift from the measured one, and Android has no Apple Vision at all, so the "two engines vote" half could not be reproduced there.
- **Local first, and a tight rule for what leaves the machine** (constitution §1, mission 3): page images go only to the configured provider, and nothing else leaves the machine.
- **One inventory.** The library, the search index and every page's provenance live in one SQLite store and one directory tree on the operator's computer.
- **The server had no authentication** and refused to bind anything but loopback unless `--allow-remote` was passed (with a warning that anyone on the network could then read the scans). A phone can only reach the computer over the network.
- **An installed app cannot be updated in step with the server.** The browser SPA is served by the same process as its API, so `/api` can change freely; an app from a store will meet servers older and newer than itself.
- AGENTS.md: gating and processing stay identical between drivers, and new behaviour goes in the core, never in a driver.

## Decision

1. **The phone is a client of the operator's own ocrmd server, not a second pipeline.** It photographs pages and uploads them; it shows the library, the pages and their disagreements, and lets a person resolve, edit and sign off. Every read, reconciliation, artifact and row stays on the computer, produced by the same `ingest` job a drag-and-drop in the browser starts. Nothing is OCR'd on the phone and the phone never talks to the model provider.
2. **A versioned, typed API for it: `/api/v1`** (`ocrmd/web/v1.py`). Every answer is a pydantic model; `tools/gen_reference.py` writes the OpenAPI document to `docs/reference/openapi-v1.json` (and a readable `api.md`), and the app's TypeScript types are generated from that file (`npm run gen:api`), so a server change that would break the app fails the app's type check in CI. Changes to v1 are additive; a breaking change is `/api/v2`. The browser keeps `/api`.
3. **Pairing, not passwords.** `ocrmd serve --lan` opens the port to the local network, and from then on every request that does not come from the computer itself must carry a paired device's bearer token (`ocrmd/web/auth.py`). A token is issued once, in exchange for a one-time 8-character code made on the computer (`ocrmd pair` or Settings → Phones & tablets), valid 10 minutes; ten wrong codes in a row void every open code. Only SHA-256 hashes of codes and tokens are stored (`devices` table); a token is never printed or logged. Listing, pairing and revoking devices answer the computer only. Loopback behaves exactly as before (no checks), and `--allow-remote` stays as the explicit, warned no-authentication mode.
4. **Shared write paths.** A person's writes to a page (edit, resolve, sign off, reopen) moved out of the web routes into `ocrmd/edits.py`, used by `/api`, `/api/v1` and `api.Library`, so the rule that makes an edit *human* (the pipeline fingerprint is left alone) is stated once. What a client is shown of a page or document moved to `ocrmd/web/views.py`.
5. **The app: Expo (React Native, TypeScript), SDK 58**, in `mobile/` — one codebase for iOS and Android. `expo-camera` gives the camera, the QR scanner for pairing and the system document scanner (VisionKit / ML Kit); `expo-file-system` keeps photos and the upload queue in the app's documents directory and uploads natively from disk; `expo-secure-store` keeps the token in the Keychain / Keystore. Shooting never waits for the network: each photo is queued with a client id, uploaded when the computer is reachable, retried with back-off, and deleted from the phone only once the server has said what it became. A retry is the same job (client id) or, after a server restart, a recognised duplicate (the capture's sha256).
6. **Provenance names the phone.** A capture's `source` is `mobile:<device>`, or `mobile_scan:<device>` when the phone's document scanner already cropped it; the page's `source.type` is `mobile` or `mobile_scan`.

## Consequences

**Positive**

- One pipeline, one inventory, one set of measured constants; the phone adds a camera and a screen without adding a second reader.
- Mission 3 holds unchanged: images still go to the configured provider only, from the computer only. The phone sends photos *to* the operator's machine, and the server answers only devices the operator paired from that machine.
- The browser UI on the computer is unaffected; the CLI gains `serve --lan`, `pair` and `devices`.
- The app can be tried with no native toolchain (Expo Go), and its logic (pairing links, the API client, the upload queue) is tested on Linux CI with Jest, against types generated from the server.

**Negative**

- The computer must be on and reachable for anything to be *read*; the phone only queues. A phone away from home cannot reach a computer on a home LAN without a VPN overlay (Tailscale, WireGuard), which ocrmd does not set up.
- Traffic on the LAN is plain HTTP: a token is sent in clear over Wi-Fi. Acceptable on a WPA2/3 home network or an encrypted overlay, not on open Wi-Fi; HTTPS with a pinned self-signed certificate is the follow-up if this is used more widely. iOS needs `NSAllowsLocalNetworking` and Android `usesCleartextTraffic` for production builds (both in `mobile/app.json`).
- A reverse proxy or tunnel running on the computer that does not set `X-Forwarded-For` would make every request look local; this is documented, not enforced.
- A second toolchain (Node, Expo) and ~1 GB of `node_modules` for anyone working on the app; the Python side does not need it. No lockfile is committed (repo rule); exact versions are pinned instead, and `expo-doctor` reports the missing lockfile as its one failed check.
- Store builds (EAS Build or `expo prebuild` + Xcode / Android Studio), signing and store listings are not set up here.

## Alternatives considered

- **Port the pipeline to the phone** (on-device OCR with Apple Vision / ML Kit, a vision model called from the phone, a local store) — two implementations of every measured decision, no RapidOCR-plus-Vision vote on Android, a second inventory to reconcile, and the provider key on a phone. Rejected. On-device OCR remains a possible *additional* reading later, uploaded as evidence beside the photo, never in place of the computer's.
- **A mobile web page served by ocrmd** — no install, but browsers only open the camera (`getUserMedia`) on a secure origin, and `http://192.168.x.x` is not one; `<input capture>` works but gives no offline queue, no keychain and no document scanner. Kept as a possible lightweight complement.
- **Capacitor around the existing SPA** — the SPA is a desktop UI (MJPEG preview, wide tables, drag-a-box) served by the same process as its private API; wrapping it would ship that coupling to a store.
- **Flutter** — comparable reach, but Dart, a heavier SDK in CI, and no zero-build way to try it on a phone; TypeScript also lets the contract be generated straight from OpenAPI.
- **Native Swift and Kotlin apps** — the best camera and scanner integration, at twice the code for a single-operator tool. `expo-camera` already exposes the system document scanners.
- **A cloud relay** so the phone works anywhere — contradicts local-first and the PRD's "no cloud hosting" stance; not proposed.
- **Passwords or a shared secret in `ocrmd.toml`** — a long-lived secret typed on a phone and shared across devices, with no per-device revocation.
- **mDNS / Bonjour discovery** — the QR code already carries the address; discovery can be added later without changing the pairing contract.

## Supersedes / Superseded by

- Supersedes: none
- Superseded by: none
