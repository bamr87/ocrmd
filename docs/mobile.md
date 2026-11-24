# The phone app

> Scanning and reviewing from an iPhone or Android phone: pairing it with the ocrmd on your computer, shooting pages with the camera or the system document scanner, the upload queue, review on the phone, the security model, and building the app.

The phone is a camera and a screen for the ocrmd on your computer, not a second ocrmd. It photographs pages and sends them to the computer, which reads every one twice and reconciles the readings exactly as it does a webcam frame or a dropped file; the library, the page files and every artifact stay on the computer. The phone never OCRs anything and never talks to the model provider. Why it is built this way: [ADR 0002](adr/0002-phone-app-as-a-paired-client.md).

## On the computer

```sh
.venv/bin/pip install -e '.[dev,rapidocr,mobile]'   # mobile: QR codes for pairing (segno); everything else is core
.venv/bin/ocrmd serve --lan                          # the web UI, and the phone's API, open to the local network
.venv/bin/ocrmd pair                                 # a one-time code and a QR code for the phone
```

`serve --lan` binds every interface (or the one `--host` names) and prints the address a phone should dial. From then on, this computer's own browser works as it always has, and **every request from another machine needs a paired device's token** or it is refused (401). Without `--lan` the server answers this computer only, as before.

`ocrmd pair` prints a QR code, the server address and an 8-character code such as `KVD6-R7QN`. The code works once and for 10 minutes; ten wrong guesses in a row cancel every open code. The same is in the web UI under **Settings → Phones & tablets**, which also lists paired devices and revokes them. From a terminal:

```sh
ocrmd devices                       # paired phones, and codes still waiting
ocrmd devices --revoke "Kitchen iPad"  # by name or id; its next request is refused
```

If the phone cannot connect, check that both are on the same network (guest Wi-Fi often isolates devices), and on macOS allow incoming connections for Python when the firewall asks.

## On the phone

1. **Pair.** Open the app, tap **Scan the QR code** and point it at the code (or point the phone's own camera at it: the QR code is an `ocrmd://pair?…` link that opens the app), or type the server address and the code. The phone is listed on the computer under its own name unless you named it with `ocrmd pair --name`.
2. **Start a document** (**New document**: a title, optional authors, and a slug — the folder name under `documents/` on the computer), or pick an existing one.
3. **Shoot.** **Camera** takes one photo per press, at full resolution, and sends it as taken: the computer finds the page, splits a spread and reads it like any other capture. **Scanner** (when the phone has one: VisionKit on iOS, ML Kit on Android) finds, flattens and crops each page itself, several per session; such pages are marked `source.type: mobile_scan` because the scanner's crop, not the camera frame, becomes the raw capture. **OCR only** skips the model for these shots (fill it in later with **Verify** or `ocrmd verify`). Nothing you deliberately shoot is refused as unreadable — that test is only for the webcam's automatic shutter.
4. **See what each shot became.** The strip at the bottom of the camera says, per photo: *page 12 · verified*, *page 13 · flagged*, *already scanned: lsat page 9*, or why it failed; tap one to open the page.
5. **Review.** A document shows its pages as thumbnails coloured by verdict, filterable to *flagged*, *unverified* or *re-shoot*. A page shows the photograph (flat, processed or raw; tap to zoom) above the text, with each disagreement as a card: **OCR**, **Model**, or type what the page says. Digits are never decided for you. With none left open, **Sign off**; it then jumps to the next flagged page. **Edit** changes the text itself. Every write is a human edit: it survives the next `verify` on the computer, exactly like an edit in the web UI or in the file.
6. **Search** looks across every page of every document.

### Offline

Shooting never waits for the network. Each photo is saved in the app's own storage and queued; the queue uploads one at a time while the app is open, backs off while the computer is unreachable (2 s doubling to 5 min), and deletes a photo from the phone only after the computer has answered for it. **Settings** lists the queue, with **Retry now** and **Delete photo** for anything stuck. A retried upload is never a second page: the app sends each photo with an id, so a retry returns the same job, and should the computer have restarted in between, the capture's own hash marks it as already scanned.

What it does not do yet: upload while the app is in the background, or wake the computer.

## Security

- **Who can connect.** Loopback (this computer) without a token; another machine only with a paired device's token, and only when you started `serve --lan`. Listing, pairing and revoking devices answer this computer only — a paired phone cannot pair another. `GET /api/v1/ping` and `POST /api/v1/pair` are the only routes open without a token.
- **What is stored.** The computer keeps a SHA-256 hash of each token and code, never the value (`devices` in `ocrmd.db`). The phone keeps its token in the iOS Keychain or Android Keystore. Tokens start `ocrmd_` so a secret scanner recognises one that leaks.
- **The network.** The connection is plain HTTP on your local network, so a token crosses Wi-Fi unencrypted. Use it on your own WPA2/3 network, or put the phone and the computer on an encrypted overlay such as Tailscale or WireGuard (`ocrmd serve --lan --host <the overlay address>`, then `ocrmd pair --host <the same address>`). Do not use it on open Wi-Fi. A reverse proxy or tunnel running on the computer without `X-Forwarded-For` would make every request look local; do not put one in front of `--lan`.
- **What leaves the computer** is unchanged: page images go to the model provider you configured, and nothing else; the phone receives what you could see in the web UI, and only once paired.

## Building and developing the app

The app is [Expo](https://expo.dev) (React Native, TypeScript, SDK 58) in [`mobile/`](../mobile/README.md):

```sh
cd mobile
npm install                # no lockfile is committed (AGENTS.md); package.json pins exact versions
npx expo start             # then open it on a phone with Expo Go, or press i / a for a simulator
npm run check              # regenerate the API types, type-check, run the tests (what CI runs)
```

On a simulator on the same computer, pair with `ocrmd pair --host 127.0.0.1` (the iOS simulator) or `--host 10.0.2.2` (the Android emulator) against a plain `ocrmd serve`. Release builds go through EAS Build (`eas build -p ios` / `-p android`, which needs an Expo account) or `npx expo prebuild` and then Xcode / Android Studio; signing and store listings are not set up in this repository.

**The contract.** The app speaks `/api/v1` only (`ocrmd/web/v1.py`, reference: [reference/api.md](reference/api.md)). After changing it:

```sh
.venv/bin/python tools/gen_reference.py --write      # docs/reference/openapi-v1.json and api.md
cd mobile && npm run gen:api && npm run typecheck     # the app's types, generated from that JSON
```

`tests/test_reference_docs.py` fails while the JSON is stale, and CI's `mobile` job fails while the generated types are, or while the app no longer type-checks against them. Changes to v1 are additive; anything an installed app would trip over is `/api/v2`.
