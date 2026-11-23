# ocrmd for iOS and Android

> The phone app: shoot pages with the phone's camera or its document scanner, and review the cross-checked transcriptions, against the ocrmd running on your own computer.

How to use it — pairing, shooting, the upload queue, review, security — is in [docs/mobile.md](../docs/mobile.md); why it is a client and not a second pipeline is [ADR 0002](../docs/adr/0002-phone-app-as-a-paired-client.md).

```sh
npm install           # Node 22+; no lockfile is committed, versions are pinned in package.json
npx expo start        # Expo Go on a phone, or i / a for a simulator
npm run check         # what CI runs: regenerate src/api/schema.d.ts, type-check, unit tests
```

| Path | What lives there |
| --- | --- |
| `src/app/` | The screens (Expo Router, one file per route): library, pairing, capture, new document, document, page review, search, settings |
| `src/api/client.ts` | The `/api/v1` client; every type comes from `schema.d.ts` |
| `src/api/schema.d.ts` | Generated from `../docs/reference/openapi-v1.json` by `npm run gen:api` — do not edit |
| `src/lib/` | Logic with no React Native in it, unit-tested: pairing links, the upload queue, slugs, disagreement marks |
| `src/state/` | The session (token in the keychain) and the upload queue wired to the phone's storage |
| `__tests__/` | Jest tests for `src/lib` and the client |

The app talks to nothing but the ocrmd server it was paired with.
