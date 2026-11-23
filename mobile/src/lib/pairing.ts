/**
 * Pairing: turning what the operator's computer shows (a QR code, or a server address and a code) into
 * something the app can dial. The computer makes the code (`ocrmd pair`, or Settings in the web UI); the
 * QR code carries `ocrmd://pair?server=<url>&code=<code>`, which is also the deep link the app registers.
 */

export const DEFAULT_PORT = 8765;
const CODE_ALPHABET = /^[ABCDEFGHJKMNPQRSTUVWXYZ23456789]{8}$/;

export type PairTarget = { server: string; code: string };

/** What a person typed (`192.168.1.5`, `mac.local:8765/`, `http://…`) as a base URL, or null if it is not one. */
export function normalizeServer(input: string): string | null {
  let s = input.trim();
  if (!s) return null;
  if (!/^https?:\/\//i.test(s)) s = `http://${s}`;
  let url: URL;
  try {
    url = new URL(s);
  } catch {
    return null;
  }
  if (!url.hostname) return null;
  const port = url.port || (/^https:/i.test(s) ? '' : String(DEFAULT_PORT));
  return `${url.protocol}//${url.hostname.includes(':') && !url.hostname.startsWith('[') ? `[${url.hostname}]` : url.hostname}${port ? `:${port}` : ''}`;
}

/** A code as the computer made it: case, spaces and the dash in `ABCD-EFGH` do not matter. */
export function normalizeCode(input: string): string {
  return input.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

export function isValidCode(code: string): boolean {
  return CODE_ALPHABET.test(normalizeCode(code));
}

/** `ABCDEFGH` → `ABCD-EFGH`, the way the computer shows it. */
export function displayCode(code: string): string {
  const c = normalizeCode(code);
  return c.length > 4 ? `${c.slice(0, 4)}-${c.slice(4)}` : c;
}

/** The QR code's text or a deep link → server + code, or null when it is not an ocrmd pairing link. */
export function parsePairLink(text: string): PairTarget | null {
  const m = text.trim().match(/^(?:exp\+)?ocrmd:\/\/\/?pair\?(.*)$/i);
  if (!m) return null;
  const params = new Map<string, string>();
  for (const part of (m[1] ?? '').split('&')) {
    const [k, v = ''] = part.split('=');
    if (k) params.set(decodeURIComponent(k), decodeURIComponent(v.replace(/\+/g, ' ')));
  }
  const server = normalizeServer(params.get('server') ?? '');
  const code = normalizeCode(params.get('code') ?? '');
  return server && isValidCode(code) ? { server, code } : null;
}
