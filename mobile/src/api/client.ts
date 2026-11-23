/**
 * The ocrmd server, as the app sees it: `/api/v1` with a device token.
 *
 * Every type here comes from `schema.d.ts`, which is generated from the server's OpenAPI document
 * (`npm run gen:api` ← docs/reference/openapi-v1.json ← ocrmd/web/v1.py). A change on the server that
 * would break this app fails `npm run typecheck` before it reaches a phone.
 */
import type { components } from './schema';

export type S = components['schemas'];
export type Ping = S['Ping'];
export type ServerInfo = S['ServerInfo'];
export type Device = S['Device'];
export type PairOut = S['PairOut'];
export type Document = S['Document'];
export type DocumentDetail = S['DocumentDetail'];
export type PageRow = S['PageRow'];
export type Page = S['Page'];
export type Flag = S['Flag'];
export type Job = S['Job'];
export type CaptureResult = S['CaptureResult'];
export type Hit = S['Hit'];
export type NewDocument = S['NewDocument'];
export type ImageKind = 'flat' | 'processed' | 'bin' | 'raw';

export const API_VERSION = 1;
const TIMEOUT_MS = 15_000;

/** A request the server answered with an error, or one that never reached it (`status` 0). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = 'ApiError';
  }

  /** The server no longer knows this device (revoked, or the data was reset): pair again. */
  get unpaired(): boolean {
    return this.status === 401;
  }

  /** Worth retrying later: the server was unreachable, timed out, or failed on its side. */
  get transient(): boolean {
    return this.status === 0 || this.status === 408 || this.status === 429 || this.status >= 500;
  }
}

type Fetch = typeof fetch;

export class OcrmdClient {
  constructor(
    readonly server: string,
    readonly token: string | null = null,
    private readonly fetchImpl: Fetch = (...a) => fetch(...a),
  ) {}

  /** An absolute URL for a path the server handed back (`images`, `thumb`, `cover`). */
  url(path: string): string {
    return /^https?:\/\//.test(path) ? path : `${this.server}${path.startsWith('/') ? '' : '/'}${path}`;
  }

  /** Headers for anything fetched outside this client: images, the capture upload. */
  headers(): Record<string, string> {
    return this.token ? { Authorization: `Bearer ${this.token}` } : {};
  }

  withToken(token: string): OcrmdClient {
    return new OcrmdClient(this.server, token, this.fetchImpl);
  }

  private async request<T>(method: string, path: string, body?: unknown, timeoutMs = TIMEOUT_MS): Promise<T> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    let res: Response;
    try {
      res = await this.fetchImpl(this.url(`/api/v1${path}`), {
        method,
        headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...this.headers() },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ctl.signal,
      });
    } catch (e) {
      const why = ctl.signal.aborted ? 'timed out' : e instanceof Error ? e.message : String(e);
      throw new ApiError(0, `cannot reach ${this.server}: ${why}`);
    } finally {
      clearTimeout(timer);
    }
    const text = await res.text();
    let data: unknown = undefined;
    try {
      data = text ? JSON.parse(text) : undefined;
    } catch {
      data = undefined;
    }
    if (!res.ok) {
      const d = (data as { detail?: unknown } | undefined)?.detail;
      const detail = typeof d === 'string' ? d : d ? JSON.stringify(d) : text || res.statusText || `HTTP ${res.status}`;
      throw new ApiError(res.status, detail);
    }
    return data as T;
  }

  // --- who is there -----------------------------------------------------------------------------
  ping = () => this.request<Ping>('GET', '/ping', undefined, 5_000);
  pair = (code: string, name: string, platform: string) => this.request<PairOut>('POST', '/pair', { code, name, platform });
  me = () => this.request<Device>('GET', '/me');
  serverInfo = () => this.request<ServerInfo>('GET', '/server');

  // --- the library ------------------------------------------------------------------------------
  documents = () => this.request<Document[]>('GET', '/documents');
  createDocument = (doc: NewDocument) => this.request<Document>('POST', '/documents', doc);
  document = (slug: string) => this.request<DocumentDetail>('GET', `/documents/${encodeURIComponent(slug)}`);
  search = (q: string, doc?: string) =>
    this.request<Hit[]>('GET', `/search?q=${encodeURIComponent(q)}${doc ? `&doc=${encodeURIComponent(doc)}` : ''}`);

  // --- a page ---------------------------------------------------------------------------------------
  private pagePath = (slug: string, seq: number) => `/documents/${encodeURIComponent(slug)}/pages/${seq}`;
  page = (slug: string, seq: number) => this.request<Page>('GET', this.pagePath(slug, seq));
  editPage = (slug: string, seq: number, edit: S['PageEditIn']) => this.request<Page>('PUT', this.pagePath(slug, seq), edit);
  resolve = (slug: string, seq: number, n: number, chosen: string) =>
    this.request<Page>('POST', `${this.pagePath(slug, seq)}/resolve`, { n, chosen });
  signOff = (slug: string, seq: number, who?: string) =>
    this.request<Page>('POST', `${this.pagePath(slug, seq)}/signoff`, who ? { who } : {});
  reopen = (slug: string, seq: number) => this.request<Page>('POST', `${this.pagePath(slug, seq)}/reopen`);
  imageUrl = (slug: string, seq: number, kind: ImageKind, max?: number) =>
    this.url(`/api/v1${this.pagePath(slug, seq)}/image/${kind}${max ? `?max=${max}` : ''}`);

  // --- captures and jobs -------------------------------------------------------------------------------
  captureUrl = (slug: string) => this.url(`/api/v1/documents/${encodeURIComponent(slug)}/captures`);
  verify = (slug: string, opts: S['VerifyIn'] = {}) =>
    this.request<Job>('POST', `/documents/${encodeURIComponent(slug)}/verify`, opts);
  job = (id: string) => this.request<Job>('GET', `/jobs/${encodeURIComponent(id)}`);
  jobs = (doc?: string, active = false) =>
    this.request<Job[]>('GET', `/jobs?active=${active}${doc ? `&doc=${encodeURIComponent(doc)}` : ''}`);
}

/** Pair with a server: check it is ocrmd and speaks this app's API, then trade the code for a token. */
export async function pairWith(
  server: string,
  code: string,
  name: string,
  platform: string,
  fetchImpl?: Fetch,
): Promise<{ client: OcrmdClient; paired: PairOut }> {
  const anon = new OcrmdClient(server, null, fetchImpl);
  const ping = await anon.ping();
  if (!ping.ocrmd) throw new ApiError(0, `${server} is not an ocrmd server`);
  if (ping.api !== API_VERSION) {
    throw new ApiError(0, `${server} speaks API v${ping.api}; this app speaks v${API_VERSION} — update ${ping.api > API_VERSION ? 'the app' : 'ocrmd on the computer'}`);
  }
  const paired = await anon.pair(code, name, platform);
  return { client: anon.withToken(paired.token), paired };
}
