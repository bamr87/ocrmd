import { ApiError, OcrmdClient, pairWith } from '../src/api/client';

type Call = { url: string; init?: RequestInit };

function fakeFetch(routes: Record<string, { status?: number; body?: unknown } | Error>) {
  const calls: Call[] = [];
  const impl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const key = Object.keys(routes).find((k) => url.endsWith(k));
    const r = key ? routes[key] : { status: 404, body: { detail: 'Not Found' } };
    if (r instanceof Error) throw r;
    const text = r?.body === undefined ? '' : JSON.stringify(r.body);
    return new Response(text, { status: r?.status ?? 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  return { impl, calls };
}

const PING = { ocrmd: true, api: 1, version: '0.2.0', access: 'paired', local: false, paired: false };

describe('OcrmdClient', () => {
  it('sends the device token and resolves server-relative paths', async () => {
    const { impl, calls } = fakeFetch({ '/api/v1/documents': { body: [] } });
    const c = new OcrmdClient('http://192.168.1.5:8765', 'ocrmd_tok', impl);
    await c.documents();
    expect(calls[0]?.url).toBe('http://192.168.1.5:8765/api/v1/documents');
    expect((calls[0]?.init?.headers as Record<string, string>).Authorization).toBe('Bearer ocrmd_tok');
    expect(c.url('/api/v1/documents/b/pages/1/image/processed?max=360')).toBe('http://192.168.1.5:8765/api/v1/documents/b/pages/1/image/processed?max=360');
    expect(c.imageUrl('my book', 3, 'flat', 1600)).toBe('http://192.168.1.5:8765/api/v1/documents/my%20book/pages/3/image/flat?max=1600');
  });

  it("turns the server's `detail` into the error, and an unreachable server into status 0", async () => {
    const { impl } = fakeFetch({
      '/signoff': { status: 409, body: { detail: '1 disagreement(s) still unresolved' } },
      '/documents': new TypeError('Network request failed'),
    });
    const c = new OcrmdClient('http://h:8765', 't', impl);
    await expect(c.signOff('b', 1)).rejects.toMatchObject({ status: 409, detail: '1 disagreement(s) still unresolved', transient: false });
    const down = await c.documents().catch((e: ApiError) => e);
    expect(down).toBeInstanceOf(ApiError);
    expect(down).toMatchObject({ status: 0, transient: true });
  });

  it('flags a revoked token as unpaired', async () => {
    const { impl } = fakeFetch({ '/documents': { status: 401, body: { detail: 'pair this device first' } } });
    const e = await new OcrmdClient('http://h:8765', 't', impl).documents().then(() => null, (x: ApiError) => x);
    expect(e?.unpaired).toBe(true);
  });
});

describe('pairWith', () => {
  it('checks the server speaks this API, then trades the code for a token', async () => {
    const paired = { token: 'ocrmd_new', device: { id: 'd1', name: 'Pixel' }, server: {} };
    const { impl, calls } = fakeFetch({ '/api/v1/ping': { body: PING }, '/api/v1/pair': { body: paired } });
    const { client } = await pairWith('http://h:8765', 'ABCD2345', 'Pixel', 'android', impl);
    expect(client.token).toBe('ocrmd_new');
    expect(JSON.parse(String(calls[1]?.init?.body))).toEqual({ code: 'ABCD2345', name: 'Pixel', platform: 'android' });
  });

  it('refuses a server on another API version before sending the code', async () => {
    const { impl, calls } = fakeFetch({ '/api/v1/ping': { body: { ...PING, api: 2 } } });
    await expect(pairWith('http://h:8765', 'ABCD2345', 'Pixel', 'ios', impl)).rejects.toThrow(/update the app/);
    expect(calls).toHaveLength(1);
  });
});
