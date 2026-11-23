import { ApiError, type Job } from '../src/api/client';
import { backoff, describe as describeItem, type QueueItem, type QueueStorage, UploadQueue } from '../src/lib/queue';

function memoryStorage(initial: QueueItem[] = []) {
  let saved = initial;
  const discarded: string[] = [];
  const storage: QueueStorage = {
    load: async () => saved.map((i) => ({ ...i })),
    save: async (items) => { saved = items.map((i) => ({ ...i })); },
    discard: async (uri) => { discarded.push(uri); },
  };
  return { storage, discarded, saved: () => saved };
}

const job = (id: string, status: Job['status'], results: Job['results'] = []): Job => ({
  id, kind: 'ingest', doc: 'book', status, created_at: 0, started_at: null, ended_at: null, error: null, done: 0, total: 1, results, log: [],
});

const page = (clientId: string, seq = 1, status = 'verified') => ({
  file: 'x.jpg', client_id: clientId, pages: [{ sequence: seq, status }], duplicate_of: null, rejected: [], error: null,
});

describe('UploadQueue', () => {
  let now = 1_000;
  const clock = () => now;
  beforeEach(() => { now = 1_000; });

  it('uploads, follows the job, and deletes the photo once the server has answered', async () => {
    const mem = memoryStorage();
    const upload = jest.fn(async (it: QueueItem) => job('j1', 'queued'));
    const fetchJob = jest.fn(async () => job('j1', 'done', [page('a')]));
    const q = new UploadQueue(mem.storage, upload, fetchJob, { now: clock });
    await q.add({ id: 'a', doc: 'book', uri: 'file:///a.jpg', origin: 'camera', noLlm: false });
    await q.tick();
    expect(upload).toHaveBeenCalledTimes(1);
    expect(q.snapshot()[0]).toMatchObject({ state: 'processing', jobId: 'j1' });
    expect(mem.discarded).toEqual([]);
    await q.tick();
    const done = q.snapshot()[0]!;
    expect(done.state).toBe('done');
    expect(describeItem(done)).toBe('page 1 · verified');
    expect(mem.discarded).toEqual(['file:///a.jpg']);
  });

  it('waits and retries while the computer is unreachable, with growing back-off', async () => {
    const mem = memoryStorage();
    let reachable = false;
    const upload = jest.fn(async () => {
      if (!reachable) throw new ApiError(0, 'cannot reach http://h:8765: Network request failed');
      return job('j2', 'done', [page('b', 4, 'flagged')]);
    });
    const q = new UploadQueue(mem.storage, upload, async () => job('j2', 'done'), { now: clock });
    await q.add({ id: 'b', doc: 'book', uri: 'file:///b.jpg', origin: 'camera', noLlm: true });
    await q.tick();
    expect(q.snapshot()[0]).toMatchObject({ state: 'queued', attempts: 1, nextAttemptAt: 1_000 + backoff(1) });
    await q.tick(); // not due yet
    expect(upload).toHaveBeenCalledTimes(1);
    now += backoff(1);
    reachable = true;
    await q.tick();
    expect(q.snapshot()[0]).toMatchObject({ state: 'done' });
    expect(describeItem(q.snapshot()[0]!)).toBe('page 4 · flagged');
  });

  it('gives up at once on what retrying cannot fix, and says why', async () => {
    const mem = memoryStorage();
    const q = new UploadQueue(mem.storage, async () => { throw new ApiError(401, 'pair this device first'); }, async () => job('x', 'done'), { now: clock });
    await q.add({ id: 'c', doc: 'book', uri: 'file:///c.jpg', origin: 'camera', noLlm: false });
    await q.tick();
    expect(q.snapshot()[0]).toMatchObject({ state: 'failed', error: 'this phone is no longer paired; pair it again' });
    expect(mem.discarded).toEqual([]); // the photo is kept for a retry after pairing again
  });

  it('re-uploads when the server forgot the job, and reports a duplicate as done', async () => {
    const mem = memoryStorage();
    const fetchJob = jest.fn(async () => { throw new ApiError(404, 'no such job'); });
    const results = [{ ...page('d'), pages: [], duplicate_of: { document: 'book', sequence: 7 } }];
    const upload = jest.fn().mockResolvedValueOnce(job('old', 'running')).mockResolvedValueOnce(job('new', 'done', results));
    const q = new UploadQueue(mem.storage, upload, fetchJob, { now: clock });
    await q.add({ id: 'd', doc: 'book', uri: 'file:///d.jpg', origin: 'scanner', noLlm: false });
    await q.tick(); // uploaded; job running
    await q.tick(); // job unknown now (server restarted) -> queued again, then re-sent at once
    expect(upload).toHaveBeenCalledTimes(2);
    const it = q.snapshot()[0]!;
    expect(it.state).toBe('done');
    expect(describeItem(it)).toBe('already scanned: book page 7');
  });

  it('picks up after the app was closed mid-upload', async () => {
    const stuck: QueueItem = { id: 'e', doc: 'book', uri: 'file:///e.jpg', origin: 'camera', noLlm: false, createdAt: 0, state: 'uploading', attempts: 0, nextAttemptAt: 0 };
    const mem = memoryStorage([stuck]);
    const upload = jest.fn(async () => job('j', 'queued'));
    const q = new UploadQueue(mem.storage, upload, async () => job('j', 'queued'), { now: clock });
    await q.tick();
    expect(upload).toHaveBeenCalledTimes(1);
    expect(mem.saved()[0]).toMatchObject({ state: 'processing', jobId: 'j' });
  });

  it('backs off exponentially up to a ceiling', () => {
    expect([1, 2, 3].map((n) => backoff(n))).toEqual([2_000, 4_000, 8_000]);
    expect(backoff(30)).toBe(300_000);
  });
});
