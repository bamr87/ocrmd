/**
 * The upload queue: every shot is kept on the phone until the server has it.
 *
 * Shooting never waits for the network. A photo is written to the app's documents directory and queued;
 * the queue uploads one at a time (the server's OCR worker is serial anyway), backs off while the
 * computer is unreachable, and then follows the server's ingest job until it says what the photo
 * became: a page with a verdict, a duplicate of a page already scanned, or a refusal. A photo is only
 * deleted from the phone once the server has answered for it.
 *
 * Every upload carries the item's id as `client_id`, so a retry after a lost response is recognised as
 * the same upload (the server returns the same job), and should the server have restarted in between,
 * the capture's own hash still makes it a duplicate rather than a second page. Nothing in this file
 * touches React Native: storage, upload and job polling are injected, so it runs under plain Jest.
 */
import { ApiError, type CaptureResult, type Job } from '../api/client';

export type ItemState = 'queued' | 'uploading' | 'processing' | 'done' | 'failed';

export type QueueItem = {
  id: string; // also the upload's client_id
  doc: string;
  uri: string; // the photo on the phone
  createdAt: number;
  state: ItemState;
  attempts: number;
  nextAttemptAt: number;
  origin: 'camera' | 'scanner';
  noLlm: boolean;
  jobId?: string;
  error?: string;
  result?: CaptureResult;
};

export interface QueueStorage {
  load(): Promise<QueueItem[]>;
  save(items: QueueItem[]): Promise<void>;
  /** Remove the local photo once the server has answered for it. */
  discard(uri: string): Promise<void>;
}

/** Sends one photo; resolves with the job the server started, rejects with an ApiError. */
export type Uploader = (item: QueueItem) => Promise<Job>;
export type JobFetcher = (id: string) => Promise<Job>;

export type QueueOptions = { now?: () => number; maxBackoffMs?: number; baseBackoffMs?: number };

export function backoff(attempts: number, base = 2_000, max = 300_000): number {
  return Math.min(max, base * 2 ** Math.max(0, attempts - 1));
}

/** One line for the operator about what became of a shot. */
export function describe(item: QueueItem): string {
  const r = item.result;
  switch (item.state) {
    case 'queued':
      return item.attempts ? `waiting for the computer (try ${item.attempts + 1})` : 'waiting to upload';
    case 'uploading':
      return 'uploading…';
    case 'processing':
      return 'reading…';
    case 'failed':
      return item.error ?? 'failed';
    case 'done':
      if (!r) return 'done';
      if (r.error) return r.error;
      if (r.pages?.length) return r.pages.map((p) => `page ${p.sequence} · ${p.status}`).join(', ');
      if (r.duplicate_of) return `already scanned: ${r.duplicate_of.document} page ${r.duplicate_of.sequence}`;
      if (r.rejected?.length) return `not read: ${r.rejected.join('; ')}`;
      return 'no page';
  }
}

export class UploadQueue {
  private items: QueueItem[] = [];
  private loaded = false;
  private running = false;
  private listeners = new Set<(items: QueueItem[]) => void>();
  private readonly now: () => number;
  private readonly base: number;
  private readonly max: number;

  constructor(
    private readonly storage: QueueStorage,
    private readonly upload: Uploader,
    private readonly fetchJob: JobFetcher,
    opts: QueueOptions = {},
  ) {
    this.now = opts.now ?? Date.now;
    this.base = opts.baseBackoffMs ?? 2_000;
    this.max = opts.maxBackoffMs ?? 300_000;
  }

  async load(): Promise<void> {
    if (this.loaded) return;
    // an upload cut off by the app closing is simply queued again; the server dedups the retry
    this.items = (await this.storage.load()).map((i) => (i.state === 'uploading' ? { ...i, state: 'queued' } : i));
    this.loaded = true;
    this.emit();
  }

  subscribe(fn: (items: QueueItem[]) => void): () => void {
    this.listeners.add(fn);
    fn(this.snapshot());
    return () => this.listeners.delete(fn);
  }

  snapshot(): QueueItem[] {
    return this.items.map((i) => ({ ...i }));
  }

  async add(item: Omit<QueueItem, 'state' | 'attempts' | 'nextAttemptAt' | 'createdAt'> & { createdAt?: number }): Promise<void> {
    await this.load();
    this.items.push({ createdAt: this.now(), ...item, state: 'queued', attempts: 0, nextAttemptAt: 0 });
    await this.persist();
  }

  /** Try a failed (or waiting) item again now. */
  async retry(id: string): Promise<void> {
    await this.update(id, (i) => ({ ...i, state: i.jobId && i.state !== 'failed' ? 'processing' : 'queued', nextAttemptAt: 0, error: undefined }));
  }

  /** Forget an item; its photo is deleted from the phone. */
  async remove(id: string): Promise<void> {
    const it = this.items.find((i) => i.id === id);
    if (!it) return;
    this.items = this.items.filter((i) => i.id !== id);
    await this.storage.discard(it.uri);
    await this.persist();
  }

  async clearFinished(): Promise<void> {
    this.items = this.items.filter((i) => i.state !== 'done');
    await this.persist();
  }

  pending(): number {
    return this.items.filter((i) => i.state !== 'done' && i.state !== 'failed').length;
  }

  /**
   * One pass: follow every job in flight, then upload the next due photo. Safe to call on a timer;
   * a pass already running makes this one a no-op.
   */
  async tick(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      await this.load();
      for (const it of this.items.filter((i) => i.state === 'processing' && i.jobId)) {
        await this.follow(it);
      }
      const next = this.items.find((i) => i.state === 'queued' && i.nextAttemptAt <= this.now());
      if (next) await this.send(next);
    } finally {
      this.running = false;
    }
  }

  private async send(item: QueueItem): Promise<void> {
    await this.update(item.id, (i) => ({ ...i, state: 'uploading' }));
    try {
      const job = await this.upload(this.items.find((i) => i.id === item.id) ?? item);
      await this.update(item.id, (i) => ({ ...i, state: 'processing', jobId: job.id, attempts: i.attempts + 1 }));
      await this.settle(item.id, job);
    } catch (e) {
      const err = e instanceof ApiError ? e : new ApiError(0, e instanceof Error ? e.message : String(e));
      await this.update(item.id, (i) => {
        const attempts = i.attempts + 1;
        // the network or the server's own trouble: wait and try again; anything else will not get better
        return err.transient
          ? { ...i, state: 'queued', attempts, error: err.detail, nextAttemptAt: this.now() + backoff(attempts, this.base, this.max) }
          : { ...i, state: 'failed', attempts, error: err.unpaired ? 'this phone is no longer paired; pair it again' : err.detail };
      });
    }
  }

  private async follow(item: QueueItem): Promise<void> {
    try {
      await this.settle(item.id, await this.fetchJob(item.jobId!));
    } catch (e) {
      // a 404 means the server restarted and forgot the job; uploading again is safe (it dedups)
      if (e instanceof ApiError && e.status === 404) {
        await this.update(item.id, (i) => ({ ...i, state: 'queued', jobId: undefined, nextAttemptAt: 0 }));
      }
      // otherwise unreachable for now: keep following on the next pass
    }
  }

  private async settle(id: string, job: Job): Promise<void> {
    if (job.status === 'queued' || job.status === 'running') return;
    const item = this.items.find((i) => i.id === id);
    if (!item) return;
    const result = job.results?.find((r) => r.client_id === id) ?? job.results?.[0];
    if (job.status === 'done' && result && !result.error) {
      await this.update(id, (i) => ({ ...i, state: 'done', result, error: undefined }));
      await this.storage.discard(item.uri);
    } else {
      await this.update(id, (i) => ({ ...i, state: 'failed', result, error: result?.error ?? job.error ?? `the server's job ended ${job.status}` }));
    }
  }

  private async update(id: string, fn: (i: QueueItem) => QueueItem): Promise<void> {
    this.items = this.items.map((i) => (i.id === id ? fn(i) : i));
    await this.persist();
  }

  private async persist(): Promise<void> {
    await this.storage.save(this.items);
    this.emit();
  }

  private emit(): void {
    const snap = this.snapshot();
    this.listeners.forEach((fn) => fn(snap));
  }
}
