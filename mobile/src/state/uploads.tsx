/**
 * The upload queue (lib/queue.ts) wired to the phone: photos and the queue's state live in the app's
 * documents directory, so neither is lost when the app is closed or the OS clears caches, and uploads
 * are native multipart transfers straight from the file (expo-file-system), not base64 in JavaScript.
 * The queue runs every few seconds while the app is in the foreground.
 */
import * as Crypto from 'expo-crypto';
import { Directory, File, Paths, UploadType } from 'expo-file-system';
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { AppState } from 'react-native';

import { ApiError, type Job, type OcrmdClient } from '@/api/client';
import { type QueueItem, type QueueStorage, UploadQueue } from '@/lib/queue';
import { useSession } from './session';

const TICK_MS = 3_000;

function storage(): QueueStorage {
  const dir = new Directory(Paths.document, 'captures');
  const index = new File(dir, 'queue.json');
  let writes: Promise<void> = Promise.resolve(); // one write at a time, in order: the last state wins
  return {
    async load() {
      if (!dir.exists) dir.create({ intermediates: true });
      if (!index.exists) return [];
      try {
        return JSON.parse(await index.text()) as QueueItem[];
      } catch {
        return [];
      }
    },
    save(items) {
      const data = JSON.stringify(items);
      writes = writes.then(async () => {
        if (!dir.exists) dir.create({ intermediates: true });
        await index.write(data);
      });
      return writes;
    },
    async discard(uri) {
      const f = new File(uri);
      if (f.exists) f.delete();
    },
  };
}

function uploader(getClient: () => OcrmdClient | null) {
  return async (item: QueueItem): Promise<Job> => {
    const client = getClient();
    if (!client) throw new ApiError(401, 'not paired');
    const file = new File(item.uri);
    if (!file.exists) throw new ApiError(410, 'the photo is no longer on this phone');
    let res;
    try {
      res = await file.upload(client.captureUrl(item.doc), {
        uploadType: UploadType.MULTIPART,
        fieldName: 'file',
        mimeType: 'image/jpeg',
        headers: client.headers(),
        parameters: { client_id: item.id, origin: item.origin, no_llm: String(item.noLlm) },
      });
    } catch (e) {
      throw new ApiError(0, `cannot reach ${client.server}: ${e instanceof Error ? e.message : String(e)}`);
    }
    let body: unknown;
    try {
      body = JSON.parse(res.body);
    } catch {
      body = undefined;
    }
    if (res.status < 200 || res.status >= 300) {
      const d = (body as { detail?: unknown } | undefined)?.detail;
      throw new ApiError(res.status, typeof d === 'string' ? d : `upload failed (HTTP ${res.status})`);
    }
    return body as Job;
  };
}

type UploadsState = {
  items: QueueItem[];
  pending: number;
  /** Move a just-taken photo into the app's own storage and queue it for `doc`. */
  enqueue: (doc: string, tempUri: string, opts?: { origin?: 'camera' | 'scanner'; noLlm?: boolean }) => Promise<void>;
  retry: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  clearFinished: () => Promise<void>;
  kick: () => void;
};

const Ctx = createContext<UploadsState | null>(null);

export function UploadsProvider({ children }: { children: ReactNode }) {
  const { client } = useSession();
  const clientRef = useRef(client);
  clientRef.current = client;
  const queue = useMemo(() => new UploadQueue(storage(), uploader(() => clientRef.current), (id) => {
    const c = clientRef.current;
    if (!c) return Promise.reject(new ApiError(401, 'not paired'));
    return c.job(id);
  }), []);
  const [items, setItems] = useState<QueueItem[]>([]);

  useEffect(() => queue.subscribe(setItems), [queue]);

  const kick = useCallback(() => {
    if (clientRef.current) queue.tick().catch(() => undefined);
  }, [queue]);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = setInterval(kick, TICK_MS);
    const sub = AppState.addEventListener('change', (s) => {
      if (s === 'active' && !timer) {
        timer = setInterval(kick, TICK_MS);
        kick();
      } else if (s !== 'active' && timer) {
        clearInterval(timer);
        timer = null;
      }
    });
    kick();
    return () => {
      if (timer) clearInterval(timer);
      sub.remove();
    };
  }, [kick]);

  const enqueue = useCallback(
    async (doc: string, tempUri: string, opts: { origin?: 'camera' | 'scanner'; noLlm?: boolean } = {}) => {
      const id = Crypto.randomUUID();
      const dir = new Directory(Paths.document, 'captures');
      if (!dir.exists) dir.create({ intermediates: true });
      const kept = new File(dir, `${id}.jpg`);
      new File(tempUri).moveSync(kept);
      await queue.add({ id, doc, uri: kept.uri, origin: opts.origin ?? 'camera', noLlm: opts.noLlm ?? false });
      kick();
    },
    [queue, kick],
  );

  const value = useMemo<UploadsState>(
    () => ({
      items,
      pending: items.filter((i) => i.state !== 'done' && i.state !== 'failed').length,
      enqueue,
      retry: (id) => queue.retry(id).then(kick),
      remove: (id) => queue.remove(id),
      clearFinished: () => queue.clearFinished(),
      kick,
    }),
    [items, enqueue, queue, kick],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useUploads(): UploadsState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useUploads outside UploadsProvider');
  return v;
}
