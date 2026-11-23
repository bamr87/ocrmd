/**
 * Which ocrmd server this phone is paired with, and the client that talks to it.
 *
 * The token is the only secret the app holds; it lives in the platform keychain (expo-secure-store:
 * iOS Keychain, Android Keystore), never in plain storage, and is never shown. Unpairing deletes it
 * here; revoking it on the computer (`ocrmd devices --revoke`) makes the server refuse it, and the
 * app then asks to pair again.
 */
import * as Device from 'expo-device';
import * as SecureStore from 'expo-secure-store';
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { Platform } from 'react-native';

import { type Device as PairedDevice, OcrmdClient, pairWith } from '@/api/client';

const KEY = 'ocrmd.session.v1';

export type Session = { server: string; token: string; device: PairedDevice };

type SessionState = {
  ready: boolean;
  session: Session | null;
  client: OcrmdClient | null;
  pair: (server: string, code: string) => Promise<void>;
  unpair: () => Promise<void>;
};

const Ctx = createContext<SessionState | null>(null);

/** What the computer will list this phone as, unless the operator named it when making the code. */
export function deviceName(): string {
  return (Device.deviceName || Device.modelName || `${Platform.OS} phone`).slice(0, 64);
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [session, setSession] = useState<Session | null>(null);

  useEffect(() => {
    SecureStore.getItemAsync(KEY)
      .then((raw) => setSession(raw ? (JSON.parse(raw) as Session) : null))
      .catch(() => setSession(null))
      .finally(() => setReady(true));
  }, []);

  const pair = useCallback(async (server: string, code: string) => {
    const { paired } = await pairWith(server, code, deviceName(), Platform.OS);
    const s: Session = { server, token: paired.token, device: paired.device };
    await SecureStore.setItemAsync(KEY, JSON.stringify(s));
    setSession(s);
  }, []);

  const unpair = useCallback(async () => {
    await SecureStore.deleteItemAsync(KEY);
    setSession(null);
  }, []);

  const client = useMemo(() => (session ? new OcrmdClient(session.server, session.token) : null), [session]);
  const value = useMemo(() => ({ ready, session, client, pair, unpair }), [ready, session, client, pair, unpair]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSession(): SessionState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useSession outside SessionProvider');
  return v;
}

/** The client for screens that are only reachable once paired. */
export function useClient(): OcrmdClient {
  const { client } = useSession();
  if (!client) throw new Error('not paired');
  return client;
}
