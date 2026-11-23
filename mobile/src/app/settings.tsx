/** The paired computer, this phone's uploads, and unpairing. */
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { ApiError, type ServerInfo } from '@/api/client';
import { Button, STATUS_COLOR, usePalette } from '@/components/ui';
import { describe } from '@/lib/queue';
import { useSession } from '@/state/session';
import { useUploads } from '@/state/uploads';

export default function Settings() {
  const { session, client, unpair } = useSession();
  const { items, retry, remove, clearFinished } = useUploads();
  const router = useRouter();
  const p = usePalette();
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    client?.serverInfo().then(setInfo, (e) => setError(e instanceof ApiError ? e.detail : String(e)));
  }, [client]);

  const row = (k: string, v: string) => (
    <View style={s.kv} key={k}>
      <Text style={{ color: p.muted }}>{k}</Text>
      <Text style={{ color: p.text, flexShrink: 1, textAlign: 'right' }}>{v}</Text>
    </View>
  );

  return (
    <ScrollView style={{ backgroundColor: p.bg }} contentContainerStyle={{ padding: 16, gap: 12 }}>
      <Text style={[s.h2, { color: p.text }]}>Computer</Text>
      <View style={[s.card, { backgroundColor: p.card, borderColor: p.border }]}>
        {row('server', session?.server ?? '—')}
        {row('this phone', session?.device.name ?? '—')}
        {info ? (
          <>
            {row('ocrmd', `${info.version} · API v${info.api}`)}
            {row('second reading', info.llm ? `${info.provider} · ${info.model}` : 'none configured — pages are OCR-only until `ocrmd verify`')}
            {row('OCR engines', info.ocr_engines.join(', ') || 'none installed')}
            {info.scanning ? row('webcam scan', info.scanning) : null}
          </>
        ) : null}
        {error ? <Text style={{ color: p.danger }}>{error}</Text> : null}
      </View>

      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        <Text style={[s.h2, { color: p.text, flex: 1 }]}>Uploads</Text>
        {items.some((i) => i.state === 'done') ? <Button title="Clear finished" onPress={clearFinished} /> : null}
      </View>
      {items.length === 0 ? <Text style={{ color: p.muted }}>Nothing waiting. Photos stay on this phone until the computer has read them.</Text> : null}
      {[...items].reverse().map((it) => (
        <View key={it.id} style={[s.card, { backgroundColor: p.card, borderColor: p.border }]}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
            <View style={[s.dot, { backgroundColor: STATUS_COLOR[it.result?.pages?.[0]?.status ?? ''] ?? (it.state === 'failed' ? '#c62828' : '#9e9e9e') }]} />
            <Text style={{ color: p.text, flex: 1 }} numberOfLines={2}>{it.doc} · {describe(it)}</Text>
          </View>
          <Text style={{ color: p.muted, fontSize: 12 }}>{new Date(it.createdAt).toLocaleString()} · {it.origin}{it.noLlm ? ' · OCR only' : ''}</Text>
          {it.state === 'failed' || (it.state === 'queued' && it.attempts) ? (
            <View style={{ flexDirection: 'row', gap: 8 }}>
              <Button title="Retry now" onPress={() => retry(it.id)} style={{ flex: 1 }} />
              <Button title="Delete photo" kind="danger" onPress={() => remove(it.id)} style={{ flex: 1 }} />
            </View>
          ) : null}
        </View>
      ))}

      <Text style={[s.h2, { color: p.text }]}>Pairing</Text>
      <Pressable>
        <Button
          title="Unpair this phone"
          kind="danger"
          onPress={() =>
            Alert.alert('Unpair?', 'This phone forgets its key. Photos still waiting stay queued for when you pair again. To stop the key working on the computer too, run `ocrmd devices --revoke` there.', [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Unpair', style: 'destructive', onPress: () => unpair().then(() => router.replace('/pair')) },
            ])
          }
        />
      </Pressable>
    </ScrollView>
  );
}

const s = StyleSheet.create({
  h2: { fontSize: 18, fontWeight: '600', marginTop: 8 },
  card: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 12, padding: 12, gap: 8 },
  kv: { flexDirection: 'row', justifyContent: 'space-between', gap: 12 },
  dot: { width: 10, height: 10, borderRadius: 5 },
});
