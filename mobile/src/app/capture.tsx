/**
 * Scan pages with the phone. Two ways in:
 *
 * - **Camera** — the phone's camera as the rig's webcam: one press, one photo, sent as it was taken.
 *   The server crops, splits spreads and reads it exactly as it would a webcam frame.
 * - **Document scanner** — the system scanner (VisionKit on iOS, ML Kit on Android) finds and flattens
 *   each page itself; several pages per session. Its crop is recorded as such (`source.type: mobile_scan`).
 *
 * Either way the photo goes into the upload queue at once, so shooting never waits for the network,
 * and the strip at the bottom says what the server made of the last shots. Pressing the shutter is a
 * decision, not a guess, so nothing a person shoots here is refused for looking unreadable.
 */
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { FlatList, Pressable, StyleSheet, Switch, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import type { Document } from '@/api/client';
import { Button, Loading, Message, STATUS_COLOR, usePalette } from '@/components/ui';
import { describe } from '@/lib/queue';
import { useClient } from '@/state/session';
import { useUploads } from '@/state/uploads';

export default function Capture() {
  useKeepAwake();
  const params = useLocalSearchParams<{ doc?: string }>();
  const [doc, setDoc] = useState<string | null>(params.doc ?? null);
  return doc ? <Shoot doc={doc} onChangeDoc={() => setDoc(null)} /> : <PickDocument onPick={setDoc} />;
}

function PickDocument({ onPick }: { onPick: (slug: string) => void }) {
  const client = useClient();
  const router = useRouter();
  const p = usePalette();
  const [docs, setDocs] = useState<Document[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    client.documents().then(setDocs, (e) => setError(String(e?.detail ?? e)));
  }, [client]);
  if (error) return <Message title="Cannot reach ocrmd" detail={error}><Button title="Back" onPress={() => router.back()} /></Message>;
  if (!docs) return <Loading />;
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }}>
      <Text style={[s.heading, { color: p.text }]}>Which document are these pages of?</Text>
      <FlatList
        data={docs}
        keyExtractor={(d) => d.slug}
        contentContainerStyle={{ padding: 12, gap: 8 }}
        renderItem={({ item }) => (
          <Pressable onPress={() => onPick(item.slug)} style={[s.row, { backgroundColor: p.card, borderColor: p.border }]}>
            <Text style={{ color: p.text, fontSize: 17, fontWeight: '600' }}>{item.title || item.slug}</Text>
            <Text style={{ color: p.muted }}>{item.pages} page(s)</Text>
          </Pressable>
        )}
      />
      <View style={{ flexDirection: 'row', gap: 10, padding: 12 }}>
        <Button title="Cancel" onPress={() => router.back()} style={{ flex: 1 }} />
        <Button title="New document" kind="primary" onPress={() => router.replace('/new')} style={{ flex: 1 }} />
      </View>
    </SafeAreaView>
  );
}

function Shoot({ doc, onChangeDoc }: { doc: string; onChangeDoc: () => void }) {
  const router = useRouter();
  const { items, enqueue } = useUploads();
  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const [busy, setBusy] = useState(false);
  const [noLlm, setNoLlm] = useState(false);
  const [flash, setFlash] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recent = items.filter((i) => i.doc === doc).slice(-3).reverse();

  if (!permission) return <Loading />;
  if (!permission.granted) {
    return (
      <Message title="ocrmd needs the camera to photograph pages" detail={permission.canAskAgain ? undefined : 'Allow it in the system Settings app.'}>
        {permission.canAskAgain ? <Button title="Allow camera" kind="primary" onPress={requestPermission} /> : null}
        <Button title="Back" onPress={() => router.back()} />
      </Message>
    );
  }

  const shoot = async () => {
    if (!camera.current || busy) return;
    setBusy(true);
    setError(null);
    try {
      // quality 0.92: full resolution, JPEG artefacts well below a glyph stroke; exif off — the
      // server reads orientation from the image the camera already rotated
      const photo = await camera.current.takePictureAsync({ quality: 0.92, exif: false });
      setFlash(true);
      setTimeout(() => setFlash(false), 120);
      await enqueue(doc, photo.uri, { origin: 'camera', noLlm });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const scanDocuments = async () => {
    setError(null);
    try {
      const res = await CameraView.scanDocumentAsync({ quality: 0.95 });
      for (const uri of res?.pages ?? []) await enqueue(doc, uri, { origin: 'scanner', noLlm });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>
      <CameraView ref={camera} style={StyleSheet.absoluteFill} facing="back" autofocus="on" animateShutter={false} />
      {flash ? <View style={[StyleSheet.absoluteFill, { backgroundColor: 'rgba(255,255,255,0.6)' }]} /> : null}
      <SafeAreaView style={s.overlay} pointerEvents="box-none">
        <View style={s.top}>
          <Button title="Done" onPress={() => router.back()} />
          <Pressable onPress={onChangeDoc} style={s.docChip}>
            <Text style={s.chipText} numberOfLines={1}>{doc} ▾</Text>
          </Pressable>
          <View style={s.toggle}>
            <Text style={s.chipText}>OCR only</Text>
            <Switch value={noLlm} onValueChange={setNoLlm} />
          </View>
        </View>
        <View style={s.bottom}>
          {error ? <Text style={s.error}>{error}</Text> : null}
          {recent.map((it) => (
            <Pressable
              key={it.id}
              onPress={() => {
                const seq = it.result?.pages?.[0]?.sequence ?? it.result?.duplicate_of?.sequence;
                const slug = it.result?.duplicate_of?.document ?? doc;
                if (seq) router.push({ pathname: '/page/[slug]/[seq]', params: { slug, seq: String(seq) } });
              }}
              style={s.result}
            >
              <View style={[s.dot, { backgroundColor: STATUS_COLOR[it.result?.pages?.[0]?.status ?? ''] ?? (it.state === 'failed' ? '#c62828' : '#9e9e9e') }]} />
              <Text style={s.resultText} numberOfLines={1}>{describe(it)}</Text>
            </Pressable>
          ))}
          <View style={s.controls}>
            {CameraView.isDocumentScannerAvailable ? <Button title="Scanner" onPress={scanDocuments} style={s.side} /> : <View style={s.side} />}
            <Pressable accessibilityLabel="Take photo" onPress={shoot} disabled={busy} style={({ pressed }) => [s.shutter, { opacity: pressed || busy ? 0.6 : 1 }]}>
              <View style={s.shutterInner} />
            </Pressable>
            <Button title="Pages" onPress={() => router.push({ pathname: '/doc/[slug]', params: { slug: doc } })} style={s.side} />
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const s = StyleSheet.create({
  heading: { fontSize: 20, fontWeight: '600', padding: 16 },
  row: { padding: 14, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, gap: 4 },
  overlay: { flex: 1, justifyContent: 'space-between' },
  top: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 12, gap: 8 },
  docChip: { flexShrink: 1, backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 16, paddingHorizontal: 12, paddingVertical: 8 },
  chipText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  toggle: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 16, paddingLeft: 12, paddingRight: 4, paddingVertical: 2 },
  bottom: { padding: 12, gap: 6, backgroundColor: 'rgba(0,0,0,0.45)' },
  result: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  resultText: { color: '#fff', fontSize: 14, flex: 1 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  error: { color: '#ff8a80' },
  controls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 6 },
  side: { width: 96 },
  shutter: { width: 76, height: 76, borderRadius: 38, borderWidth: 4, borderColor: '#fff', alignItems: 'center', justifyContent: 'center' },
  shutterInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: '#fff' },
});
