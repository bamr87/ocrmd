/**
 * Pairing: scan the QR code the computer shows (`ocrmd pair`, or Settings in the web UI), follow the
 * `ocrmd://pair?…` link it carries, or type the server and code. The code works once, for 10 minutes.
 */
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { ApiError } from '@/api/client';
import { Button, usePalette } from '@/components/ui';
import { displayCode, isValidCode, normalizeCode, normalizeServer, parsePairLink } from '@/lib/pairing';
import { useSession } from '@/state/session';

export default function Pair() {
  const params = useLocalSearchParams<{ server?: string; code?: string }>();
  const { pair } = useSession();
  const router = useRouter();
  const p = usePalette();
  const [server, setServer] = useState(params.server ?? '');
  const [code, setCode] = useState(params.code ? displayCode(params.code) : '');
  const [scanning, setScanning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const handled = useRef(false);

  const submit = async (srv = server, c = code) => {
    const url = normalizeServer(srv);
    if (!url) return setError('Type the address the computer shows, e.g. 192.168.1.5:8765');
    if (!isValidCode(c)) return setError('The code is 8 letters and digits, like ABCD-2345');
    setBusy(true);
    setError(null);
    try {
      await pair(url, normalizeCode(c));
      router.replace('/');
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };

  // opened from a link (QR code scanned with the phone's own camera): both fields are filled, so go
  useEffect(() => {
    if (params.server && params.code && !handled.current) {
      handled.current = true;
      submit(params.server, params.code);
    }
  }, [params.server, params.code]);

  if (scanning) {
    return (
      <View style={{ flex: 1, backgroundColor: '#000' }}>
        <CameraView
          style={StyleSheet.absoluteFill}
          facing="back"
          barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
          onBarcodeScanned={({ data }) => {
            const target = parsePairLink(data);
            if (!target || handled.current) return;
            handled.current = true;
            setScanning(false);
            setServer(target.server);
            setCode(displayCode(target.code));
            submit(target.server, target.code).finally(() => { handled.current = false; });
          }}
        />
        <View style={s.scanFooter}>
          <Text style={s.scanHint}>Point at the QR code from `ocrmd pair` or the web UI's Settings</Text>
          <Button title="Type it instead" onPress={() => setScanning(false)} />
        </View>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1, backgroundColor: p.bg }}>
      <ScrollView contentContainerStyle={{ padding: 20, gap: 14 }} keyboardShouldPersistTaps="handled">
        <Text style={[s.lead, { color: p.text }]}>
          Pair this phone with the computer that runs ocrmd. On the computer, start the server with{' '}
          <Text style={s.mono}>ocrmd serve --lan</Text> and make a code with <Text style={s.mono}>ocrmd pair</Text> (or
          Settings → Phones & tablets in the web UI). Both must be on the same network.
        </Text>
        <Button
          title="Scan the QR code"
          kind="primary"
          onPress={async () => {
            if (!permission?.granted && !(await requestPermission()).granted) {
              return setError('Camera access is off; type the server and code instead, or allow it in Settings.');
            }
            handled.current = false;
            setScanning(true);
          }}
        />
        <Text style={[s.label, { color: p.muted }]}>Server</Text>
        <TextInput
          value={server}
          onChangeText={setServer}
          placeholder="192.168.1.5:8765"
          placeholderTextColor={p.muted}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          style={[s.input, { color: p.text, borderColor: p.border, backgroundColor: p.card }]}
        />
        <Text style={[s.label, { color: p.muted }]}>Code</Text>
        <TextInput
          value={code}
          onChangeText={(t) => setCode(displayCode(t))}
          placeholder="ABCD-2345"
          placeholderTextColor={p.muted}
          autoCapitalize="characters"
          autoCorrect={false}
          maxLength={9}
          style={[s.input, s.mono, { color: p.text, borderColor: p.border, backgroundColor: p.card, fontSize: 22, letterSpacing: 2 }]}
        />
        {error ? <Text style={{ color: p.danger }}>{error}</Text> : null}
        <Button title="Pair" kind="primary" busy={busy} onPress={() => submit()} />
        <Text style={{ color: p.muted, fontSize: 13 }}>
          The code is used once. The computer gives this phone its own key, kept in the phone's keychain; revoke it any time
          with <Text style={s.mono}>ocrmd devices --revoke</Text>.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const s = StyleSheet.create({
  lead: { fontSize: 16, lineHeight: 22 },
  label: { fontSize: 13, textTransform: 'uppercase', marginBottom: -8 },
  input: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, fontSize: 17 },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', default: 'monospace' }) },
  scanFooter: { position: 'absolute', left: 0, right: 0, bottom: 0, padding: 24, gap: 12, backgroundColor: 'rgba(0,0,0,0.55)' },
  scanHint: { color: '#fff', textAlign: 'center', fontSize: 15 },
});
