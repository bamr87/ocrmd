/**
 * Review one page: the photograph beside the reconciled text. Each disagreement between the local OCR and
 * the model is a card with both readings — pick one or type your own; digits are never decided for you.
 * With none left open, sign the page off. Every write here is a human edit: it survives the next
 * `verify` on the computer, exactly like an edit made in the web UI or in the file.
 */
import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { Alert, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ApiError, type Flag, type ImageKind, type Page } from '@/api/client';
import { AuthImage, Button, Loading, Message, StatusPill, usePalette } from '@/components/ui';
import { splitMarks } from '@/lib/marks';
import { useClient } from '@/state/session';

const VIEWS: ImageKind[] = ['flat', 'processed', 'raw'];

export default function PageScreen() {
  const params = useLocalSearchParams<{ slug: string; seq: string }>();
  const slug = params.slug;
  const seq = Number(params.seq);
  const client = useClient();
  const router = useRouter();
  const p = usePalette();
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ImageKind>('flat');
  const [zoom, setZoom] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setPage(await client.page(slug, seq));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    }
  }, [client, slug, seq]);

  useEffect(() => {
    setPage(null);
    setEditing(null);
    load();
  }, [load]);

  /** Run a write; true when the server took it (on failure the edit stays on screen, unsaved). */
  const act = async (fn: () => Promise<Page>): Promise<boolean> => {
    setBusy(true);
    try {
      setPage(await fn());
      return true;
    } catch (e) {
      Alert.alert('Not saved', e instanceof ApiError ? e.detail : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  };

  const go = (to: number | null | undefined) => {
    if (to != null) router.replace({ pathname: '/page/[slug]/[seq]', params: { slug, seq: String(to) } });
  };

  if (error && !page) return <Message title="Cannot open this page" detail={error}><Button title="Try again" onPress={load} /></Message>;
  if (!page) return <Loading />;

  const kinds = VIEWS.filter((k) => page.images[k]);
  const shown = kinds.includes(view) ? view : kinds[0];
  const reviewed = page.status === 'reviewed';

  return (
    <SafeAreaView edges={['bottom']} style={{ flex: 1, backgroundColor: p.bg }}>
      <Stack.Screen options={{ title: `Page ${page.sequence}${page.page_label ? ` · p.${page.page_label}` : ''}` }} />
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }} keyboardVerticalOffset={90}>
        <ScrollView contentContainerStyle={{ padding: 12, gap: 12 }} keyboardShouldPersistTaps="handled">
          {shown ? (
            <View style={{ gap: 6 }}>
              <Pressable onPress={() => setZoom(true)}>
                <AuthImage client={client} path={client.imageUrl(slug, seq, shown, 1600)} style={[s.image, { backgroundColor: p.card }]} />
              </Pressable>
              <View style={s.row}>
                {kinds.map((k) => (
                  <Pressable key={k} onPress={() => setView(k)} style={[s.seg, { borderColor: p.border, backgroundColor: k === shown ? p.accent : p.card }]}>
                    <Text style={{ color: k === shown ? '#fff' : p.text }}>{k}</Text>
                  </Pressable>
                ))}
              </View>
            </View>
          ) : null}

          <View style={[s.row, { alignItems: 'center' }]}>
            <StatusPill status={page.status} />
            {page.agreement != null ? <Text style={{ color: p.muted }}>agreement {page.agreement.toFixed(2)}</Text> : null}
            {page.overlap != null ? <Text style={{ color: p.muted }}>overlap {page.overlap.toFixed(2)}</Text> : null}
            {page.reviewed_by ? <Text style={{ color: p.muted }}>signed off by {page.reviewed_by}</Text> : null}
          </View>
          {page.advice ? <Text style={{ color: p.muted }}>{page.advice}</Text> : null}
          {page.llm_error ? <Text style={{ color: p.danger }}>model: {page.llm_error}</Text> : null}

          {page.flags.map((f) => (
            <FlagCard key={f.n} flag={f} busy={busy} onChoose={(chosen) => act(() => client.resolve(slug, seq, f.n, chosen))} />
          ))}

          <View style={s.row}>
            <Text style={[s.h2, { color: p.text, flex: 1 }]}>Text</Text>
            {editing == null ? (
              <Button title="Edit" onPress={() => setEditing(page.body)} />
            ) : (
              <>
                <Button title="Cancel" onPress={() => setEditing(null)} />
                <Button title="Save" kind="primary" busy={busy} onPress={() => act(() => client.editPage(slug, seq, { body: editing })).then((ok) => ok && setEditing(null))} />
              </>
            )}
          </View>
          {editing != null ? (
            <TextInput
              value={editing}
              onChangeText={setEditing}
              multiline
              autoCorrect={false}
              style={[s.editor, { color: p.text, borderColor: p.border, backgroundColor: p.card }]}
            />
          ) : (
            <Text selectable style={[s.body, { color: p.text }]}>
              {page.exists ? splitMarks(page.body).map((part, i) => (part.mark ? <Text key={i} style={s.mark}>{part.text}</Text> : part.text)) : 'Not read yet: the computer has not processed this page.'}
            </Text>
          )}
        </ScrollView>
      </KeyboardAvoidingView>

      <View style={[s.bar, { borderColor: p.border, backgroundColor: p.bg }]}>
        <Button title="‹" onPress={() => go(page.prev)} disabled={page.prev == null} style={s.nav} />
        {reviewed ? (
          <Button title="Reopen" busy={busy} onPress={() => act(() => client.reopen(slug, seq))} style={{ flex: 1 }} />
        ) : (
          <Button
            title={page.unresolved ? `${page.unresolved} open` : 'Sign off'}
            kind="primary"
            disabled={page.unresolved > 0 || !page.exists}
            busy={busy}
            onPress={() => act(() => client.signOff(slug, seq)).then((ok) => ok && page.next_flagged != null && go(page.next_flagged))}
            style={{ flex: 1 }}
          />
        )}
        <Button title="Next flagged" onPress={() => go(page.next_flagged)} disabled={page.next_flagged == null} style={{ flex: 1 }} />
        <Button title="›" onPress={() => go(page.next)} disabled={page.next == null} style={s.nav} />
      </View>

      {zoom && shown ? (
        <View style={[StyleSheet.absoluteFill, { backgroundColor: '#000' }]}>
          <ScrollView maximumZoomScale={6} minimumZoomScale={1} centerContent contentContainerStyle={{ flexGrow: 1 }}>
            <AuthImage client={client} path={client.imageUrl(slug, seq, shown)} style={{ flex: 1, minHeight: 500 }} />
          </ScrollView>
          <Button title="Close" onPress={() => setZoom(false)} style={s.close} />
        </View>
      ) : null}
    </SafeAreaView>
  );
}

function FlagCard({ flag, busy, onChoose }: { flag: Flag; busy: boolean; onChoose: (chosen: string) => void }) {
  const p = usePalette();
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <View style={[s.card, { backgroundColor: p.card, borderColor: '#e0a100' }]}>
      <Text style={{ color: p.muted }}>#{flag.n} · {flag.context}</Text>
      <View style={s.row}>
        <Button title={`OCR: ${flag.ocr || '∅'}`} disabled={busy} onPress={() => onChoose(flag.ocr)} style={{ flex: 1 }} />
        <Button title={`Model: ${flag.llm || '∅'}`} disabled={busy} onPress={() => onChoose(flag.llm)} style={{ flex: 1 }} />
      </View>
      {typing == null ? (
        <Button title="Type what the page says…" onPress={() => setTyping(flag.llm)} />
      ) : (
        <View style={s.row}>
          <TextInput value={typing} onChangeText={setTyping} autoFocus autoCorrect={false} style={[s.input, { color: p.text, borderColor: p.border }]} />
          <Button title="Use" kind="primary" disabled={busy} onPress={() => onChoose(typing)} />
        </View>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  image: { width: '100%', aspectRatio: 0.75, borderRadius: 8 },
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  seg: { borderRadius: 14, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: 12, paddingVertical: 6 },
  h2: { fontSize: 18, fontWeight: '600' },
  body: { fontSize: 16, lineHeight: 23 },
  mark: { backgroundColor: '#ffe08a', color: '#1d1d1f' },
  editor: { minHeight: 320, borderWidth: StyleSheet.hairlineWidth, borderRadius: 8, padding: 10, fontSize: 15, textAlignVertical: 'top' },
  card: { borderWidth: 1.5, borderRadius: 12, padding: 10, gap: 8 },
  input: { flex: 1, borderWidth: StyleSheet.hairlineWidth, borderRadius: 8, paddingHorizontal: 10, fontSize: 16, minHeight: 44 },
  bar: { flexDirection: 'row', gap: 8, padding: 10, borderTopWidth: StyleSheet.hairlineWidth },
  nav: { width: 48 },
  close: { position: 'absolute', top: 50, right: 16 },
});
