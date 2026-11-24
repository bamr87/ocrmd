/** One document: its progress, a grid of its pages by verdict, and the ways on — scan more, review, verify. */
import { Stack, useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, FlatList, Pressable, RefreshControl, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ApiError, type DocumentDetail, type PageRow } from '@/api/client';
import { AuthImage, Button, Loading, Message, STATUS_COLOR, usePalette } from '@/components/ui';
import { useClient } from '@/state/session';

const FILTERS = ['all', 'flagged', 'unverified', 'reshoot'] as const;
type Filter = (typeof FILTERS)[number];

export default function DocumentScreen() {
  const { slug } = useLocalSearchParams<{ slug: string }>();
  const client = useClient();
  const router = useRouter();
  const p = usePalette();
  const { width } = useWindowDimensions();
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [refreshing, setRefreshing] = useState(false);
  const [verifying, setVerifying] = useState(false);

  const load = useCallback(async () => {
    try {
      setDoc(await client.document(slug));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    }
  }, [client, slug]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  if (error && !doc) return <Message title="Cannot open this document" detail={error}><Button title="Try again" onPress={load} /></Message>;
  if (!doc) return <Loading />;

  const reshoot = new Set(doc.needs_reshoot);
  const pages = doc.page_list.filter((r) =>
    filter === 'all' ? true : filter === 'reshoot' ? reshoot.has(r.sequence) : filter === 'unverified' ? r.status === 'unverified' || r.status === 'error' : r.status === filter,
  );
  const firstFlagged = doc.page_list.find((r) => r.status === 'flagged');
  const columns = Math.max(3, Math.floor(width / 120));
  const tile = (width - 12 * 2 - 8 * (columns - 1)) / columns;

  const verify = async () => {
    setVerifying(true);
    try {
      const job = await client.verify(doc.slug);
      Alert.alert('Verifying', `${doc.unverified} page(s) queued on the computer (job ${job.id}). Pull down to refresh.`);
    } catch (e) {
      Alert.alert('Could not start', e instanceof ApiError ? e.detail : String(e));
    } finally {
      setVerifying(false);
    }
  };

  const open = (r: PageRow) => router.push({ pathname: '/page/[slug]/[seq]', params: { slug: doc.slug, seq: String(r.sequence) } });

  return (
    <SafeAreaView edges={['bottom']} style={{ flex: 1, backgroundColor: p.bg }}>
      <Stack.Screen options={{ title: doc.title || doc.slug }} />
      <FlatList
        data={pages}
        key={columns}
        numColumns={columns}
        keyExtractor={(r) => String(r.sequence)}
        columnWrapperStyle={{ gap: 8 }}
        contentContainerStyle={{ padding: 12, gap: 8 }}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await load(); setRefreshing(false); }} />}
        ListHeaderComponent={
          <View style={{ gap: 10, marginBottom: 6 }}>
            <Text style={{ color: p.muted }}>
              {doc.pages} page(s){doc.page_labels ? ` · printed ${doc.page_labels}` : ''} · {doc.done} done · {doc.flagged} to review · {doc.unverified} without the model's reading
              {doc.scanning ? ' · the computer is scanning this now' : ''}
            </Text>
            <View style={s.row}>
              <Button title="Scan more" kind="primary" onPress={() => router.push({ pathname: '/capture', params: { doc: doc.slug } })} style={{ flex: 1 }} />
              {firstFlagged ? <Button title={`Review ${doc.flagged}`} onPress={() => open(firstFlagged)} style={{ flex: 1 }} /> : null}
              {doc.unverified ? <Button title="Verify" busy={verifying} onPress={verify} style={{ flex: 1 }} /> : null}
            </View>
            <View style={s.row}>
              {FILTERS.map((f) => (
                <Pressable key={f} onPress={() => setFilter(f)} style={[s.filter, { borderColor: p.border, backgroundColor: filter === f ? p.accent : p.card }]}>
                  <Text style={{ color: filter === f ? '#fff' : p.text }}>{f === 'reshoot' ? 're-shoot' : f}</Text>
                </Pressable>
              ))}
            </View>
          </View>
        }
        ListEmptyComponent={<Message title={doc.pages ? 'No pages match' : 'No pages yet'} detail={doc.pages ? undefined : 'Scan some with the camera.'} />}
        renderItem={({ item: r }) => (
          <Pressable onPress={() => open(r)} style={{ width: tile }}>
            {r.thumb ? (
              <AuthImage client={client} path={r.thumb} style={[s.thumb, { height: tile * 1.35, borderColor: STATUS_COLOR[r.status] ?? p.border }]} contentFit="cover" />
            ) : (
              <View style={[s.thumb, { height: tile * 1.35, borderColor: p.border, backgroundColor: p.card }]} />
            )}
            <Text style={{ color: p.text, fontSize: 12 }} numberOfLines={1}>
              {r.sequence}
              {r.page_label ? ` · p.${r.page_label}` : ''}
            </Text>
            <Text style={{ color: STATUS_COLOR[r.status] ?? p.muted, fontSize: 11 }}>{r.status}</Text>
          </Pressable>
        )}
      />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  filter: { borderRadius: 14, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: 12, paddingVertical: 6 },
  thumb: { width: '100%', borderRadius: 6, borderWidth: 2 },
});
