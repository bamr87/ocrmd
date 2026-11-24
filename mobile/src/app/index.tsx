/** The library: every document on the paired computer, with its progress. The app's home screen. */
import { Link, Redirect, Stack, useFocusEffect, useRouter } from 'expo-router';
import { useCallback, useState } from 'react';
import { FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ApiError, type Document } from '@/api/client';
import { AuthImage, Button, Loading, Message, usePalette } from '@/components/ui';
import { useSession } from '@/state/session';
import { useUploads } from '@/state/uploads';

export default function Library() {
  const { ready, session, client } = useSession();
  const { pending } = useUploads();
  const router = useRouter();
  const p = usePalette();
  const [docs, setDocs] = useState<Document[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    if (!client) return;
    try {
      setDocs(await client.documents());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, String(e)));
    }
  }, [client]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  if (!ready) return <Loading />;
  if (!session || !client) return <Redirect href="/pair" />;

  const header = (
    <Stack.Screen
      options={{
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 16 }}>
            <Link href="/search" style={{ color: p.accent, fontSize: 16 }}>Search</Link>
            <Link href="/settings" style={{ color: p.accent, fontSize: 16 }}>{pending ? `Settings (${pending})` : 'Settings'}</Link>
          </View>
        ),
      }}
    />
  );

  if (error && !docs) {
    return (
      <>
        {header}
        <Message title={error.unpaired ? 'This phone is no longer paired' : 'Cannot reach ocrmd'} detail={error.detail}>
          {error.unpaired ? <Button title="Pair again" kind="primary" onPress={() => router.push('/settings')} /> : <Button title="Try again" onPress={load} />}
        </Message>
      </>
    );
  }
  if (!docs) return <Loading />;

  return (
    <SafeAreaView edges={['bottom']} style={{ flex: 1, backgroundColor: p.bg }}>
      {header}
      <FlatList
        data={docs}
        keyExtractor={(d) => d.slug}
        contentContainerStyle={{ padding: 12, gap: 10 }}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await load(); setRefreshing(false); }} />}
        ListEmptyComponent={<Message title="Nothing scanned yet" detail="Start a document, then shoot its pages with the camera." />}
        renderItem={({ item: d }) => (
          <Pressable onPress={() => router.push({ pathname: '/doc/[slug]', params: { slug: d.slug } })} style={[s.card, { backgroundColor: p.card, borderColor: p.border }]}>
            {d.cover ? <AuthImage client={client} path={d.cover} style={s.cover} contentFit="cover" /> : <View style={[s.cover, { backgroundColor: p.border }]} />}
            <View style={{ flex: 1, gap: 4 }}>
              <Text style={[s.title, { color: p.text }]} numberOfLines={2}>{d.title || d.slug}</Text>
              <Text style={{ color: p.muted }}>{d.slug}{d.author?.length ? ` · ${d.author.join(', ')}` : ''}</Text>
              <Text style={{ color: p.muted }}>
                {d.pages} page{d.pages === 1 ? '' : 's'} · {d.done} done
                {d.flagged ? ` · ${d.flagged} to review` : ''}
                {d.needs_reshoot.length ? ` · ${d.needs_reshoot.length} to re-shoot` : ''}
              </Text>
              {d.pages ? (
                <View style={[s.bar, { backgroundColor: p.border }]}>
                  <View style={[s.barFill, { width: `${Math.round((100 * d.done) / d.pages)}%`, backgroundColor: p.accent }]} />
                </View>
              ) : null}
            </View>
          </Pressable>
        )}
      />
      <View style={[s.footer, { borderColor: p.border, backgroundColor: p.bg }]}>
        <Button title="New document" onPress={() => router.push('/new')} style={{ flex: 1 }} />
        <Button title="Scan pages" kind="primary" onPress={() => router.push('/capture')} style={{ flex: 1 }} />
      </View>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  card: { flexDirection: 'row', gap: 12, padding: 10, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth },
  cover: { width: 64, height: 86, borderRadius: 6 },
  title: { fontSize: 17, fontWeight: '600' },
  bar: { height: 4, borderRadius: 2, overflow: 'hidden', marginTop: 4 },
  barFill: { height: 4 },
  footer: { flexDirection: 'row', gap: 10, padding: 12, borderTopWidth: StyleSheet.hairlineWidth },
});
