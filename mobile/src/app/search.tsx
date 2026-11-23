/** Full-text search across every page of every document on the computer. */
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { FlatList, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { ApiError, type Hit } from '@/api/client';
import { Message, StatusPill, usePalette } from '@/components/ui';
import { useClient } from '@/state/session';

export default function Search() {
  const client = useClient();
  const router = useRouter();
  const p = usePalette();
  const [q, setQ] = useState('');
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!q.trim()) return;
    try {
      setHits(await client.search(q.trim()));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: p.bg }}>
      <TextInput
        value={q}
        onChangeText={setQ}
        onSubmitEditing={run}
        returnKeyType="search"
        placeholder='Words, or "an exact phrase"'
        placeholderTextColor={p.muted}
        autoFocus
        autoCorrect={false}
        style={[s.input, { color: p.text, borderColor: p.border, backgroundColor: p.card }]}
      />
      {error ? <Text style={{ color: p.danger, paddingHorizontal: 16 }}>{error}</Text> : null}
      <FlatList
        data={hits ?? []}
        keyExtractor={(h) => h.page_id}
        contentContainerStyle={{ padding: 12, gap: 8 }}
        ListEmptyComponent={hits ? <Message title="No page says that" /> : undefined}
        renderItem={({ item: h }) => (
          <Pressable
            onPress={() => router.push({ pathname: '/page/[slug]/[seq]', params: { slug: h.document, seq: String(h.sequence) } })}
            style={[s.hit, { backgroundColor: p.card, borderColor: p.border }]}
          >
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', gap: 8 }}>
              <Text style={{ color: p.text, fontWeight: '600', flexShrink: 1 }} numberOfLines={1}>
                {h.title || h.document} · page {h.sequence}{h.page_label ? ` (p.${h.page_label})` : ''}
              </Text>
              {h.status ? <StatusPill status={h.status} /> : null}
            </View>
            <Text style={{ color: p.muted }} numberOfLines={3}>{h.snippet.replace(/\s+/g, ' ')}</Text>
          </Pressable>
        )}
      />
    </View>
  );
}

const s = StyleSheet.create({
  input: { margin: 12, borderWidth: StyleSheet.hairlineWidth, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, fontSize: 17 },
  hit: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 10, padding: 10, gap: 4 },
});
