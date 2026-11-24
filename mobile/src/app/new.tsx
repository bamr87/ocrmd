/** Start a document: a slug (the folder name on the computer) and a title, then straight to the camera. */
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { ScrollView, StyleSheet, Text, TextInput } from 'react-native';

import { ApiError } from '@/api/client';
import { Button, usePalette } from '@/components/ui';
import { SLUG_RE, slugify } from '@/lib/slug';
import { useClient } from '@/state/session';

export default function NewDocument() {
  const client = useClient();
  const router = useRouter();
  const p = usePalette();
  const [title, setTitle] = useState('');
  const [slug, setSlug] = useState('');
  const [author, setAuthor] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const effective = slug || slugify(title);

  const create = async () => {
    if (!SLUG_RE.test(effective)) return setError('A slug is letters, digits, dots, dashes and underscores, starting with a letter or digit');
    setBusy(true);
    setError(null);
    try {
      await client.createDocument({ slug: effective, title: title || null, author: author ? author.split(',').map((a) => a.trim()).filter(Boolean) : [] });
      router.replace({ pathname: '/capture', params: { doc: effective } });
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };

  const field = [s.input, { color: p.text, borderColor: p.border, backgroundColor: p.card }];
  return (
    <ScrollView style={{ backgroundColor: p.bg }} contentContainerStyle={{ padding: 20, gap: 12 }} keyboardShouldPersistTaps="handled">
      <Text style={[s.label, { color: p.muted }]}>Title</Text>
      <TextInput value={title} onChangeText={setTitle} placeholder="Pride and Prejudice" placeholderTextColor={p.muted} style={field} autoFocus />
      <Text style={[s.label, { color: p.muted }]}>Author(s)</Text>
      <TextInput value={author} onChangeText={setAuthor} placeholder="Jane Austen" placeholderTextColor={p.muted} style={field} />
      <Text style={[s.label, { color: p.muted }]}>Slug</Text>
      <TextInput value={slug} onChangeText={(t) => setSlug(t.toLowerCase().replace(/[^a-z0-9._-]+/g, '-'))} placeholder={slugify(title) || 'pride-and-prejudice'} placeholderTextColor={p.muted} autoCapitalize="none" autoCorrect={false} style={field} />
      <Text style={{ color: p.muted, fontSize: 13 }}>The folder the pages land in on the computer: documents/{effective || '…'}/</Text>
      {error ? <Text style={{ color: p.danger }}>{error}</Text> : null}
      <Button title="Create and scan" kind="primary" busy={busy} onPress={create} />
    </ScrollView>
  );
}

const s = StyleSheet.create({
  label: { fontSize: 13, textTransform: 'uppercase', marginBottom: -6 },
  input: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, fontSize: 17 },
});
