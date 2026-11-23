/** Small shared pieces: colours that follow the system theme, status pills, buttons, authenticated images. */
import { Image, type ImageStyle } from 'expo-image';
import type { ReactNode } from 'react';
import { ActivityIndicator, Pressable, type StyleProp, StyleSheet, Text, useColorScheme, View, type ViewStyle } from 'react-native';

import type { OcrmdClient } from '@/api/client';

export type Palette = { bg: string; card: string; text: string; muted: string; border: string; accent: string; danger: string };

const LIGHT: Palette = { bg: '#f6f5f2', card: '#ffffff', text: '#1d1d1f', muted: '#6e6e73', border: '#dedcd6', accent: '#2f6fde', danger: '#c62828' };
const DARK: Palette = { bg: '#111214', card: '#1c1d20', text: '#f2f2f4', muted: '#9a9aa1', border: '#2e2f33', accent: '#6ea0ff', danger: '#ff6b6b' };

export function usePalette(): Palette {
  return useColorScheme() === 'dark' ? DARK : LIGHT;
}

/** The same verdict colours the web UI uses (ocrmd/web/static/app.css), so a page reads the same everywhere. */
export const STATUS_COLOR: Record<string, string> = {
  verified: '#2e7d32',
  reviewed: '#1b5e20',
  flagged: '#e0a100',
  suspect: '#d84315',
  illegible: '#c62828',
  blank: '#1565c0',
  unverified: '#757575',
  error: '#c62828',
};

export function StatusPill({ status }: { status: string }) {
  return (
    <View style={[styles.pill, { backgroundColor: STATUS_COLOR[status] ?? '#757575' }]}>
      <Text style={styles.pillText}>{status}</Text>
    </View>
  );
}

export function Button({
  title,
  onPress,
  kind = 'plain',
  disabled,
  busy,
  style,
}: {
  title: string;
  onPress: () => void;
  kind?: 'primary' | 'plain' | 'danger';
  disabled?: boolean;
  busy?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  const p = usePalette();
  const bg = kind === 'primary' ? p.accent : kind === 'danger' ? p.danger : p.card;
  const fg = kind === 'plain' ? p.text : '#fff';
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={disabled || busy}
      style={({ pressed }) => [styles.button, { backgroundColor: bg, borderColor: p.border, opacity: disabled ? 0.45 : pressed ? 0.75 : 1 }, style]}
    >
      {busy ? <ActivityIndicator color={fg} /> : <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>}
    </Pressable>
  );
}

/** An image from the ocrmd server: every request carries the device token. */
export function AuthImage({ client, path, style, contentFit = 'contain' }: { client: OcrmdClient; path: string; style: StyleProp<ImageStyle>; contentFit?: 'contain' | 'cover' }) {
  return (
    <Image
      source={{ uri: client.url(path), headers: client.headers() }}
      style={style}
      contentFit={contentFit}
      cachePolicy="memory-disk"
      transition={120}
      recyclingKey={path}
    />
  );
}

export function Message({ title, detail, children }: { title: string; detail?: string | null; children?: ReactNode }) {
  const p = usePalette();
  return (
    <View style={styles.message}>
      <Text style={[styles.messageTitle, { color: p.text }]}>{title}</Text>
      {detail ? <Text style={[styles.messageDetail, { color: p.muted }]}>{detail}</Text> : null}
      {children}
    </View>
  );
}

export function Loading() {
  return (
    <View style={styles.message}>
      <ActivityIndicator />
    </View>
  );
}

export const styles = StyleSheet.create({
  pill: { borderRadius: 10, paddingHorizontal: 8, paddingVertical: 2, alignSelf: 'flex-start' },
  pillText: { color: '#fff', fontSize: 12, fontWeight: '600' },
  button: { borderRadius: 10, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: 14, paddingVertical: 10, alignItems: 'center', justifyContent: 'center', minHeight: 44 },
  buttonText: { fontSize: 16, fontWeight: '600' },
  message: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 12 },
  messageTitle: { fontSize: 18, fontWeight: '600', textAlign: 'center' },
  messageDetail: { fontSize: 15, textAlign: 'center' },
});
