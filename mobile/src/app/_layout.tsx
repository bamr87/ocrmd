import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useColorScheme } from 'react-native';

import { SessionProvider } from '@/state/session';
import { UploadsProvider } from '@/state/uploads';

export default function RootLayout() {
  const scheme = useColorScheme();
  return (
    <ThemeProvider value={scheme === 'dark' ? DarkTheme : DefaultTheme}>
      <SessionProvider>
        <UploadsProvider>
          <StatusBar style="auto" />
          <Stack>
            <Stack.Screen name="index" options={{ title: 'Library' }} />
            <Stack.Screen name="pair" options={{ title: 'Pair with ocrmd' }} />
            <Stack.Screen name="capture" options={{ title: 'Scan pages', headerShown: false }} />
            <Stack.Screen name="new" options={{ title: 'New document', presentation: 'modal' }} />
            <Stack.Screen name="search" options={{ title: 'Search' }} />
            <Stack.Screen name="settings" options={{ title: 'Settings' }} />
            <Stack.Screen name="doc/[slug]" options={{ title: 'Document' }} />
            <Stack.Screen name="page/[slug]/[seq]" options={{ title: 'Page' }} />
          </Stack>
        </UploadsProvider>
      </SessionProvider>
    </ThemeProvider>
  );
}
