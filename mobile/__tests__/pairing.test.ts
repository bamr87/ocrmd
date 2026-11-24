import { displayCode, isValidCode, normalizeCode, normalizeServer, parsePairLink } from '../src/lib/pairing';
import { slugify, SLUG_RE } from '../src/lib/slug';
import { splitMarks } from '../src/lib/marks';

describe('pairing links', () => {
  it('reads the link `ocrmd pair` puts in its QR code', () => {
    // exactly what ocrmd/devices.py pair_link() writes (tests/test_mobile_api.py pins the server side)
    expect(parsePairLink('ocrmd://pair?server=http%3A%2F%2F192.168.1.5%3A8765&code=ABCD2345')).toEqual({
      server: 'http://192.168.1.5:8765',
      code: 'ABCD2345',
    });
  });

  it('refuses anything that is not an ocrmd pairing link', () => {
    expect(parsePairLink('https://example.com/?code=ABCD2345')).toBeNull();
    expect(parsePairLink('ocrmd://pair?server=http%3A%2F%2Fhost&code=ABC')).toBeNull(); // too short
    expect(parsePairLink('ocrmd://pair?code=ABCD2345')).toBeNull(); // no server
  });

  it('accepts a server however it was typed', () => {
    expect(normalizeServer('192.168.1.5')).toBe('http://192.168.1.5:8765');
    expect(normalizeServer(' mac.local:9000/ ')).toBe('http://mac.local:9000');
    expect(normalizeServer('http://10.0.0.2:8765/api')).toBe('http://10.0.0.2:8765');
    expect(normalizeServer('')).toBeNull();
  });

  it('treats a code without regard to case, spaces or the dash', () => {
    expect(normalizeCode(' abcd-2345 ')).toBe('ABCD2345');
    expect(displayCode('abcd2345')).toBe('ABCD-2345');
    expect(isValidCode('abcd-2345')).toBe(true);
    expect(isValidCode('ABCD-0O1I')).toBe(false); // look-alikes are never generated
  });
});

describe('slugs and marks', () => {
  it('makes a slug the server accepts from a title', () => {
    expect(slugify('Pride and Prejudice (1813)')).toBe('pride-and-prejudice-1813');
    expect(slugify('Économie générale')).toBe('economie-generale');
    expect(SLUG_RE.test(slugify('  --The Book--  '))).toBe(true);
  });

  it('splits a body into text and disagreement marks', () => {
    const body = 'total ==[44.80]==[^ocrmd-1] paid\n\n[^ocrmd-1]: ocr="44.60" llm="44.80" — unresolved';
    expect(splitMarks(body)).toEqual([
      { text: 'total ' },
      { text: '44.80', mark: 1 },
      { text: ' paid\n\n[^ocrmd-1]: ocr="44.60" llm="44.80" — unresolved' },
    ]);
    expect(splitMarks('no marks')).toEqual([{ text: 'no marks' }]);
  });
});
