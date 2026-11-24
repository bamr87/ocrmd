/**
 * The page dialect's disagreement marks, for display. `reconcile` writes each one as
 * `==[model reading]==[^ocrmd-n]` with a footnote `[^ocrmd-n]: ocr="…" llm="…" — unresolved`
 * (ocrmd/review.py); the app shows the reading highlighted and leaves the footnotes as text.
 */
const MARK = /==\[([\s\S]*?)\]==\[\^ocrmd-(\d+)\]/g;

export type Part = { text: string; mark?: number };

export function splitMarks(body: string): Part[] {
  const out: Part[] = [];
  let last = 0;
  for (const m of body.matchAll(MARK)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ text: body.slice(last, at) });
    out.push({ text: m[1] ?? '', mark: Number(m[2]) });
    last = at + m[0].length;
  }
  if (last < body.length) out.push({ text: body.slice(last) });
  return out;
}
