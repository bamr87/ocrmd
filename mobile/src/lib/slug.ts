/** "Pride and Prejudice (1813)" → "pride-and-prejudice-1813": a title as a document slug the server accepts. */
export const SLUG_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

export function slugify(title: string): string {
  return title
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^[-._]+|[-._]+$/g, '')
    .slice(0, 64);
}
