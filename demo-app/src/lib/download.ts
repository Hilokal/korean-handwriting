export function downloadBlob(
  content: string,
  filename: string,
  mime: string,
): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Safe-ish filename from user text: keep Hangul/alnum, cap length. */
export function slugify(text: string): string {
  const cleaned = Array.from(text)
    .filter((ch) => /[\p{L}\p{N}]/u.test(ch))
    .join("")
    .slice(0, 24);
  return cleaned || "handwriting";
}
