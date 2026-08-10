// Port of handwriting-generation/tokenizer.py — keep the two in sync.
//
// tokenize(text) -> one [leading, vowel, trailing, symbol] row per unit.
// Hangul syllables decompose into their three jamo (symbol 0); space and
// punctuation get jamo slots 0 and a symbol id >= 1; anything else maps to the
// unknown-symbol id rather than throwing. The last row is always the EOT unit
// (trained end-of-text position; generation stops when the window's peak
// reaches it), so U = characters + 1.

export const LeadingCount = 19;
export const VowelCount = 21;
export const TrailingCount = 28;

const SYMBOLS = [" ", ".", ",", "!", "?"];
const UNKNOWN_SYMBOL_ID = SYMBOLS.length + 1;
export const EOT_SYMBOL_ID = SYMBOLS.length + 2;

const SyllableBase = 0xac00;
const SyllableCount = LeadingCount * VowelCount * TrailingCount;

export type TokenRow = [number, number, number, number];

export function isHangulSyllable(ch: string): boolean {
  const idx = ch.codePointAt(0)! - SyllableBase;
  return idx >= 0 && idx < SyllableCount;
}

export function tokenize(text: string): TokenRow[] {
  const rows: TokenRow[] = [];
  for (const ch of text) {
    const idx = ch.codePointAt(0)! - SyllableBase;
    if (idx >= 0 && idx < SyllableCount) {
      const nCount = VowelCount * TrailingCount;
      const leading = Math.floor(idx / nCount);
      const vowel = Math.floor((idx % nCount) / TrailingCount);
      const trailing = idx % TrailingCount;
      rows.push([leading, vowel, trailing, 0]);
    } else {
      const s = SYMBOLS.indexOf(ch);
      rows.push([0, 0, 0, s >= 0 ? s + 1 : UNKNOWN_SYMBOL_ID]);
    }
  }
  rows.push([0, 0, 0, EOT_SYMBOL_ID]);
  return rows;
}
