// Port of handwriting-generation/tokenizer.py (standard Unicode decomposition).

export const LEADING_COUNT = 19;
export const VOWEL_COUNT = 21;
export const TRAILING_COUNT = 28;

const SYLLABLE_BASE = 0xac00; // 가
const N_COUNT = VOWEL_COUNT * TRAILING_COUNT;
const SYLLABLE_COUNT = LEADING_COUNT * N_COUNT;

export function isHangulSyllable(ch: string): boolean {
  const index = ch.codePointAt(0)! - SYLLABLE_BASE;
  return index >= 0 && index < SYLLABLE_COUNT;
}

/** Returns [leadingIndex, vowelIndex, trailingIndex]; trailing 0 = no 받침. */
export function decompose(ch: string): [number, number, number] {
  const index = ch.codePointAt(0)! - SYLLABLE_BASE;
  if (index < 0 || index >= SYLLABLE_COUNT) {
    throw new Error(`Not a Hangul syllable: ${ch}`);
  }
  return [
    Math.floor(index / N_COUNT),
    Math.floor((index % N_COUNT) / TRAILING_COUNT),
    index % TRAILING_COUNT,
  ];
}

/** Occurrence count of each Hangul syllable in the text. */
export function syllableCounts(text: string): Map<string, number> {
  const counts = new Map<string, number>();
  for (const ch of text) {
    if (isHangulSyllable(ch)) {
      counts.set(ch, (counts.get(ch) ?? 0) + 1);
    }
  }
  return counts;
}

/**
 * Split a sentence into prompt chunks of at most maxChars (including spaces),
 * breaking at word boundaries so a chunk always fits one physical line of the
 * worker's Ncode notebook. Overlong single words (rare) are hard-split.
 */
export function chunkText(text: string, maxChars: number): string[] {
  const chunks: string[] = [];
  let current = "";
  for (let word of text.split(" ")) {
    while (word.length > maxChars) {
      if (current) {
        chunks.push(current);
        current = "";
      }
      chunks.push(word.slice(0, maxChars));
      word = word.slice(maxChars);
    }
    if (!word) continue;
    if (!current) {
      current = word;
    } else if (current.length + 1 + word.length <= maxChars) {
      current += " " + word;
    } else {
      chunks.push(current);
      current = word;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}
