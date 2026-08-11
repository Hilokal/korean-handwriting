import fs from "node:fs";
import { db } from "./db.js";
import { syllableCounts } from "./hangul.js";

const CANDIDATE_PATHS = [
  process.env.SENTENCES_PATH,
  "./seed/all.jsonl", // bundled into the Docker image (npm run seed:copy)
  "../data/reference-texts/all.jsonl", // local dev, repo layout
].filter((p): p is string => Boolean(p));

/** Import reference sentences on first boot. Idempotent: skips if any exist. */
export function seedSentences(): void {
  const existing = db.prepare("SELECT COUNT(*) AS n FROM sentences").get() as {
    n: number;
  };
  if (existing.n > 0) return;

  const path = CANDIDATE_PATHS.find((p) => fs.existsSync(p));
  if (!path) {
    console.warn(
      `no sentences file found (tried: ${CANDIDATE_PATHS.join(", ")}) — skipping seed`,
    );
    return;
  }

  const lines = fs.readFileSync(path, "utf-8").split("\n").filter(Boolean);
  importSentences(
    lines.map((line) => JSON.parse(line) as { text: string; source?: string }),
  );
  const count = (db.prepare("SELECT COUNT(*) AS n FROM sentences").get() as { n: number }).n;
  console.log(`seeded ${count} sentences from ${path}`);
}

/**
 * Bulk-insert sentences into the pool with their syllable index (the
 * assignment queue reads sentence_syllables, so a sentence without index rows
 * would never be assigned). Duplicate texts are skipped -- re-importing a
 * seed file is idempotent. Shared by first-boot seeding and the admin
 * /sentences/import route (curated batches, e.g. curated/haeyo-2026-08.jsonl).
 */
export function importSentences(
  rows: { text: string; source?: string | null }[],
): { inserted: number; skipped: number } {
  const insertSentence = db.prepare(
    "INSERT OR IGNORE INTO sentences (text, source) VALUES (?, ?)",
  );
  const insertSyllable = db.prepare(
    "INSERT INTO sentence_syllables (syllable, sentence_id, occurrences) VALUES (?, ?, ?)",
  );
  let inserted = 0;
  db.transaction(() => {
    for (const { text, source } of rows) {
      const trimmed = text?.trim();
      if (!trimmed) continue;
      const result = insertSentence.run(trimmed, source ?? null);
      if (result.changes === 0) continue; // duplicate text
      inserted++;
      const sentenceId = result.lastInsertRowid as number;
      for (const [syllable, occurrences] of syllableCounts(trimmed)) {
        insertSyllable.run(syllable, sentenceId, occurrences);
      }
    }
  })();
  return { inserted, skipped: rows.length - inserted };
}
