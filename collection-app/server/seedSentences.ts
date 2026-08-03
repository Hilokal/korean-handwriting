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

  const insertSentence = db.prepare(
    "INSERT OR IGNORE INTO sentences (text, source) VALUES (?, ?)",
  );
  const insertSyllable = db.prepare(
    "INSERT INTO sentence_syllables (syllable, sentence_id, occurrences) VALUES (?, ?, ?)",
  );

  const lines = fs.readFileSync(path, "utf-8").split("\n").filter(Boolean);
  const seedAll = db.transaction(() => {
    for (const line of lines) {
      const { text, source } = JSON.parse(line) as {
        text: string;
        source?: string;
      };
      const result = insertSentence.run(text, source ?? null);
      if (result.changes === 0) continue; // duplicate text
      const sentenceId = result.lastInsertRowid as number;
      for (const [syllable, occurrences] of syllableCounts(text)) {
        insertSyllable.run(syllable, sentenceId, occurrences);
      }
    }
  });
  seedAll();
  const count = (db.prepare("SELECT COUNT(*) AS n FROM sentences").get() as { n: number }).n;
  console.log(`seeded ${count} sentences from ${path}`);
}
