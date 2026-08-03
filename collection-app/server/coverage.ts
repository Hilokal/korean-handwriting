import { db } from "./db.js";
import { syllableCounts } from "./hangul.js";

const upsert = () =>
  db.prepare(
    `INSERT INTO user_syllable_counts (user_id, syllable, count) VALUES (?, ?, ?)
     ON CONFLICT (user_id, syllable) DO UPDATE SET count = count + excluded.count`,
  );

/** Credit a user's coverage for one recorded text (a chunk, or whole sentence). */
export function bumpCoverageForText(userId: number, text: string): void {
  const stmt = upsert();
  for (const [syllable, count] of syllableCounts(text)) {
    stmt.run(userId, syllable, count);
  }
}

/**
 * Recompute user_syllable_counts from scratch for one user (or all users)
 * from their non-rejected recordings. Counts derive from each recording's
 * chunk_text (what was actually written); recordings from before chunking
 * fall back to the full sentence text. Used after QA rejects/restores and as
 * the admin "rebuild" action if the materialized counts ever drift.
 */
export function rebuildCoverage(userId?: number): void {
  const rows = db
    .prepare(
      `SELECT r.user_id AS userId, COALESCE(r.chunk_text, s.text) AS text
       FROM recordings r JOIN sentences s ON s.id = r.sentence_id
       WHERE r.status = 'ok' ${userId !== undefined ? "AND r.user_id = ?" : ""}`,
    )
    .all(...(userId !== undefined ? [userId] : [])) as Array<{
    userId: number;
    text: string;
  }>;

  const totals = new Map<number, Map<string, number>>();
  for (const row of rows) {
    let user = totals.get(row.userId);
    if (!user) totals.set(row.userId, (user = new Map()));
    for (const [syllable, count] of syllableCounts(row.text)) {
      user.set(syllable, (user.get(syllable) ?? 0) + count);
    }
  }

  const insert = db.prepare(
    "INSERT INTO user_syllable_counts (user_id, syllable, count) VALUES (?, ?, ?)",
  );
  const rebuild = db.transaction(() => {
    if (userId !== undefined) {
      db.prepare("DELETE FROM user_syllable_counts WHERE user_id = ?").run(userId);
    } else {
      db.prepare("DELETE FROM user_syllable_counts").run();
    }
    for (const [uid, syllables] of totals) {
      for (const [syllable, count] of syllables) {
        insert.run(uid, syllable, count);
      }
    }
  });
  rebuild();
}
