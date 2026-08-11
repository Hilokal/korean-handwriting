import { db } from "./db.js";
import { chunkText } from "./hangul.js";

export interface Assignment {
  assignmentId: number;
  sentenceId: number;
  text: string;
  chunks: string[];
  nextChunkIndex: number;
}

const TARGET_SYLLABLES = 20;
const CANDIDATE_LIMIT = 200;

// Max characters per prompt chunk (including spaces) so every chunk fits one
// physical line of the workers' Ncode notebooks. Jon measured 32-34 chars per
// line in his own (non-native) handwriting; 30 leaves margin for workers who
// write larger. Chunks are frozen per assignment, so changing this only
// affects new assignments.
const CHUNK_MAX_CHARS = Number(process.env.CHUNK_MAX_CHARS ?? 30);

interface ActiveRow {
  assignmentId: number;
  sentenceId: number;
  text: string;
  chunks: string | null;
  submitted: number;
}

/** Load or lazily backfill the frozen chunk list for an assignment. */
function assignmentChunks(row: ActiveRow): string[] {
  if (row.chunks) return JSON.parse(row.chunks) as string[];
  const chunks = chunkText(row.text, CHUNK_MAX_CHARS);
  db.prepare("UPDATE assignments SET chunks = ? WHERE id = ?").run(
    JSON.stringify(chunks),
    row.assignmentId,
  );
  return chunks;
}

/**
 * Coverage-driven sentence assignment, scoped per user: pick the sentence that
 * most improves THIS user's syllable coverage. The dataset goal is a balance
 * of handwriting styles per syllable, so each worker chases their own
 * least-recorded syllables rather than the aggregate ones.
 *
 * Long sentences are recorded chunk by chunk (one physical line each);
 * the lease stays active until every chunk is submitted.
 */
export function getOrAssignNext(userId: number): Assignment | null {
  const active = db
    .prepare(
      `SELECT a.id AS assignmentId, s.id AS sentenceId, s.text, a.chunks,
              (SELECT COUNT(*) FROM recordings r WHERE r.assignment_id = a.id) AS submitted
       FROM assignments a JOIN sentences s ON s.id = a.sentence_id
       WHERE a.user_id = ? AND a.status = 'active'`,
    )
    .get(userId) as ActiveRow | undefined;
  if (active) {
    const chunks = assignmentChunks(active);
    if (active.submitted >= chunks.length) {
      // Shouldn't happen (completion is transactional with the last submit),
      // but self-heal rather than wedge the worker's queue.
      db.prepare(
        "UPDATE assignments SET status = 'completed', resolved_at = datetime('now') WHERE id = ?",
      ).run(active.assignmentId);
    } else {
      return {
        assignmentId: active.assignmentId,
        sentenceId: active.sentenceId,
        text: active.text,
        chunks,
        nextChunkIndex: active.submitted,
      };
    }
  }

  // Curated batches jump the queue: any active priority > 0 sentence this
  // user hasn't finished (or skipped/reported) is served before the
  // coverage-driven pool -- highest priority tier first, best coverage gain
  // within the tier. Coverage targeting alone would bury curated sentences
  // whose syllables are already well covered, which is exactly the case for
  // register-gap batches like the 해요체 set (common syllables, rare position).
  const prioRows = db
    .prepare(
      `SELECT id, priority FROM sentences
       WHERE active = 1 AND priority > 0
         AND id NOT IN (
           SELECT sentence_id FROM assignments
           WHERE user_id = ? AND status IN ('completed','skipped','reported'))
       ORDER BY priority DESC
       LIMIT ${CANDIDATE_LIMIT}`,
    )
    .all(userId) as Array<{ id: number; priority: number }>;
  if (prioRows.length > 0) {
    const topTier = prioRows.filter((r) => r.priority === prioRows[0].priority);
    const best = pickByCoverage(
      userId,
      topTier.map((r) => r.id),
    );
    if (best !== null) return createAssignment(userId, best);
  }

  // This user's least-covered syllables (never-recorded count as 0).
  const targets = db
    .prepare(
      `SELECT s.syllable
       FROM (SELECT DISTINCT syllable FROM sentence_syllables) s
       LEFT JOIN user_syllable_counts u
         ON u.user_id = ? AND u.syllable = s.syllable
       ORDER BY COALESCE(u.count, 0) ASC, RANDOM()
       LIMIT ${TARGET_SYLLABLES}`,
    )
    .all(userId)
    .map((r) => (r as { syllable: string }).syllable);
  if (targets.length === 0) return null; // no sentences seeded

  const placeholders = targets.map(() => "?").join(",");
  const candidateSql = (excludeDone: boolean) => `
    SELECT DISTINCT ss.sentence_id
    FROM sentence_syllables ss
    JOIN sentences s ON s.id = ss.sentence_id AND s.active = 1
    WHERE ss.syllable IN (${placeholders})
    ${
      excludeDone
        ? `AND ss.sentence_id NOT IN (
             SELECT sentence_id FROM assignments
             WHERE user_id = ? AND status IN ('completed','skipped','reported'))`
        : ""
    }
    LIMIT ${CANDIDATE_LIMIT}`;

  let candidates = db
    .prepare(candidateSql(true))
    .all(...targets, userId)
    .map((r) => (r as { sentence_id: number }).sentence_id);
  if (candidates.length === 0) {
    // User has done every sentence containing their rarest syllables: allow
    // repeats — multiple recordings of a sentence are still valid style samples.
    candidates = db
      .prepare(candidateSql(false))
      .all(...targets)
      .map((r) => (r as { sentence_id: number }).sentence_id);
  }
  if (candidates.length === 0) return null;

  const best = pickByCoverage(userId, candidates);
  if (best === null) return null;
  return createAssignment(userId, best);
}

/** Among candidate sentence ids, the one with the best coverage gain for this
 *  user. score = Σ occurrences/(1+count), normalized by sqrt(length) so wholly
 *  under-covered sentences win without always favoring the longest ones. */
function pickByCoverage(userId: number, candidates: number[]): number | null {
  if (candidates.length === 0) return null;

  // Score in JS with the user's full coverage map (≤ ~1500 rows).
  const counts = new Map<string, number>(
    (
      db
        .prepare("SELECT syllable, count FROM user_syllable_counts WHERE user_id = ?")
        .all(userId) as Array<{ syllable: string; count: number }>
    ).map((r) => [r.syllable, r.count]),
  );
  const candidatePlaceholders = candidates.map(() => "?").join(",");
  const rows = db
    .prepare(
      `SELECT sentence_id, syllable, occurrences FROM sentence_syllables
       WHERE sentence_id IN (${candidatePlaceholders})`,
    )
    .all(...candidates) as Array<{
    sentence_id: number;
    syllable: string;
    occurrences: number;
  }>;

  const scores = new Map<number, { gain: number; length: number }>();
  for (const row of rows) {
    const entry = scores.get(row.sentence_id) ?? { gain: 0, length: 0 };
    entry.gain += row.occurrences / (1 + (counts.get(row.syllable) ?? 0));
    entry.length += row.occurrences;
    scores.set(row.sentence_id, entry);
  }
  let best: number | null = null;
  let bestScore = -Infinity;
  for (const [sentenceId, { gain, length }] of scores) {
    const score = (gain / Math.sqrt(length)) * (1 + Math.random() * 0.01);
    if (score > bestScore) {
      bestScore = score;
      best = sentenceId;
    }
  }
  return best;
}

function createAssignment(userId: number, sentenceId: number): Assignment {
  const text = (
    db.prepare("SELECT text FROM sentences WHERE id = ?").get(sentenceId) as {
      text: string;
    }
  ).text;
  const chunks = chunkText(text, CHUNK_MAX_CHARS);
  const assignmentId = db
    .prepare("INSERT INTO assignments (user_id, sentence_id, chunks) VALUES (?, ?, ?)")
    .run(userId, sentenceId, JSON.stringify(chunks)).lastInsertRowid as number;
  return { assignmentId, sentenceId, text, chunks, nextChunkIndex: 0 };
}
