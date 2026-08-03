import { Router } from "express";
import { db } from "../db.js";
import { createInvite, requireAdmin } from "../auth.js";
import { rebuildCoverage } from "../coverage.js";

export const adminRoutes = Router();
adminRoutes.use(requireAdmin);

function inviteUrl(req: { headers: Record<string, unknown> }, token: string): string {
  const origin =
    process.env.APP_ORIGIN ??
    `${(req.headers["x-forwarded-proto"] as string) ?? "http"}://${req.headers.host}`;
  return `${origin}/invite/${token}`;
}

adminRoutes.post("/users", (req, res) => {
  const { name, username } = req.body ?? {};
  const normalized = typeof username === "string" ? username.trim().toLowerCase() : "";
  if (
    typeof name !== "string" ||
    !name.trim() ||
    normalized.length < 3 ||
    /\s/.test(normalized)
  ) {
    res.status(400).json({ error: "name and username (3+ chars, no spaces) required" });
    return;
  }
  const existing = db.prepare("SELECT 1 FROM users WHERE username = ?").get(normalized);
  if (existing) {
    res.status(409).json({ error: "a user with that username already exists" });
    return;
  }
  const userId = db
    .prepare("INSERT INTO users (name, username) VALUES (?, ?)")
    .run(name.trim(), normalized).lastInsertRowid as number;
  const token = createInvite(userId);
  res.json({ userId, inviteUrl: inviteUrl(req, token) });
});

adminRoutes.post("/users/:id/invite", (req, res) => {
  const user = db.prepare("SELECT id FROM users WHERE id = ?").get(req.params.id);
  if (!user) {
    res.status(404).json({ error: "no such user" });
    return;
  }
  const token = createInvite(Number(req.params.id));
  res.json({ inviteUrl: inviteUrl(req, token) });
});

adminRoutes.post("/users/:id/disable", (req, res) => {
  const result = db
    .prepare("UPDATE users SET disabled = 1 WHERE id = ? AND is_admin = 0")
    .run(req.params.id);
  db.prepare("DELETE FROM sessions WHERE user_id = ?").run(req.params.id);
  res.json({ ok: result.changes > 0 });
});

// --- Recordings browser + QA ---

const RECORDINGS_PAGE_SIZE = 20;

adminRoutes.get("/recordings", (req, res) => {
  const conditions: string[] = [];
  const params: unknown[] = [];
  if (req.query.userId) {
    conditions.push("r.user_id = ?");
    params.push(Number(req.query.userId));
  }
  if (req.query.sentenceId) {
    conditions.push("r.sentence_id = ?");
    params.push(Number(req.query.sentenceId));
  }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const page = Math.max(0, Number(req.query.page ?? 0));
  const total = (
    db.prepare(`SELECT COUNT(*) AS n FROM recordings r ${where}`).get(...params) as {
      n: number;
    }
  ).n;
  const recordings = db
    .prepare(
      `SELECT r.id, r.user_id, u.name AS userName, r.sentence_id,
              COALESCE(r.chunk_text, s.text) AS text, r.chunk_index,
              r.dot_count, r.status, r.created_at
       FROM recordings r
       JOIN users u ON u.id = r.user_id
       JOIN sentences s ON s.id = r.sentence_id
       ${where}
       ORDER BY r.id DESC
       LIMIT ${RECORDINGS_PAGE_SIZE} OFFSET ${page * RECORDINGS_PAGE_SIZE}`,
    )
    .all(...params);
  res.json({ recordings, total, pageSize: RECORDINGS_PAGE_SIZE });
});

adminRoutes.get("/recordings/:id", (req, res) => {
  const row = db
    .prepare(
      `SELECT r.id, r.user_id, u.name AS userName, r.sentence_id,
              COALESCE(r.chunk_text, s.text) AS text, r.chunk_index,
              r.dot_count, r.status, r.created_at, r.pen_mac, r.dots_json
       FROM recordings r
       JOIN users u ON u.id = r.user_id
       JOIN sentences s ON s.id = r.sentence_id
       WHERE r.id = ?`,
    )
    .get(req.params.id) as { dots_json: string } | undefined;
  if (!row) {
    res.status(404).json({ error: "no such recording" });
    return;
  }
  const { dots_json, ...rest } = row;
  res.json({ ...rest, dots: JSON.parse(dots_json) });
});

function setRecordingStatus(id: string, status: "ok" | "rejected"): boolean {
  const row = db
    .prepare("SELECT user_id, status FROM recordings WHERE id = ?")
    .get(id) as { user_id: number; status: string } | undefined;
  if (!row) return false;
  if (row.status !== status) {
    db.prepare("UPDATE recordings SET status = ? WHERE id = ?").run(status, id);
    rebuildCoverage(row.user_id);
  }
  return true;
}

adminRoutes.post("/recordings/:id/reject", (req, res) => {
  if (!setRecordingStatus(req.params.id, "rejected")) {
    res.status(404).json({ error: "no such recording" });
    return;
  }
  res.json({ ok: true });
});

adminRoutes.post("/recordings/:id/restore", (req, res) => {
  if (!setRecordingStatus(req.params.id, "ok")) {
    res.status(404).json({ error: "no such recording" });
    return;
  }
  res.json({ ok: true });
});

// --- Progress dashboard ---

adminRoutes.get("/progress", (_req, res) => {
  const totalSyllables = (
    db.prepare("SELECT COUNT(DISTINCT syllable) AS n FROM sentence_syllables").get() as {
      n: number;
    }
  ).n;
  // Bucketed histogram of per-syllable recording counts for each worker.
  const buckets = db
    .prepare(
      `SELECT user_id AS userId,
              CASE WHEN count >= 5 THEN '5+' ELSE CAST(count AS TEXT) END AS bucket,
              COUNT(*) AS syllables
       FROM user_syllable_counts
       WHERE count > 0
       GROUP BY user_id, bucket`,
    )
    .all();
  const users = db
    .prepare(
      `SELECT u.id, u.name,
              (SELECT COUNT(*) FROM recordings r WHERE r.user_id = u.id AND r.status = 'ok') AS recordings,
              (SELECT COUNT(*) FROM user_syllable_counts c WHERE c.user_id = u.id AND c.count > 0) AS syllablesCovered
       FROM users u
       WHERE u.disabled = 0
       ORDER BY u.id`,
    )
    .all();
  res.json({ totalSyllables, users, buckets });
});

// --- Reported sentences ---

adminRoutes.get("/reports", (_req, res) => {
  const reports = db
    .prepare(
      `SELECT a.id, a.sentence_id AS sentenceId, s.text, s.active,
              a.problem_note AS note, a.resolved_at AS reportedAt, u.name AS userName
       FROM assignments a
       JOIN sentences s ON s.id = a.sentence_id
       JOIN users u ON u.id = a.user_id
       WHERE a.status = 'reported'
       ORDER BY a.resolved_at DESC`,
    )
    .all();
  res.json({ reports });
});

adminRoutes.post("/sentences/:id/deactivate", (req, res) => {
  const result = db
    .prepare("UPDATE sentences SET active = 0 WHERE id = ?")
    .run(req.params.id);
  res.json({ ok: result.changes > 0 });
});

adminRoutes.post("/sentences/:id/activate", (req, res) => {
  const result = db
    .prepare("UPDATE sentences SET active = 1 WHERE id = ?")
    .run(req.params.id);
  res.json({ ok: result.changes > 0 });
});

// --- Maintenance ---

adminRoutes.post("/coverage/rebuild", (_req, res) => {
  rebuildCoverage();
  res.json({ ok: true });
});

adminRoutes.get("/users", (_req, res) => {
  const totalSyllables = (
    db.prepare("SELECT COUNT(DISTINCT syllable) AS n FROM sentence_syllables").get() as {
      n: number;
    }
  ).n;
  const users = db
    .prepare(
      `SELECT u.id, u.name, u.username, u.is_admin, u.disabled, u.created_at,
              u.password_hash IS NOT NULL AS accepted,
              (SELECT COUNT(*) FROM recordings r WHERE r.user_id = u.id AND r.status = 'ok') AS recordings,
              (SELECT COUNT(*) FROM user_syllable_counts c WHERE c.user_id = u.id AND c.count > 0) AS syllablesCovered,
              (SELECT MAX(created_at) FROM recordings r WHERE r.user_id = u.id) AS lastRecordingAt
       FROM users u ORDER BY u.created_at`,
    )
    .all();
  res.json({ users, totalSyllables });
});
