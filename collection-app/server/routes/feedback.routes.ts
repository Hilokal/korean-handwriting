import type { NextFunction, Request, Response } from "express";
import { Router } from "express";
import { requireAdmin } from "../auth.js";
import { db } from "../db.js";
import { syllableCounts } from "../hangul.js";

// Feedback from the public demo site (hangul.ink). The POST endpoint is the
// only unauthenticated write in the app: cross-origin, no session — so it is
// origin-checked, rate-limited, and strictly validated.

const DEFAULT_ORIGINS = [
  "https://hangul.ink",
  "https://www.hangul.ink",
  "https://hangeul.ink",
  "http://localhost:5174", // demo-app vite dev
];
const ALLOWED_ORIGINS = process.env.FEEDBACK_ORIGINS
  ? process.env.FEEDBACK_ORIGINS.split(",").map((s) => s.trim())
  : DEFAULT_ORIGINS;

function cors(req: Request, res: Response, next: NextFunction): void {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Max-Age", "86400");
  }
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }
  next();
}

// In-memory rate limit (single process by design, like login's).
const submissions = new Map<string, { count: number; resetAt: number }>();
const MAX_PER_HOUR = 30;

function rateLimit(req: Request, res: Response, next: NextFunction): void {
  const ip = req.ip ?? "unknown";
  const now = Date.now();
  const entry = submissions.get(ip);
  if (!entry || entry.resetAt < now) {
    submissions.set(ip, { count: 1, resetAt: now + 3600_000 });
  } else if (++entry.count > MAX_PER_HOUR) {
    res.status(429).json({ error: "too many submissions, try again later" });
    return;
  }
  next();
}

const MAX_TEXT = 100;
const MAX_COMMENT = 1000;
const MAX_DOTS = 10000;

export const feedbackRoutes = Router();
feedbackRoutes.use(cors);

feedbackRoutes.post("/", rateLimit, (req, res) => {
  const b = req.body ?? {};
  const dots = b.dots;
  const valid =
    (b.rating === "good" || b.rating === "bad") &&
    typeof b.text === "string" &&
    b.text.trim().length > 0 &&
    b.text.length <= MAX_TEXT &&
    Number.isFinite(b.temperature) &&
    b.temperature >= 0 &&
    b.temperature <= 10 &&
    Number.isFinite(b.bias) &&
    b.bias >= 0 &&
    b.bias <= 10 &&
    Number.isInteger(b.seed) &&
    typeof b.modelVersion === "string" &&
    b.modelVersion.length <= 40 &&
    Array.isArray(dots) &&
    dots.length > 0 &&
    dots.length <= MAX_DOTS &&
    dots.every(
      (d: unknown) =>
        typeof d === "object" &&
        d !== null &&
        Number.isFinite((d as { x: unknown }).x) &&
        Number.isFinite((d as { y: unknown }).y) &&
        ((d as { penState: unknown }).penState === 0 ||
          (d as { penState: unknown }).penState === 1),
    ) &&
    (b.comment === undefined || (typeof b.comment === "string" && b.comment.length <= MAX_COMMENT)) &&
    (b.locale === undefined || b.locale === "en" || b.locale === "ko");
  if (!valid) {
    res.status(400).json({ error: "invalid feedback payload" });
    return;
  }

  db.prepare(
    `INSERT INTO demo_feedback
       (rating, text, temperature, bias, seed, model_version, dots_json, comment, locale)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    b.rating,
    b.text,
    b.temperature,
    b.bias,
    b.seed,
    b.modelVersion,
    JSON.stringify(
      dots.map((d: { x: number; y: number; penState: 0 | 1 }) => ({
        x: d.x,
        y: d.y,
        penState: d.penState,
      })),
    ),
    b.comment?.trim() || null,
    b.locale ?? null,
  );
  res.json({ ok: true });
});

// --- Admin review ---

export const feedbackAdminRoutes = Router();
feedbackAdminRoutes.use(requireAdmin);

const PAGE_SIZE = 20;

feedbackAdminRoutes.get("/", (req, res) => {
  const conditions: string[] = [];
  const params: unknown[] = [];
  if (req.query.status) {
    conditions.push("status = ?");
    params.push(String(req.query.status));
  }
  if (req.query.rating) {
    conditions.push("rating = ?");
    params.push(String(req.query.rating));
  }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const page = Math.max(0, Number(req.query.page ?? 0));
  const total = (
    db.prepare(`SELECT COUNT(*) AS n FROM demo_feedback ${where}`).get(...params) as {
      n: number;
    }
  ).n;
  const feedback = db
    .prepare(
      `SELECT id, rating, text, temperature, bias, seed, model_version, dots_json,
              comment, locale, status, created_at
       FROM demo_feedback ${where}
       ORDER BY id DESC LIMIT ? OFFSET ?`,
    )
    .all(...params, PAGE_SIZE, page * PAGE_SIZE);
  res.json({ feedback, total, pageSize: PAGE_SIZE });
});

feedbackAdminRoutes.post("/:id/review", (req, res) => {
  const result = db
    .prepare("UPDATE demo_feedback SET status = 'reviewed' WHERE id = ?")
    .run(req.params.id);
  res.json({ ok: result.changes > 0 });
});

// Close the loop: push the text the model failed on into the assignment pool,
// so workers record it and it lands in the next training export.
feedbackAdminRoutes.post("/:id/promote", (req, res) => {
  const row = db
    .prepare("SELECT text FROM demo_feedback WHERE id = ?")
    .get(req.params.id) as { text: string } | undefined;
  if (!row) {
    res.status(404).json({ error: "no such feedback" });
    return;
  }
  const promote = db.transaction(() => {
    const result = db
      .prepare(
        "INSERT OR IGNORE INTO sentences (text, source) VALUES (?, 'demo-feedback')",
      )
      .run(row.text);
    if (result.changes > 0) {
      const sentenceId = result.lastInsertRowid as number;
      const insertSyllable = db.prepare(
        `INSERT INTO sentence_syllables (syllable, sentence_id, occurrences)
         VALUES (?, ?, ?)`,
      );
      for (const [syllable, occurrences] of syllableCounts(row.text)) {
        insertSyllable.run(syllable, sentenceId, occurrences);
      }
    }
    db.prepare("UPDATE demo_feedback SET status = 'promoted' WHERE id = ?").run(
      req.params.id,
    );
    return result.changes > 0;
  });
  const inserted = promote();
  res.json({ ok: true, alreadyExisted: !inserted });
});

feedbackAdminRoutes.delete("/:id", (req, res) => {
  const result = db
    .prepare("DELETE FROM demo_feedback WHERE id = ?")
    .run(req.params.id);
  res.json({ ok: result.changes > 0 });
});
