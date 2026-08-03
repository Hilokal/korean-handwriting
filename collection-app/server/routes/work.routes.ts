import { Router } from "express";
import { db } from "../db.js";
import { requireAuth } from "../auth.js";
import { getOrAssignNext } from "../assignment.js";
import { bumpCoverageForText } from "../coverage.js";

export const workRoutes = Router();
workRoutes.use(requireAuth);

const MAX_DOTS = 100_000;

interface SubmitBody {
  startTime: number;
  endTime: number;
  dotCount: number;
  chunkIndex: number;
  dots: Array<{ x: number; y: number; dotType: number; timeStamp: number }>;
  penMac?: string;
  pageInfo?: { section: number; owner: number; book: number; page: number };
}

function validateSubmit(body: unknown): string | null {
  const b = body as SubmitBody;
  if (!b || !Array.isArray(b.dots)) return "dots array required";
  if (b.dots.length === 0) return "no dots recorded";
  if (b.dots.length > MAX_DOTS) return "too many dots";
  if (b.dotCount !== b.dots.length) return "dotCount does not match dots";
  if (typeof b.startTime !== "number" || typeof b.endTime !== "number")
    return "startTime/endTime required";
  if (!Number.isInteger(b.chunkIndex) || b.chunkIndex < 0) return "chunkIndex required";
  for (const dot of b.dots) {
    // dotType range matches the SDK's DotTypes enum: DOWN/MOVE/UP/HOVER/INFO/ERROR.
    if (
      typeof dot.x !== "number" ||
      typeof dot.y !== "number" ||
      !Number.isInteger(dot.dotType) ||
      dot.dotType < 0 ||
      dot.dotType > 5
    ) {
      return "malformed dot";
    }
  }
  return null;
}

/** Look up an assignment only if it belongs to this user and is active. */
function getActiveAssignment(assignmentId: string, userId: number) {
  return db
    .prepare(
      `SELECT id, sentence_id, chunks,
              (SELECT COUNT(*) FROM recordings r WHERE r.assignment_id = assignments.id) AS submitted
       FROM assignments
       WHERE id = ? AND user_id = ? AND status = 'active'`,
    )
    .get(assignmentId, userId) as
    | { id: number; sentence_id: number; chunks: string | null; submitted: number }
    | undefined;
}

workRoutes.get("/next", (req, res) => {
  const assignment = getOrAssignNext(req.user!.id);
  if (!assignment) {
    res.status(404).json({ error: "no sentences available" });
    return;
  }
  res.json(assignment);
});

workRoutes.post("/:id/submit", (req, res) => {
  const assignment = getActiveAssignment(req.params.id, req.user!.id);
  if (!assignment) {
    res.status(404).json({ error: "no such active assignment" });
    return;
  }
  const error = validateSubmit(req.body);
  if (error) {
    res.status(400).json({ error });
    return;
  }
  const body = req.body as SubmitBody;

  // Chunked prompts: the client must submit the chunk the lease expects.
  // chunks is always populated by getOrAssignNext before a client can submit.
  const chunks: string[] = assignment.chunks ? JSON.parse(assignment.chunks) : [];
  const expectedChunk = assignment.submitted;
  if (body.chunkIndex !== expectedChunk) {
    res.status(409).json({ error: `expected chunk ${expectedChunk}` });
    return;
  }
  const chunkTextStr = chunks[expectedChunk] ?? null;
  const isLastChunk = expectedChunk >= chunks.length - 1;

  const submit = db.transaction(() => {
    const recordingId = db
      .prepare(
        `INSERT INTO recordings
           (user_id, sentence_id, assignment_id, chunk_index, chunk_text,
            start_time, end_time, dot_count,
            pen_mac, page_section, page_owner, page_book, page_page, dots_json)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        req.user!.id,
        assignment.sentence_id,
        assignment.id,
        expectedChunk,
        chunkTextStr,
        body.startTime,
        body.endTime,
        body.dots.length,
        body.penMac ?? null,
        body.pageInfo?.section ?? null,
        body.pageInfo?.owner ?? null,
        body.pageInfo?.book ?? null,
        body.pageInfo?.page ?? null,
        JSON.stringify(body.dots),
      ).lastInsertRowid as number;
    if (isLastChunk) {
      db.prepare(
        "UPDATE assignments SET status = 'completed', resolved_at = datetime('now') WHERE id = ?",
      ).run(assignment.id);
    }
    if (chunkTextStr) bumpCoverageForText(req.user!.id, chunkTextStr);
    return recordingId;
  });

  const recordingId = submit();
  // Not last chunk → the same lease continues with the next chunk; last
  // chunk → a fresh assignment. Either way the client just renders `next`.
  const next = getOrAssignNext(req.user!.id);
  res.json({ recordingId, next });
});

workRoutes.post("/:id/skip", (req, res) => {
  const assignment = getActiveAssignment(req.params.id, req.user!.id);
  if (!assignment) {
    res.status(404).json({ error: "no such active assignment" });
    return;
  }
  db.prepare(
    "UPDATE assignments SET status = 'skipped', resolved_at = datetime('now') WHERE id = ?",
  ).run(assignment.id);
  const next = getOrAssignNext(req.user!.id);
  res.json({ next });
});

workRoutes.post("/:id/report", (req, res) => {
  const assignment = getActiveAssignment(req.params.id, req.user!.id);
  if (!assignment) {
    res.status(404).json({ error: "no such active assignment" });
    return;
  }
  const note = typeof req.body?.note === "string" ? req.body.note.slice(0, 500) : null;
  db.prepare(
    `UPDATE assignments SET status = 'reported', problem_note = ?,
     resolved_at = datetime('now') WHERE id = ?`,
  ).run(note, assignment.id);
  const next = getOrAssignNext(req.user!.id);
  res.json({ next });
});
