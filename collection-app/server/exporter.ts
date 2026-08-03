import archiver from "archiver";
import type { Response } from "express";
import { db } from "./db.js";

interface ExportFilters {
  since?: string; // ISO date: only recordings created after this
  userId?: number;
}

interface RecordingRow {
  id: number;
  user_id: number;
  user_name: string;
  sentence_id: number;
  text: string;
  chunk_index: number;
  chunk_text: string | null;
  start_time: number;
  end_time: number;
  dot_count: number;
  pen_mac: string | null;
  dots_json: string;
  created_at: string;
}

const EXPORT_BATCH = 50;

function recordingQuery(filters: ExportFilters, withDots: boolean, afterId?: number) {
  const conditions = ["r.status = 'ok'"];
  const params: unknown[] = [];
  if (filters.since) {
    conditions.push("r.created_at > ?");
    params.push(filters.since);
  }
  if (filters.userId) {
    conditions.push("r.user_id = ?");
    params.push(filters.userId);
  }
  if (afterId !== undefined) {
    conditions.push("r.id > ?");
    params.push(afterId);
  }
  const stmt = db.prepare(
    `SELECT r.id, r.user_id, u.name AS user_name, r.sentence_id, s.text,
            r.chunk_index, r.chunk_text,
            r.start_time, r.end_time, r.dot_count, r.pen_mac, r.created_at
            ${withDots ? ", r.dots_json" : ""}
     FROM recordings r
     JOIN users u ON u.id = r.user_id
     JOIN sentences s ON s.id = r.sentence_id
     WHERE ${conditions.join(" AND ")}
     ORDER BY r.id
     ${afterId !== undefined ? `LIMIT ${EXPORT_BATCH}` : ""}`,
  );
  return { stmt, params };
}

function recordingRows(filters: ExportFilters): RecordingRow[] {
  const { stmt, params } = recordingQuery(filters, false);
  return stmt.all(...params) as RecordingRow[];
}

/**
 * Yield rows in id-keyed batches. dots_json is large, so this bounds memory —
 * and unlike a long-lived stmt.iterate(), it never holds the connection open
 * across awaits (better-sqlite3 rejects concurrent queries mid-iteration,
 * which would 500 any worker submitting during an export).
 */
function* recordingRowsWithDots(filters: ExportFilters): Generator<RecordingRow> {
  let lastId = 0;
  for (;;) {
    const { stmt, params } = recordingQuery(filters, true, lastId);
    const batch = stmt.all(...params) as RecordingRow[];
    if (batch.length === 0) return;
    yield* batch;
    lastId = batch[batch.length - 1].id;
  }
}

export function buildManifest(filters: ExportFilters) {
  return recordingRows(filters).map((r) => ({
    file: `recordings/u${r.user_id}/s${r.sentence_id}/rec-${r.id}.json`,
    recordingId: r.id,
    userId: r.user_id,
    userName: r.user_name,
    sentenceId: r.sentence_id,
    sentenceText: r.text,
    // What was physically written: the chunk if this recording came from a
    // chunked prompt, else the whole sentence (pre-chunking recordings).
    text: r.chunk_text ?? r.text,
    chunkIndex: r.chunk_index,
    createdAt: r.created_at,
    dotCount: r.dot_count,
    penMac: r.pen_mac,
  }));
}

/**
 * Append one entry and wait until the archiver has consumed it. Without this,
 * a tight append loop queues every recording's JSON string in memory at once.
 */
function appendEntry(
  archive: archiver.Archiver,
  data: string,
  name: string,
): Promise<void> {
  return new Promise((resolve) => {
    const onEntry = (entry: { name?: string }) => {
      if (entry.name === name) {
        archive.off("entry", onEntry);
        resolve();
      }
    };
    archive.on("entry", onEntry);
    archive.append(data, { name });
  });
}

/**
 * Streams a zip of all (non-rejected) recordings. Each rec-*.json is
 * byte-compatible with the training pipeline's RecordingSegment shape
 * (handwriting_dataset.py reads dots/dotType/x/y; extra metadata keys are
 * additive and safe).
 */
export async function streamExportZip(
  res: Response,
  filters: ExportFilters,
): Promise<void> {
  const stamp = new Date().toISOString().slice(0, 10);
  res.setHeader("Content-Type", "application/zip");
  res.setHeader(
    "Content-Disposition",
    `attachment; filename="handwriting-export-${stamp}.zip"`,
  );
  const archive = archiver("zip", { zlib: { level: 6 } });
  archive.pipe(res);

  const manifest = buildManifest(filters);
  await appendEntry(archive, JSON.stringify(manifest, null, 1), "manifest.json");

  const sentenceIds = [...new Set(manifest.map((m) => m.sentenceId))];
  const sentenceStmt = db.prepare("SELECT id, text, source FROM sentences WHERE id = ?");
  const sentencesJsonl = sentenceIds
    .map((id) => JSON.stringify(sentenceStmt.get(id)))
    .join("\n");
  await appendEntry(archive, sentencesJsonl, "sentences.jsonl");

  const exportedAt = new Date().toISOString();
  for (const r of recordingRowsWithDots(filters)) {
    const segment = {
      startTime: r.start_time,
      endTime: r.end_time,
      dotCount: r.dot_count,
      dots: JSON.parse(r.dots_json),
      metadata: {
        exportedAt,
        penMac: r.pen_mac ?? undefined,
        text: r.chunk_text ?? r.text,
        sentenceText: r.text,
        chunkIndex: r.chunk_index,
        sentenceId: r.sentence_id,
        userId: r.user_id,
        recordingId: r.id,
      },
    };
    await appendEntry(
      archive,
      JSON.stringify(segment),
      `recordings/u${r.user_id}/s${r.sentence_id}/rec-${r.id}.json`,
    );
  }
  await archive.finalize();
}
