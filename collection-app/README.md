# Handwriting Collection

Web tool for collecting Korean handwriting stroke data from remote workers
using Neo Smartpens on Ncode paper. Express + SQLite backend, React frontend,
one deployable unit.

## Requirements for workers

- Neo Smartpen (shipped to them) + Ncode notebook
- Google Chrome or Microsoft Edge on desktop, or Chrome on Android
  (Web Bluetooth; Safari/iOS not supported)

## Local development

```bash
npm install
npm run dev          # server on :8080, Vite client on :3000 (proxies /api)
```

The server seeds sentences from `../data/reference-texts/all.jsonl` on first
boot (override with `SENTENCES_PATH`). Create the first admin by setting env
vars before first boot:

```bash
ADMIN_EMAIL=you@example.com ADMIN_PASSWORD=... npm run dev:server
```

## How it works

- Admin creates a worker on `/admin` → gets an invite URL to send manually.
- Worker opens the invite, sets a password, connects their pen, and writes the
  sentence shown. Recording starts automatically at the first pen stroke;
  after "Done writing" they review an SVG render of their strokes and submit.
- Sentences are assigned per worker by syllable coverage: each worker is
  steered toward the syllables *they personally* have recorded least, so every
  handwriting style covers the syllable space (see `server/assignment.ts`).
- Long sentences are shown in line-sized chunks ("부분 1/2") so nothing wraps
  onto a second physical line — each chunk is its own recording labeled with
  exactly what was written. Chunk size is `CHUNK_MAX_CHARS` (default 30,
  including spaces); chunks are frozen per assignment, so changing it only
  affects new assignments.
- Recordings are stored in SQLite (`recordings.dots_json`) in the exact
  `RecordingSegment` shape the training pipeline consumes.

## Deploy (Fly.io)

```bash
npm run seed:copy                 # bundle sentences into the image
fly launch --no-deploy            # once; creates the app (keep the existing fly.toml)
fly volumes create data --size 10 --region nrt   # once
fly secrets set ADMIN_EMAIL=... ADMIN_PASSWORD=...
fly deploy
```

Optional continuous backup with Litestream (recommended once real data flows):

```bash
fly secrets set LITESTREAM_REPLICA_URL=s3://bucket/collection.db \
  LITESTREAM_ACCESS_KEY_ID=... LITESTREAM_SECRET_ACCESS_KEY=...
```

Keep exactly one machine (`min_machines_running = 1`, no autoscaling):
SQLite has a single writer.
