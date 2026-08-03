import { db } from "./db.js";

const migrations: string[] = [
  // 1: initial schema
  `
  CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    is_admin INTEGER NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE invites (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    used_at TEXT
  );

  CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
  );
  CREATE INDEX idx_sessions_user ON sessions(user_id);

  CREATE TABLE sentences (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL UNIQUE,
    source TEXT,
    active INTEGER NOT NULL DEFAULT 1
  );

  CREATE TABLE sentence_syllables (
    syllable TEXT NOT NULL,
    sentence_id INTEGER NOT NULL REFERENCES sentences(id),
    occurrences INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (syllable, sentence_id)
  ) WITHOUT ROWID;
  CREATE INDEX idx_ss_sentence ON sentence_syllables(sentence_id);

  CREATE TABLE assignments (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    sentence_id INTEGER NOT NULL REFERENCES sentences(id),
    status TEXT NOT NULL DEFAULT 'active'
      CHECK (status IN ('active','completed','skipped','reported')),
    problem_note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
  );
  CREATE UNIQUE INDEX idx_one_active_assignment
    ON assignments(user_id) WHERE status = 'active';
  CREATE INDEX idx_assignments_user ON assignments(user_id, status);

  CREATE TABLE recordings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    sentence_id INTEGER NOT NULL REFERENCES sentences(id),
    assignment_id INTEGER REFERENCES assignments(id),
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    dot_count INTEGER NOT NULL,
    pen_mac TEXT,
    page_section INTEGER, page_owner INTEGER, page_book INTEGER, page_page INTEGER,
    dots_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','rejected')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX idx_recordings_user ON recordings(user_id);
  CREATE INDEX idx_recordings_sentence ON recordings(sentence_id);

  CREATE TABLE user_syllable_counts (
    user_id INTEGER NOT NULL REFERENCES users(id),
    syllable TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, syllable)
  ) WITHOUT ROWID;
  `,
  // 2: prompt chunking — long sentences are shown to workers in line-sized
  // pieces; each recording captures one chunk.
  `
  ALTER TABLE assignments ADD COLUMN chunks TEXT;
  ALTER TABLE recordings ADD COLUMN chunk_index INTEGER NOT NULL DEFAULT 0;
  ALTER TABLE recordings ADD COLUMN chunk_text TEXT;
  `,
  // 3: logins are plain usernames handed out by the admin — the app never
  // sends email, so requiring an email address was pointless friction.
  // Existing accounts keep their address as their username.
  `
  ALTER TABLE users RENAME COLUMN email TO username;
  `,
];

export function migrate(): void {
  db.exec(
    "CREATE TABLE IF NOT EXISTS migrations (id INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))",
  );
  const applied = new Set(
    db
      .prepare("SELECT id FROM migrations")
      .all()
      .map((r) => (r as { id: number }).id),
  );
  const run = db.transaction((id: number, sql: string) => {
    db.exec(sql);
    db.prepare("INSERT INTO migrations (id) VALUES (?)").run(id);
  });
  migrations.forEach((sql, i) => {
    const id = i + 1;
    if (!applied.has(id)) {
      run(id, sql);
      console.log(`applied migration ${id}`);
    }
  });
}
