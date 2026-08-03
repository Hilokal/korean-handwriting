import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";

const dbPath =
  process.env.DATABASE_PATH ?? path.join(process.cwd(), "data", "collection.db");
fs.mkdirSync(path.dirname(dbPath), { recursive: true });

export const db = new Database(dbPath);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");
