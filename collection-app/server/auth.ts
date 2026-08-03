import crypto from "node:crypto";
import type { NextFunction, Request, Response } from "express";
import { db } from "./db.js";

export interface User {
  id: number;
  name: string;
  username: string;
  is_admin: number;
  disabled: number;
}

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      user?: User;
    }
  }
}

const SESSION_COOKIE = "sid";
const SESSION_DAYS = 30;
const INVITE_DAYS = 14;
const SCRYPT_N = 16384,
  SCRYPT_R = 8,
  SCRYPT_P = 1,
  KEY_LEN = 64;

// --- Passwords (scrypt: built-in, no native deps) ---

export function hashPassword(password: string): string {
  const salt = crypto.randomBytes(16).toString("hex");
  const hash = crypto
    .scryptSync(password, salt, KEY_LEN, { N: SCRYPT_N, r: SCRYPT_R, p: SCRYPT_P })
    .toString("hex");
  return `scrypt$${SCRYPT_N}$${SCRYPT_R}$${SCRYPT_P}$${salt}$${hash}`;
}

export function verifyPassword(password: string, stored: string): boolean {
  const [scheme, n, r, p, salt, hash] = stored.split("$");
  if (scheme !== "scrypt") return false;
  const computed = crypto.scryptSync(password, salt, KEY_LEN, {
    N: Number(n),
    r: Number(r),
    p: Number(p),
  });
  return crypto.timingSafeEqual(computed, Buffer.from(hash, "hex"));
}

// --- Sessions ---

export function createSession(res: Response, userId: number): void {
  const id = crypto.randomBytes(32).toString("hex");
  db.prepare(
    `INSERT INTO sessions (id, user_id, expires_at)
     VALUES (?, ?, datetime('now', '+${SESSION_DAYS} days'))`,
  ).run(id, userId);
  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  res.setHeader(
    "Set-Cookie",
    `${SESSION_COOKIE}=${id}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${SESSION_DAYS * 86400}${secure}`,
  );
}

export function destroySession(req: Request, res: Response): void {
  const id = getCookie(req, SESSION_COOKIE);
  if (id) db.prepare("DELETE FROM sessions WHERE id = ?").run(id);
  res.setHeader(
    "Set-Cookie",
    `${SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0`,
  );
}

function getCookie(req: Request, name: string): string | undefined {
  const header = req.headers.cookie;
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return undefined;
}

// --- Middleware ---

export function requireAuth(req: Request, res: Response, next: NextFunction): void {
  const id = getCookie(req, SESSION_COOKIE);
  if (!id) {
    res.status(401).json({ error: "not logged in" });
    return;
  }
  const row = db
    .prepare(
      `SELECT u.id, u.name, u.username, u.is_admin, u.disabled, s.expires_at
       FROM sessions s JOIN users u ON u.id = s.user_id
       WHERE s.id = ? AND s.expires_at > datetime('now')`,
    )
    .get(id) as (User & { expires_at: string }) | undefined;
  if (!row || row.disabled) {
    res.status(401).json({ error: "not logged in" });
    return;
  }
  // Sliding expiry: renew when less than half the window remains.
  db.prepare(
    `UPDATE sessions SET expires_at = datetime('now', '+${SESSION_DAYS} days')
     WHERE id = ? AND expires_at < datetime('now', '+${SESSION_DAYS / 2} days')`,
  ).run(id);
  req.user = { id: row.id, name: row.name, username: row.username, is_admin: row.is_admin, disabled: row.disabled };
  next();
}

export function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  requireAuth(req, res, () => {
    if (!req.user?.is_admin) {
      res.status(403).json({ error: "admin only" });
      return;
    }
    next();
  });
}

/** CSRF guard for cookie-authed JSON API: require a JSON content type on writes. */
export function requireJson(req: Request, res: Response, next: NextFunction): void {
  if (["POST", "PUT", "PATCH", "DELETE"].includes(req.method)) {
    if (!req.is("application/json")) {
      res.status(415).json({ error: "expected application/json" });
      return;
    }
  }
  next();
}

// --- Login rate limiting (in-memory; single process by design) ---

const attempts = new Map<string, { count: number; resetAt: number }>();
const MAX_ATTEMPTS = 10;
const WINDOW_MS = 60_000;

export function rateLimitLogin(req: Request, res: Response, next: NextFunction): void {
  const ip = req.ip ?? "unknown";
  const now = Date.now();
  const entry = attempts.get(ip);
  if (!entry || entry.resetAt < now) {
    attempts.set(ip, { count: 1, resetAt: now + WINDOW_MS });
  } else if (++entry.count > MAX_ATTEMPTS) {
    res.status(429).json({ error: "too many attempts, try again in a minute" });
    return;
  }
  next();
}

// --- Invites ---

export function createInvite(userId: number): string {
  const token = crypto.randomBytes(32).toString("hex");
  db.prepare(
    `INSERT INTO invites (token, user_id, expires_at)
     VALUES (?, ?, datetime('now', '+${INVITE_DAYS} days'))`,
  ).run(token, userId);
  return token;
}

/** Create the first admin from env vars if no admin exists yet. */
export function seedAdmin(): void {
  const hasAdmin = db
    .prepare("SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1")
    .get();
  if (hasAdmin) return;
  const username = process.env.ADMIN_USERNAME ?? process.env.ADMIN_EMAIL;
  const password = process.env.ADMIN_PASSWORD;
  if (!username || !password) {
    console.warn("no admin user exists and ADMIN_USERNAME/ADMIN_PASSWORD not set");
    return;
  }
  db.prepare(
    "INSERT INTO users (name, username, password_hash, is_admin) VALUES ('Admin', ?, ?, 1)",
  ).run(username, hashPassword(password));
  console.log(`seeded admin user ${username}`);
}
