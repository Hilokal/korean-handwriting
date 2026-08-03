import { Router } from "express";
import { db } from "../db.js";
import {
  createSession,
  destroySession,
  hashPassword,
  rateLimitLogin,
  requireAuth,
  verifyPassword,
} from "../auth.js";

export const authRoutes = Router();

function publicUser(u: { id: number; name: string; username: string; is_admin: number }) {
  return { id: u.id, name: u.name, username: u.username, isAdmin: Boolean(u.is_admin) };
}

authRoutes.post("/login", rateLimitLogin, (req, res) => {
  const { username, password } = req.body ?? {};
  if (typeof username !== "string" || typeof password !== "string") {
    res.status(400).json({ error: "username and password required" });
    return;
  }
  const user = db
    .prepare("SELECT * FROM users WHERE username = ? AND disabled = 0")
    .get(username.trim().toLowerCase()) as
    | { id: number; name: string; username: string; is_admin: number; password_hash: string | null }
    | undefined;
  if (!user?.password_hash || !verifyPassword(password, user.password_hash)) {
    res.status(401).json({ error: "invalid username or password" });
    return;
  }
  createSession(res, user.id);
  res.json({ user: publicUser(user) });
});

authRoutes.post("/logout", (req, res) => {
  destroySession(req, res);
  res.json({ ok: true });
});

authRoutes.get("/me", requireAuth, (req, res) => {
  const stats = db
    .prepare(
      `SELECT
        (SELECT COUNT(*) FROM recordings WHERE user_id = ? AND status = 'ok') AS recordings,
        (SELECT COUNT(*) FROM user_syllable_counts WHERE user_id = ? AND count > 0) AS syllablesCovered,
        (SELECT COUNT(DISTINCT syllable) FROM sentence_syllables) AS totalSyllables`,
    )
    .get(req.user!.id, req.user!.id);
  res.json({ user: publicUser({ ...req.user!, is_admin: req.user!.is_admin }), stats });
});

authRoutes.get("/invites/:token", (req, res) => {
  const invite = db
    .prepare(
      `SELECT i.token, i.used_at, i.expires_at > datetime('now') AS valid, u.name
       FROM invites i JOIN users u ON u.id = i.user_id WHERE i.token = ?`,
    )
    .get(req.params.token) as
    | { used_at: string | null; valid: number; name: string }
    | undefined;
  if (!invite || invite.used_at || !invite.valid) {
    res.status(404).json({ error: "invite not found or expired" });
    return;
  }
  res.json({ name: invite.name, valid: true });
});

authRoutes.post("/invites/:token/accept", (req, res) => {
  const { password } = req.body ?? {};
  if (typeof password !== "string" || password.length < 10) {
    res.status(400).json({ error: "password must be at least 10 characters" });
    return;
  }
  const invite = db
    .prepare(
      `SELECT token, user_id FROM invites
       WHERE token = ? AND used_at IS NULL AND expires_at > datetime('now')`,
    )
    .get(req.params.token) as { token: string; user_id: number } | undefined;
  if (!invite) {
    res.status(404).json({ error: "invite not found or expired" });
    return;
  }
  const accept = db.transaction(() => {
    db.prepare("UPDATE users SET password_hash = ? WHERE id = ?").run(
      hashPassword(password),
      invite.user_id,
    );
    db.prepare("UPDATE invites SET used_at = datetime('now') WHERE token = ?").run(
      invite.token,
    );
  });
  accept();
  createSession(res, invite.user_id);
  const user = db
    .prepare("SELECT id, name, username, is_admin FROM users WHERE id = ?")
    .get(invite.user_id) as { id: number; name: string; username: string; is_admin: number };
  res.json({ user: publicUser(user) });
});
