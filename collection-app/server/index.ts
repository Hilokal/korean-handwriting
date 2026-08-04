import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { migrate } from "./migrate.js";
import { seedSentences } from "./seedSentences.js";
import { requireJson, seedAdmin } from "./auth.js";
import { authRoutes } from "./routes/auth.routes.js";
import { workRoutes } from "./routes/work.routes.js";
import { adminRoutes } from "./routes/admin.routes.js";
import { exportRoutes } from "./routes/export.routes.js";
import { feedbackAdminRoutes, feedbackRoutes } from "./routes/feedback.routes.js";

migrate();
seedSentences();
seedAdmin();

const app = express();
app.set("trust proxy", true); // Fly.io terminates TLS; trust X-Forwarded-*
app.use(express.json({ limit: "8mb" }));
app.use("/api", requireJson);

app.use("/api", authRoutes);
app.use("/api/work", workRoutes);
app.use("/api/feedback", feedbackRoutes); // public: demo-site feedback (CORS)
app.use("/api/admin/export", exportRoutes);
app.use("/api/admin/feedback", feedbackAdminRoutes);
app.use("/api/admin", adminRoutes);

app.get("/api/health", (_req, res) => {
  res.json({ ok: true });
});

// Serve the built client (production). In dev, Vite serves the client and
// proxies /api here.
const clientDist = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "client",
  "dist",
);
if (fs.existsSync(clientDist)) {
  app.use(express.static(clientDist));
  app.get("*", (_req, res) => {
    res.sendFile(path.join(clientDist, "index.html"));
  });
}

const port = Number(process.env.PORT ?? 8080);
app.listen(port, () => {
  console.log(`listening on :${port}`);
});
