import { Router } from "express";
import { requireAdmin } from "../auth.js";
import { buildManifest, streamExportZip } from "../exporter.js";

export const exportRoutes = Router();
exportRoutes.use(requireAdmin);

function parseFilters(query: Record<string, unknown>) {
  return {
    since: typeof query.since === "string" && query.since ? query.since : undefined,
    userId: query.userId ? Number(query.userId) : undefined,
  };
}

exportRoutes.get("/", (req, res) => {
  void streamExportZip(res, parseFilters(req.query));
});

exportRoutes.get("/manifest", (req, res) => {
  res.json({ recordings: buildManifest(parseFilters(req.query)) });
});
