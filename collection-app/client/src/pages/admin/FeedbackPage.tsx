import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";

// Review feedback submitted from the public demo site (hangul.ink).
// dots_json is the MODEL's delta-encoded output (dx, dy, penState) — not pen
// Ncode dots — so it gets its own tiny renderer here. Convention matches
// training/render.py: a point starts a new stroke when the PREVIOUS point was
// end-of-stroke (penState == 1).

interface FeedbackRow {
  id: number;
  rating: "good" | "bad";
  text: string;
  temperature: number;
  bias: number;
  seed: number;
  model_version: string;
  dots_json: string;
  comment: string | null;
  locale: string | null;
  status: "new" | "reviewed" | "promoted";
  created_at: string;
}

function DotsThumbnail({ dotsJson }: { dotsJson: string }) {
  let dots: { x: number; y: number; penState: number }[];
  try {
    const parsed: unknown = JSON.parse(dotsJson);
    if (!Array.isArray(parsed)) return <div className="thumb">bad data</div>;
    dots = parsed as { x: number; y: number; penState: number }[];
  } catch {
    return <div className="thumb">bad data</div>;
  }

  let x = 0,
    y = 0,
    minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  let d = "";
  let newStroke = true;
  for (const dot of dots) {
    x += dot.x;
    y += dot.y;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    d += `${newStroke ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)} `;
    newStroke = dot.penState >= 0.5;
  }
  const w = maxX - minX || 1;
  const h = maxY - minY || 1;

  return (
    <svg
      className="thumb"
      viewBox={`${(minX - 0.4).toFixed(2)} ${(minY - 0.4).toFixed(2)} ${(w + 0.8).toFixed(2)} ${(h + 0.8).toFixed(2)}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <path
        d={d}
        fill="none"
        stroke="#222"
        strokeWidth={Math.max(h * 0.07, w * 0.008)}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function FeedbackPage() {
  const [rows, setRows] = useState<FeedbackRow[]>([]);
  const [total, setTotal] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    const params = new URLSearchParams({ page: String(page) });
    if (statusFilter) params.set("status", statusFilter);
    api
      .get<{ feedback: FeedbackRow[]; total: number; pageSize: number }>(
        `/api/admin/feedback?${params}`,
      )
      .then((r) => {
        setRows(r.feedback);
        setTotal(r.total);
        setPageSize(r.pageSize);
      })
      .catch((e) => setError(e.message));
  }, [page, statusFilter]);

  useEffect(reload, [reload]);

  const act = async (id: number, action: "review" | "promote") => {
    await api.post(`/api/admin/feedback/${id}/${action}`);
    reload();
  };
  const remove = async (id: number) => {
    await api.del(`/api/admin/feedback/${id}`);
    reload();
  };

  const pages = Math.ceil(total / pageSize);

  return (
    <div>
      <h1>Demo feedback ({total})</h1>
      <div className="actions">
        <select
          value={statusFilter}
          onChange={(e) => {
            setPage(0);
            setStatusFilter(e.target.value);
          }}
        >
          <option value="">all statuses</option>
          <option value="new">new</option>
          <option value="reviewed">reviewed</option>
          <option value="promoted">promoted</option>
        </select>
      </div>
      {error && <p className="banner banner-error">{error}</p>}
      <div className="recordings-grid">
        {rows.map((r) => (
          <div
            key={r.id}
            className={`card recording-card${r.status !== "new" ? " row-disabled" : ""}`}
          >
            <DotsThumbnail dotsJson={r.dots_json} />
            <p className="recording-text">
              {r.rating === "bad" ? "👎" : "👍"} {r.text}
            </p>
            {r.comment && <p className="recording-text">“{r.comment}”</p>}
            <p className="recording-meta">
              #{r.id} · temp {r.temperature} · bias {r.bias} · seed {r.seed} ·
              model {r.model_version} · {r.locale ?? "?"} · {r.created_at} ·{" "}
              {r.status}
            </p>
            <div className="actions">
              {r.status === "new" && (
                <button className="link-button" onClick={() => act(r.id, "review")}>
                  mark reviewed
                </button>
              )}
              {r.rating === "bad" && r.status !== "promoted" && (
                <button className="link-button" onClick={() => act(r.id, "promote")}>
                  send to writers
                </button>
              )}
              <button className="link-button" onClick={() => remove(r.id)}>
                delete
              </button>
            </div>
          </div>
        ))}
      </div>
      {pages > 1 && (
        <div className="actions">
          <button
            className="btn btn-secondary"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </button>
          <span className="hint">
            page {page + 1} / {pages}
          </span>
          <button
            className="btn btn-secondary"
            disabled={page >= pages - 1}
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
