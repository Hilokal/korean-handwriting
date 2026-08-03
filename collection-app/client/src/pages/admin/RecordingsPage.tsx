import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";
import { RecordedDot } from "../../pen/types";
import { recordedDotsToSvgPath } from "../../components/strokeSvg";

interface RecordingRow {
  id: number;
  user_id: number;
  userName: string;
  sentence_id: number;
  text: string;
  dot_count: number;
  status: "ok" | "rejected";
  created_at: string;
}

function Thumbnail({ recordingId }: { recordingId: number }) {
  const [dots, setDots] = useState<RecordedDot[] | null>(null);
  useEffect(() => {
    api
      .get<{ dots: RecordedDot[] }>(`/api/admin/recordings/${recordingId}`)
      .then((r) => setDots(r.dots))
      .catch(() => setDots([]));
  }, [recordingId]);

  if (dots === null) return <div className="thumb thumb-loading" />;
  if (dots.length === 0) return <div className="thumb">no ink</div>;
  const svg = recordedDotsToSvgPath(dots);
  return (
    <svg className="thumb" viewBox={svg.viewBox} preserveAspectRatio="xMidYMid meet">
      <path
        d={svg.pathData}
        fill="none"
        stroke="#222"
        strokeWidth={svg.height / 60}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function RecordingsPage() {
  const [rows, setRows] = useState<RecordingRow[]>([]);
  const [total, setTotal] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    api
      .get<{ recordings: RecordingRow[]; total: number; pageSize: number }>(
        `/api/admin/recordings?page=${page}`,
      )
      .then((r) => {
        setRows(r.recordings);
        setTotal(r.total);
        setPageSize(r.pageSize);
      })
      .catch((e) => setError(e.message));
  }, [page]);

  useEffect(reload, [reload]);

  const setStatus = async (id: number, action: "reject" | "restore") => {
    await api.post(`/api/admin/recordings/${id}/${action}`);
    reload();
  };

  const pages = Math.ceil(total / pageSize);

  return (
    <div>
      <h1>Recordings ({total})</h1>
      {error && <p className="banner banner-error">{error}</p>}
      <div className="recordings-grid">
        {rows.map((r) => (
          <div key={r.id} className={`card recording-card${r.status === "rejected" ? " row-disabled" : ""}`}>
            <Thumbnail recordingId={r.id} />
            <p className="recording-text">{r.text}</p>
            <p className="recording-meta">
              #{r.id} · {r.userName} · {r.dot_count} dots · {r.created_at}
              {r.status === "rejected" && " · REJECTED"}
            </p>
            {r.status === "ok" ? (
              <button className="link-button" onClick={() => setStatus(r.id, "reject")}>
                reject
              </button>
            ) : (
              <button className="link-button" onClick={() => setStatus(r.id, "restore")}>
                restore
              </button>
            )}
          </div>
        ))}
      </div>
      {pages > 1 && (
        <div className="actions">
          <button className="btn btn-secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>
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
