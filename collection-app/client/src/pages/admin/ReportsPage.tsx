import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";

interface Report {
  id: number;
  sentenceId: number;
  text: string;
  active: number;
  note: string | null;
  reportedAt: string;
  userName: string;
}

export function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    api
      .get<{ reports: Report[] }>("/api/admin/reports")
      .then((r) => setReports(r.reports))
      .catch((e) => setError(e.message));
  }, []);

  useEffect(reload, [reload]);

  const setActive = async (sentenceId: number, active: boolean) => {
    await api.post(`/api/admin/sentences/${sentenceId}/${active ? "activate" : "deactivate"}`);
    reload();
  };

  return (
    <div>
      <h1>Reported sentences</h1>
      {error && <p className="banner banner-error">{error}</p>}
      {reports.length === 0 && <p className="hint">No reports.</p>}
      <table className="users-table">
        <thead>
          <tr>
            <th>Sentence</th>
            <th>Note</th>
            <th>By</th>
            <th>When</th>
            <th>Status</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {reports.map((r) => (
            <tr key={r.id}>
              <td>{r.text}</td>
              <td>{r.note ?? "—"}</td>
              <td>{r.userName}</td>
              <td>{r.reportedAt}</td>
              <td>{r.active ? "active" : "deactivated"}</td>
              <td>
                {r.active ? (
                  <button className="link-button" onClick={() => setActive(r.sentenceId, false)}>
                    deactivate
                  </button>
                ) : (
                  <button className="link-button" onClick={() => setActive(r.sentenceId, true)}>
                    reactivate
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
