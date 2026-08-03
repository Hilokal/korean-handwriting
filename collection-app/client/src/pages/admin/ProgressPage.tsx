import { useEffect, useState } from "react";
import { api } from "../../api";

interface ProgressUser {
  id: number;
  name: string;
  recordings: number;
  syllablesCovered: number;
}

interface Bucket {
  userId: number;
  bucket: string;
  syllables: number;
}

const BUCKET_ORDER = ["1", "2", "3", "4", "5+"];

export function ProgressPage() {
  const [users, setUsers] = useState<ProgressUser[]>([]);
  const [buckets, setBuckets] = useState<Bucket[]>([]);
  const [totalSyllables, setTotalSyllables] = useState(1);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<{ users: ProgressUser[]; buckets: Bucket[]; totalSyllables: number }>(
        "/api/admin/progress",
      )
      .then((r) => {
        setUsers(r.users);
        setBuckets(r.buckets);
        setTotalSyllables(r.totalSyllables || 1);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="banner banner-error">{error}</p>;

  return (
    <div>
      <h1>Coverage progress</h1>
      <p className="hint">
        For each worker: how many of the {totalSyllables} syllables they have
        recorded, and how many times. The goal is every syllable covered a few
        times per worker (per handwriting style).
      </p>
      {users.map((u) => {
        const userBuckets = new Map(
          buckets.filter((b) => b.userId === u.id).map((b) => [b.bucket, b.syllables]),
        );
        const covered = u.syllablesCovered;
        return (
          <div key={u.id} className="card progress-card">
            <div className="progress-header">
              <strong>{u.name}</strong>
              <span>
                {u.recordings} recordings · {covered}/{totalSyllables} syllables (
                {Math.round((100 * covered) / totalSyllables)}%)
              </span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${(100 * covered) / totalSyllables}%` }}
              />
            </div>
            <div className="bucket-row">
              <span className="bucket" title="not yet recorded">
                0×: {totalSyllables - covered}
              </span>
              {BUCKET_ORDER.map((b) => (
                <span key={b} className="bucket">
                  {b}×: {userBuckets.get(b) ?? 0}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
