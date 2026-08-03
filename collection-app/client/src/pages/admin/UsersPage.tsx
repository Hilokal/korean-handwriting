import { FormEvent, useCallback, useEffect, useState } from "react";
import { api } from "../../api";

interface AdminUser {
  id: number;
  name: string;
  username: string;
  is_admin: number;
  disabled: number;
  accepted: number;
  recordings: number;
  syllablesCovered: number;
  lastRecordingAt: string | null;
}

export function UsersPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [totalSyllables, setTotalSyllables] = useState(1);
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [inviteUrl, setInviteUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    api
      .get<{ users: AdminUser[]; totalSyllables: number }>("/api/admin/users")
      .then((r) => {
        setUsers(r.users);
        setTotalSyllables(r.totalSyllables || 1);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(reload, [reload]);

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const result = await api.post<{ inviteUrl: string }>("/api/admin/users", {
        name,
        username,
      });
      setInviteUrl(result.inviteUrl);
      setName("");
      setUsername("");
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed");
    }
  };

  const handleReinvite = async (id: number) => {
    const result = await api.post<{ inviteUrl: string }>(`/api/admin/users/${id}/invite`);
    setInviteUrl(result.inviteUrl);
  };

  const handleDisable = async (id: number) => {
    await api.post(`/api/admin/users/${id}/disable`);
    reload();
  };

  return (
    <div>
      <h1>Workers</h1>
      {error && <p className="banner banner-error">{error}</p>}
      {inviteUrl && (
        <p className="banner banner-info">
          Invite link (copy and send it): <code>{inviteUrl}</code>
        </p>
      )}

      <form className="inline-form" onSubmit={handleCreate}>
        <input
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
        <input
          placeholder="Username (login id)"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <button className="btn btn-primary">Add worker</button>
      </form>

      <table className="users-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Username</th>
            <th>Status</th>
            <th>Recordings</th>
            <th>Coverage</th>
            <th>Last active</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.id} className={u.disabled ? "row-disabled" : undefined}>
              <td>
                {u.name}
                {u.is_admin ? " (admin)" : ""}
              </td>
              <td>{u.username}</td>
              <td>{u.disabled ? "disabled" : u.accepted ? "active" : "invited"}</td>
              <td>{u.recordings}</td>
              <td>
                {u.syllablesCovered}/{totalSyllables} (
                {Math.round((100 * u.syllablesCovered) / totalSyllables)}%)
              </td>
              <td>{u.lastRecordingAt ?? "—"}</td>
              <td>
                {!u.is_admin && (
                  <>
                    {!u.accepted && (
                      <button className="link-button" onClick={() => handleReinvite(u.id)}>
                        re-invite
                      </button>
                    )}
                    {!u.disabled && (
                      <button className="link-button" onClick={() => handleDisable(u.id)}>
                        disable
                      </button>
                    )}
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
