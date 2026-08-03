import { FormEvent, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, PublicUser } from "../api";
import { LangToggle, useI18n } from "../i18n";

export function InvitePage({ onAccept }: { onAccept: (u: PublicUser) => void }) {
  const { s } = useI18n();
  const { token } = useParams<{ token: string }>();
  const navigate = useNavigate();
  const [name, setName] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .get<{ name: string }>(`/api/invites/${token}`)
      .then((r) => setName(r.name))
      .catch(() => setInvalid(true));
  }, [token]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (password !== confirm) {
      setError(s.passwordMismatch);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { user } = await api.post<{ user: PublicUser }>(
        `/api/invites/${token}/accept`,
        { password },
      );
      onAccept(user);
      navigate("/record");
    } catch {
      setError(s.errorGeneric);
    } finally {
      setBusy(false);
    }
  };

  if (invalid) {
    return (
      <div className="page-center">
        <div className="card">
          <h1>{s.inviteNotFound}</h1>
          <p>{s.inviteNotFoundBody}</p>
        </div>
      </div>
    );
  }
  if (name === null) return <div className="page-center">{s.loading}</div>;

  return (
    <div className="page-center">
      <form className="card form" onSubmit={handleSubmit}>
        <div className="form-header">
          <h1>{s.welcome(name)}</h1>
          <LangToggle />
        </div>
        <p>{s.choosePassword}</p>
        <label>
          {s.passwordMin}
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            minLength={10}
            required
          />
        </label>
        <label>
          {s.confirmPassword}
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            required
          />
        </label>
        {error && <p className="error">{error}</p>}
        <button className="btn btn-primary" disabled={busy}>
          {busy ? s.creatingAccount : s.createAccount}
        </button>
      </form>
    </div>
  );
}
