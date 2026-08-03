import { FormEvent, useState } from "react";
import { api, PublicUser } from "../api";
import { LangToggle, useI18n } from "../i18n";

export function LoginPage({ onLogin }: { onLogin: (u: PublicUser) => void }) {
  const { s } = useI18n();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { user } = await api.post<{ user: PublicUser }>("/api/login", {
        username,
        password,
      });
      onLogin(user);
    } catch {
      setError(s.loginFailed);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-center">
      <form className="card form" onSubmit={handleSubmit}>
        <div className="form-header">
          <h1>{s.appTitle}</h1>
          <LangToggle />
        </div>
        <label>
          {s.username}
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoCapitalize="none"
            required
          />
        </label>
        <label>
          {s.password}
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>
        {error && <p className="error">{error}</p>}
        <button className="btn btn-primary" disabled={busy}>
          {busy ? s.loggingIn : s.logIn}
        </button>
      </form>
    </div>
  );
}
