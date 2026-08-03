import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { api, ApiError, PublicUser } from "./api";
import { LangToggle, useI18n } from "./i18n";
import { LoginPage } from "./pages/LoginPage";
import { InvitePage } from "./pages/InvitePage";
import { RecordPage } from "./pages/RecordPage";
import { AdminLayout } from "./pages/admin/AdminLayout";
import { UsersPage } from "./pages/admin/UsersPage";
import { ProgressPage } from "./pages/admin/ProgressPage";
import { RecordingsPage } from "./pages/admin/RecordingsPage";
import { ReportsPage } from "./pages/admin/ReportsPage";

export default function App() {
  const [user, setUser] = useState<PublicUser | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { s } = useI18n();

  useEffect(() => {
    api
      .get<{ user: PublicUser }>("/api/me")
      .then((r) => setUser(r.user))
      .catch((e) => {
        if (!(e instanceof ApiError && e.status === 401)) console.error(e);
      })
      .finally(() => setLoading(false));
  }, []);

  const handleLogout = useCallback(async () => {
    await api.post("/api/logout");
    setUser(null);
    navigate("/login");
  }, [navigate]);

  if (loading) return <div className="page-center">{s.loading}</div>;

  return (
    <>
      {user && (
        <header className="topbar">
          <span className="topbar-title">{s.appTitle}</span>
          <span className="topbar-user">
            {user.name}
            {user.isAdmin && (
              <>
                {" · "}
                <a href="/admin">{s.admin}</a>
              </>
            )}
            {" · "}
            <button className="link-button" onClick={handleLogout}>
              {s.logOut}
            </button>
            {" · "}
            <LangToggle />
          </span>
        </header>
      )}
      <Routes>
        <Route
          path="/login"
          element={user ? <Navigate to="/record" /> : <LoginPage onLogin={setUser} />}
        />
        <Route path="/invite/:token" element={<InvitePage onAccept={setUser} />} />
        <Route
          path="/record"
          element={user ? <RecordPage /> : <Navigate to="/login" />}
        />
        <Route
          path="/admin"
          element={user?.isAdmin ? <AdminLayout /> : <Navigate to="/login" />}
        >
          <Route index element={<UsersPage />} />
          <Route path="progress" element={<ProgressPage />} />
          <Route path="recordings" element={<RecordingsPage />} />
          <Route path="reports" element={<ReportsPage />} />
        </Route>
        <Route path="*" element={<Navigate to={user ? "/record" : "/login"} />} />
      </Routes>
    </>
  );
}
