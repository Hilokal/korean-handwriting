import { NavLink, Outlet } from "react-router-dom";

export function AdminLayout() {
  return (
    <div className="admin-page">
      <nav className="admin-nav">
        <NavLink to="/admin" end>
          Workers
        </NavLink>
        <NavLink to="/admin/progress">Progress</NavLink>
        <NavLink to="/admin/recordings">Recordings</NavLink>
        <NavLink to="/admin/reports">Reports</NavLink>
        <a href="/api/admin/export" className="export-link">
          Export zip
        </a>
      </nav>
      <Outlet />
    </div>
  );
}
