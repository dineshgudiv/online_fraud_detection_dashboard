"use client";

import { useCallback, useEffect, useState } from "react";

import RequireRole from "@/app/_components/RequireRole";
import PageHeader from "@/app/_components/PageHeader";
import DataTable from "@/app/_components/DataTable";

type UserRecord = {
  id: string;
  name: string;
  email: string;
  role: "Admin" | "Analyst" | "Read-only";
  status: "Active" | "Disabled";
  last_seen: string;
};

const roleOptions: UserRecord["role"][] = ["Admin", "Analyst", "Read-only"];

export default function UsersSettingsPage() {
  const [users, setUsers] = useState<UserRecord[]>([
    {
      id: "user-1",
      name: "A. Perez",
      email: "analyst@demo",
      role: "Analyst",
      status: "Active",
      last_seen: "5 minutes ago"
    },
    {
      id: "user-2",
      name: "J. Park",
      email: "viewer@demo",
      role: "Read-only",
      status: "Disabled",
      last_seen: "2 days ago"
    }
  ]);
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteName, setInviteName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<UserRecord["role"]>("Analyst");

  const closeInvite = useCallback(() => {
    setInviteOpen(false);
    setInviteName("");
    setInviteEmail("");
    setInviteRole("Analyst");
  }, []);

  useEffect(() => {
    if (!inviteOpen) {
      return;
    }
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeInvite();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [inviteOpen, closeInvite]);

  const updateRole = (id: string, role: UserRecord["role"]) => {
    setUsers((prev) =>
      prev.map((user) => (user.id === id ? { ...user, role } : user))
    );
  };

  return (
    <RequireRole roles={["ADMIN"]}>
      <section className="section">
        <PageHeader
          title="User settings"
          subtitle="Manage access roles, status, and onboarding."
          actions={
            <div className="page-actions">
              <button className="button" type="button" onClick={() => setInviteOpen(true)}>
                Invite user
              </button>
            </div>
          }
        />

        <DataTable
          title="Users"
          subtitle="Roles and access status."
          columns={["Name", "Email", "Role", "Status", "Last seen"]}
          rows={users.map((user) => [
            user.name,
            user.email,
            <select
              key={`role-${user.id}`}
              value={user.role}
              onChange={(event) =>
                updateRole(user.id, event.target.value as UserRecord["role"])
              }
            >
              {roleOptions.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>,
            user.status,
            user.last_seen
          ])}
          emptyText="No users found."
          actions={
            <button className="button secondary" type="button">
              Export list
            </button>
          }
          footer={
            <div className="table-pagination">
              <span>Showing {users.length} users</span>
            </div>
          }
        />

        {inviteOpen && (
          <div
            className="modal-backdrop"
            role="dialog"
            aria-modal="true"
            onClick={closeInvite}
          >
            <div className="modal" onClick={(event) => event.stopPropagation()}>
              <div className="modal-header">
                <h3>Invite user</h3>
                <p className="muted">
                  Create a new user invite. This is a placeholder only.
                </p>
              </div>
              <div className="form">
                <input
                  placeholder="Name"
                  value={inviteName}
                  onChange={(event) => setInviteName(event.target.value)}
                />
                <input
                  placeholder="Email"
                  value={inviteEmail}
                  onChange={(event) => setInviteEmail(event.target.value)}
                />
                <select
                  value={inviteRole}
                  onChange={(event) =>
                    setInviteRole(event.target.value as UserRecord["role"])
                  }
                >
                  {roleOptions.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </div>
              <div className="modal-actions">
                <button
                  className="button secondary"
                  type="button"
                  onClick={closeInvite}
                >
                  Cancel
                </button>
                <button
                  className="button"
                  type="button"
                  onClick={() => {
                    // TODO: GET/POST /api/settings/users
                    closeInvite();
                  }}
                >
                  Send invite
                </button>
              </div>
            </div>
          </div>
        )}
      </section>
    </RequireRole>
  );
}
