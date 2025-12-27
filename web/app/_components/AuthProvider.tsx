"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { apiFetch } from "@/lib/api";

export type AuthRole = "ADMIN" | "ANALYST" | "READONLY";

export type AuthUser = {
  id: string;
  email: string;
  role: AuthRole;
};

type AuthContextValue = {
  user: AuthUser | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  hasRole: (...roles: AuthRole[]) => boolean;
  can: (permission: string) => boolean;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const normalizeRole = (role: string): AuthRole => {
  if (role === "VIEWER") {
    return "READONLY";
  }
  if (role === "ADMIN" || role === "ANALYST" || role === "READONLY") {
    return role;
  }
  return "READONLY";
};

const PERMISSIONS: Record<string, AuthRole[]> = {
  "model:retrain": ["ADMIN"],
  "security:cert:download": ["ADMIN", "ANALYST"],
  "security:settings": ["ADMIN"]
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiFetch<{ id: string; email: string; role: string }>("/auth/me");
      setUser({ id: resp.id, email: resp.email, role: normalizeRole(resp.role) });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load user");
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const hasRole = useCallback(
    (...roles: AuthRole[]) => {
      if (!user) {
        return false;
      }
      return roles.includes(user.role);
    },
    [user]
  );

  const can = useCallback(
    (permission: string) => {
      if (!user) {
        return false;
      }
      const allowed = PERMISSIONS[permission];
      if (!allowed) {
        return false;
      }
      return allowed.includes(user.role);
    },
    [user]
  );

  const value = useMemo(
    () => ({ user, loading, error, refresh, hasRole, can }),
    [user, loading, error, refresh, hasRole, can]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
