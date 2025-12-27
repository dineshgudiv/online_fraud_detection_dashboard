"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";

import { AuthRole, useAuth } from "@/app/_components/AuthProvider";

type RequireRoleProps = {
  roles: AuthRole[];
  children: React.ReactNode;
};

export default function RequireRole({ roles, children }: RequireRoleProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { loading, hasRole } = useAuth();

  useEffect(() => {
    if (loading) {
      return;
    }
    if (!hasRole(...roles)) {
      const target = `/forbidden?from=${encodeURIComponent(pathname ?? "/")}`;
      router.replace(target);
    }
  }, [hasRole, loading, pathname, roles, router]);

  if (loading) {
    return <div className="empty">Loading permissions...</div>;
  }

  if (!hasRole(...roles)) {
    return null;
  }

  return <>{children}</>;
}
