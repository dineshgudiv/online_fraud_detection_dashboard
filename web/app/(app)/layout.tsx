import AppShell from "@/components/AppShell";
import { AuthProvider } from "@/app/_components/AuthProvider";

export default function AppLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthProvider>
      <AppShell>{children}</AppShell>
    </AuthProvider>
  );
}

