import type { ReactNode } from "react";

import { SiteHeader } from "@/components/layout/site-header";

export function DashboardShell({
  children,
  currentPath,
}: {
  children: ReactNode;
  currentPath: string;
}) {
  return (
    <div className="min-h-screen bg-background">
      <SiteHeader currentPath={currentPath} />

      <main>{children}</main>
    </div>
  );
}
