import type { ReactNode } from "react";
import Link from "next/link";
import { Calculator, Home, Sparkles } from "lucide-react";

import { ThemeToggle } from "@/components/layout/theme-toggle";
import { cn } from "@/lib/utils";

const navigation = [
  { href: "/dashboard/solve", label: "Solver", icon: Sparkles },
  { href: "/", label: "Home", icon: Home },
];

export function DashboardShell({
  children,
  currentPath,
}: {
  children: ReactNode;
  currentPath: string;
}) {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[100rem] flex-wrap items-center justify-between gap-3 px-4 py-3 lg:px-6">
          <Link className="flex items-center gap-3" href="/dashboard/solve">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Calculator className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">IntoMath</p>
              <p className="text-xs text-muted-foreground">Clear math steps</p>
            </div>
          </Link>

          <div className="flex items-center gap-2">
            <nav className="flex rounded-full border border-border/70 bg-card p-1 text-sm shadow-sm">
              {navigation.map(({ href, label, icon: Icon }) => {
                const active = currentPath === href;
                return (
                  <Link
                    key={href}
                    className={cn(
                      "flex items-center gap-2 rounded-full px-3 py-2 transition-colors",
                      active
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                    )}
                    href={href}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{label}</span>
                  </Link>
                );
              })}
            </nav>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main>{children}</main>
    </div>
  );
}
