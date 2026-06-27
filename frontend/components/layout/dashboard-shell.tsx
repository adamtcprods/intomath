import type { ReactNode } from "react";
import Link from "next/link";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { Home, LayoutPanelLeft, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

const navigation = [
  { href: "/dashboard/solve", label: "Solver", icon: Sparkles },
  { href: "/", label: "Home", icon: Home },
];

const pageMeta: Record<
  string,
  { eyebrow: string; title: string; description: string }
> = {
  "/dashboard/solve": {
    eyebrow: "Solver",
    title: "Solve one problem at a time",
    description:
      "Enter a problem, get a structured answer, and inspect a graph or construction when supported.",
  },
};

export function DashboardShell({
  children,
  currentPath,
}: {
  children: ReactNode;
  currentPath: string;
}) {
  const isSolvePage = currentPath === "/dashboard/solve";
  const activePage = pageMeta[currentPath] ?? {
    eyebrow: "Solver",
    title: "Open the solver",
    description:
      "IntoMath is focused on solving typed math problems with clear steps and useful visuals.",
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="grid min-h-screen lg:grid-cols-[280px_1fr]">
        <aside className="border-r border-border/70 bg-card">
          <div className="flex h-full flex-col px-5 py-6">
            <div className="mb-8 flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                <LayoutPanelLeft className="h-5 w-5" />
              </div>
              <div>
                <p className="font-semibold tracking-tight">IntoMath Solver</p>
                <p className="text-sm text-muted-foreground">
                  Focused problem solving
                </p>
              </div>
            </div>

            <nav className="space-y-1">
              {navigation.map(({ href, label, icon: Icon }) => {
                const active = currentPath === href;
                return (
                  <Link
                    key={href}
                    className={cn(
                      "flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition-colors",
                      active
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                    )}
                    href={href}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </aside>
        <main className="min-w-0">
          <div className="flex min-h-screen flex-col">
            {isSolvePage ? null : (
              <header className="sticky top-0 z-20 border-b border-border/70 bg-background/90 px-4 py-4 backdrop-blur-xl lg:px-8">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-[0.18em] text-primary">
                      {activePage.eyebrow}
                    </p>
                    <h1 className="mt-2 text-2xl font-semibold tracking-tight text-foreground">
                      {activePage.title}
                    </h1>
                    <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                      {activePage.description}
                    </p>
                  </div>
                  <ThemeToggle />
                </div>
              </header>
            )}
            {isSolvePage ? (
              <div className="sticky top-0 z-20 border-b border-border/70 bg-background/90 px-4 py-3 backdrop-blur-xl lg:px-8">
                <div className="flex justify-end">
                  <ThemeToggle />
                </div>
              </div>
            ) : null}
            <div className="flex-1">{children}</div>
          </div>
        </main>
      </div>
    </div>
  );
}
