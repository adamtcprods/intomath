import Link from "next/link";
import { Calculator, PenTool } from "lucide-react";

import { ThemeToggle } from "@/components/layout/theme-toggle";
import { Button } from "@/components/ui/button";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4 lg:px-8">
        <Link className="flex items-center gap-3" href="/">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <Calculator className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-semibold tracking-tight">IntoMath</p>
            <p className="text-xs text-muted-foreground">Clear math steps</p>
          </div>
        </Link>

        <nav className="hidden items-center gap-8 text-sm text-muted-foreground md:flex">
          <Link href="#features">Features</Link>
          <Link href="#how-it-works">How it works</Link>
          <Link href="/dashboard/solve">Solver</Link>
        </nav>

        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link href="/dashboard/solve">
            <Button className="gap-2" size="sm">
              <PenTool className="h-4 w-4" />
              Open solver
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
}
