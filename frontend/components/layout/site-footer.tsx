import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="border-t border-border/70 bg-background">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-10 text-sm text-muted-foreground lg:grid-cols-[1.4fr_1fr_1fr] lg:px-8">
        <div className="space-y-3">
          <p className="text-base font-semibold text-foreground">IntoMath</p>
          <p>
            A focused math solver for clear answers, guided steps, and helpful
            visuals.
          </p>
        </div>
        <div>
          <p className="mb-3 font-medium text-foreground">Product</p>
          <div className="space-y-2">
            <Link href="#features">Features</Link>
            <Link href="/dashboard/solve">Solver</Link>
          </div>
        </div>
        <div>
          <p className="mb-3 font-medium text-foreground">Resources</p>
          <div className="space-y-2">
            <Link href="#how-it-works">How it works</Link>
            <Link href="https://www.geogebra.org" target="_blank">
              GeoGebra
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
