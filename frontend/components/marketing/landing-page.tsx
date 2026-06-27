import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  FunctionSquare,
  PenTool,
  Shapes,
} from "lucide-react";

import { SiteFooter } from "@/components/layout/site-footer";
import { SiteHeader } from "@/components/layout/site-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const workingNow = [
  {
    icon: PenTool,
    title: "Typed problem solving",
    copy: "Send a real request to the FastAPI solver and get a structured answer with steps.",
  },
  {
    icon: FunctionSquare,
    title: "Graphs that render",
    copy: "Quadratic/function prompts generate GeoGebra commands that render in the solver.",
  },
  {
    icon: Shapes,
    title: "Basic constructions",
    copy: "Common circle, triangle, midpoint, and perpendicular-bisector prompts produce construction data.",
  },
];

export function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main>
        <section className="border-b border-border/70 bg-grid-fade">
          <div className="mx-auto grid max-w-7xl gap-10 px-6 py-20 lg:grid-cols-[1fr_0.85fr] lg:px-8 lg:py-24">
            <div className="max-w-3xl">
              <Badge>IntoMath</Badge>
              <h1 className="mt-5 text-5xl font-semibold tracking-tight text-foreground sm:text-6xl">
                Math help that actually solves.
              </h1>
              <p className="mt-6 text-lg leading-8 text-muted-foreground">
                A focused workspace for entering a problem, reading the steps,
                and seeing a graph or construction when the backend can build
                one. No mock panels; just the solver.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-4">
                <Link href="/dashboard/solve">
                  <Button className="gap-2" size="lg">
                    Open solver
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Link href="#how-it-works">
                  <Button size="lg" variant="outline">
                    How it works
                  </Button>
                </Link>
              </div>
            </div>

            <Card className="border-border/70 bg-card/95 shadow-soft">
              <CardHeader>
                <Badge className="w-fit" variant="success">
                  Working path
                </Badge>
                <CardTitle className="mt-3 text-2xl">
                  Try a supported prompt
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
                {[
                  "Solve 2x + 5 = 17 and explain each step.",
                  "Draw the graph of y = x^2 - 4x + 3 and describe the vertex.",
                  "Construct triangle ABC and draw the perpendicular bisector of AB.",
                ].map((prompt) => (
                  <div
                    className="rounded-2xl border border-border/70 bg-background p-4 text-foreground/90"
                    key={prompt}
                  >
                    {prompt}
                  </div>
                ))}
                <p>
                  Local mode handles common deterministic cases. Connect an
                  OpenRouter API key for broader AI solving and OCR.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-6 py-16 lg:px-8" id="features">
          <div className="max-w-2xl">
            <Badge variant="secondary">What works now</Badge>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight">
              A smaller product surface, focused on solving
            </h2>
          </div>
          <div className="mt-8 grid gap-5 md:grid-cols-3">
            {workingNow.map(({ icon: Icon, title, copy }) => (
              <Card className="border-border/70" key={title}>
                <CardHeader>
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                    <Icon className="h-5 w-5" />
                  </div>
                  <CardTitle>{title}</CardTitle>
                </CardHeader>
                <CardContent className="text-sm leading-7 text-muted-foreground">
                  {copy}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section
          className="border-y border-border/70 bg-secondary/30 py-16"
          id="how-it-works"
        >
          <div className="mx-auto grid max-w-7xl gap-5 px-6 lg:grid-cols-3 lg:px-8">
            {[
              "Type a clear problem or choose an example.",
              "The backend classifies it and solves with either local deterministic logic or the configured model backend.",
              "If a graph or construction is recognized, the app renders the generated GeoGebra commands.",
            ].map((step, index) => (
              <div
                className="rounded-2xl border border-border/70 bg-card p-5"
                key={step}
              >
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                  <p className="text-sm font-medium text-primary">
                    Step {index + 1}
                  </p>
                </div>
                <p className="mt-4 text-sm leading-7 text-muted-foreground">
                  {step}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-4xl px-6 py-16 text-center lg:px-8">
          <h2 className="text-3xl font-semibold tracking-tight">
            Ready to solve something?
          </h2>
          <p className="mt-4 text-muted-foreground">
            Go straight to the workspace. The rest of the site now stays out of
            the way.
          </p>
          <Link className="mt-8 inline-flex" href="/dashboard/solve">
            <Button className="gap-2" size="lg">
              Open solver
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
