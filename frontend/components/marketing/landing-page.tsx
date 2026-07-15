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

const highlights = [
  {
    icon: PenTool,
    title: "Step-by-step answers",
    copy: "Work through equations and word problems with concise explanations you can follow.",
  },
  {
    icon: FunctionSquare,
    title: "Visuals when they help",
    copy: "Graphs and constructions appear next to the solution when a picture makes the idea clearer.",
  },
  {
    icon: Shapes,
    title: "One focused workspace",
    copy: "Type a question, attach a photo, and review the answer without extra panels or distractions.",
  },
];

const examplePrompts = [
  "Solve 2x + 5 = 17 and explain each step.",
  "Draw the graph of y = x^2 - 4x + 3 and describe the vertex.",
  "Construct triangle ABC and draw the perpendicular bisector of AB.",
];

export function LandingPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />
      <main className="flex-1">
        <section className="border-b border-border/70 bg-grid-fade">
          <div className="mx-auto grid max-w-7xl gap-10 px-6 py-20 lg:grid-cols-[1fr_0.85fr] lg:px-8 lg:py-24">
            <div className="max-w-3xl">
              <h1 className="text-5xl font-semibold tracking-tight text-foreground sm:text-6xl">
                Solve math with clear steps.
              </h1>
              <p className="mt-6 text-lg leading-8 text-muted-foreground">
                A clean workspace for typing a problem, following the reasoning,
                and seeing a graph or construction when it helps the
                explanation.
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
                  Try it now
                </Badge>
                <CardTitle className="mt-3 text-2xl">
                  Start with a clear question
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
                {examplePrompts.map((prompt) => (
                  <Link
                    className="group flex items-center justify-between gap-4 rounded-2xl border border-border/70 bg-background p-4 text-foreground/90 transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    href={{
                      pathname: "/dashboard/solve",
                      query: { prompt },
                    }}
                    key={prompt}
                  >
                    <span>{prompt}</span>
                    <ArrowRight className="h-4 w-4 shrink-0 transition-transform group-hover:translate-x-1" />
                  </Link>
                ))}
                <p>
                  Choose an example, type your own problem, or attach an image
                  in the solver.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-6 py-16 lg:px-8" id="features">
          <div className="max-w-2xl">
            <h2 className="text-3xl font-semibold tracking-tight">
              Everything points back to the solution
            </h2>
          </div>
          <div className="mt-8 grid gap-5 md:grid-cols-3">
            {highlights.map(({ icon: Icon, title, copy }) => (
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
              "Enter a problem or upload a photo of the question.",
              "Read the final answer first, then move through the steps at your own pace.",
              "Use the visual panel when a graph or construction makes the solution easier to understand.",
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
            Open the solver, ask one question, and focus on the steps.
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
