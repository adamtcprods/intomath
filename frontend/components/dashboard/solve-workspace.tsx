"use client";

import { type ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BookOpenCheck,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ImagePlus,
  LoaderCircle,
  Sparkles,
} from "lucide-react";

import { MathText } from "@/components/math/math-text";
import { GeoGebraApplet } from "@/components/visualization/geogebra-applet";
import { useSolveProblem } from "@/features/solver/hooks/use-solve-problem";
import { useSolveWorkspaceStore } from "@/stores/solve-workspace";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn, formatConfidence, titleCase } from "@/lib/utils";

const examples = [
  "Solve 2x + 5 = 17 and explain each step.",
  "Draw the graph of y = x^2 - 4x + 3 and describe the vertex.",
  "Construct triangle ABC and draw the perpendicular bisector of AB.",
  "A circle has center O and radius 5. Plot point A at (3,4).",
];

export function SolveWorkspace() {
  const {
    input,
    imageBase64,
    imageMimeType,
    setInput,
    setImageBase64,
    setImageMimeType,
  } = useSolveWorkspaceStore();
  const solveMutation = useSolveProblem();
  const result = solveMutation.data;
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [activeQuestionIndex, setActiveQuestionIndex] = useState(0);

  const activePrompt = useMemo(() => input.trim(), [input]);
  const commands = result?.visualization.geogebra?.commands ?? [];
  const visualizationKey = useMemo(() => commands.join("\n"), [commands]);
  const hasVisualization = commands.length > 0;
  const questionParts = result?.parts ?? [];
  const hasQuestionSwitcher = questionParts.length > 1;
  const safeActiveQuestionIndex = questionParts.length
    ? Math.min(activeQuestionIndex, questionParts.length - 1)
    : 0;
  const activeQuestionPart = questionParts[safeActiveQuestionIndex] ?? null;
  const activeQuestionLabel = toQuestionLabel(safeActiveQuestionIndex);
  const activeAnswer = activeQuestionPart?.answer ?? result?.answer ?? null;
  const resultSteps = result?.steps ?? [];
  const steps = activeQuestionPart ? activeQuestionPart.steps : resultSteps;
  const safeActiveStepIndex = steps.length
    ? Math.min(activeStepIndex, steps.length - 1)
    : 0;
  const activeStep = steps[safeActiveStepIndex] ?? null;
  const activeStepCommonMistakes = activeStep?.common_mistakes ?? [];
  const activeStepSupportItems = activeStep
    ? [
        ...(activeStep.hints ?? []),
        ...(activeStep.alternative_approaches ?? []),
        ...(activeStep.exam_tip ? [activeStep.exam_tip] : []),
      ]
    : [];
  const activeStepHasBothSupportSections =
    activeStepCommonMistakes.length > 0 && activeStepSupportItems.length > 0;

  useEffect(() => {
    setActiveQuestionIndex(0);
    setActiveStepIndex(0);
  }, [result?.request_id]);

  useEffect(() => {
    setActiveStepIndex(0);
  }, [safeActiveQuestionIndex]);

  async function handleFileUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const base64 =
        typeof reader.result === "string"
          ? (reader.result.split(",")[1] ?? null)
          : null;
      setImageBase64(base64);
      setImageMimeType(file.type || null);
    };
    reader.readAsDataURL(file);
  }

  function clearAttachment() {
    setImageBase64(null);
    setImageMimeType(null);
  }

  async function onSolve() {
    if (!activePrompt && !imageBase64) return;

    try {
      await solveMutation.mutateAsync({
        input: {
          text: activePrompt,
          image_base64: imageBase64,
          image_mime_type: imageMimeType,
          language: "en",
        },
        options: {
          include_visualization: true,
        },
      });
    } catch {
      // React Query exposes the error through solveMutation.error.
    }
  }

  return (
    <div
      className={cn(
        "mx-auto grid w-full gap-6 px-4 py-6 lg:px-6 lg:py-8",
        hasVisualization
          ? "max-w-[100rem] lg:grid-cols-[minmax(320px,0.85fr)_minmax(0,1.25fr)] xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.25fr)_minmax(440px,0.9fr)]"
          : "max-w-7xl lg:grid-cols-[minmax(320px,0.85fr)_minmax(0,1.35fr)]",
      )}
    >
      <section className="space-y-4">
        <Card className="border-border/70">
          <CardHeader>
            <CardTitle>Problem</CardTitle>
            <CardDescription>
              Plain typed math works best. Image OCR requires a configured model
              backend.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <Textarea
              className="min-h-[180px]"
              onChange={(event) => setInput(event.target.value)}
              placeholder="Example: Solve 2x + 5 = 17 and explain each step."
              value={input}
            />

            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {examples.map((example) => (
                  <button
                    key={example}
                    className="rounded-full border border-border px-3 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary"
                    onClick={() => setInput(example)}
                    type="button"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-center">
              <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground hover:border-primary/40 hover:bg-primary/5">
                <ImagePlus className="h-4 w-4 text-primary" />
                <span>{imageBase64 ? "Image attached" : "Attach image"}</span>
                <Input
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileUpload}
                  type="file"
                />
              </label>
              {imageBase64 ? (
                <Button
                  onClick={clearAttachment}
                  type="button"
                  variant="outline"
                >
                  Remove
                </Button>
              ) : null}
            </div>

            <Button
              className="w-full gap-2"
              disabled={
                solveMutation.isPending || (!activePrompt && !imageBase64)
              }
              onClick={onSolve}
            >
              {solveMutation.isPending ? (
                <LoaderCircle className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              Solve
            </Button>

            {solveMutation.error ? (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-300">
                {solveMutation.error.message}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </section>

      <section className="space-y-4">
        {result ? (
          <>
            <Card className="border-border/70">
              <CardHeader>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="success">Solved</Badge>
                      {hasQuestionSwitcher ? (
                        <Badge variant="secondary">
                          Question {activeQuestionLabel}
                        </Badge>
                      ) : null}
                    </div>
                    <CardTitle className="mt-3">Answer</CardTitle>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">
                      {titleCase(result.problem_type)}
                    </Badge>
                    <Badge variant="outline">
                      Confidence {formatConfidence(result.confidence)}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {activeAnswer ? (
                  <MathText
                    latex={activeAnswer.latex}
                    text={activeAnswer.text}
                  />
                ) : null}
              </CardContent>
            </Card>

            {hasQuestionSwitcher ? (
              <Card className="border-border/70">
                <CardHeader>
                  <CardTitle className="text-base">Questions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {questionParts.map((part, index) => {
                      const label = toQuestionLabel(index);
                      return (
                        <button
                          aria-label={`Go to question ${label}`}
                          className={cn(
                            "rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                            safeActiveQuestionIndex === index
                              ? "border-primary bg-primary/10 text-primary"
                              : "border-border text-muted-foreground hover:border-primary/40 hover:bg-primary/5 hover:text-primary",
                          )}
                          key={`${label}-${part.question}`}
                          onClick={() => setActiveQuestionIndex(index)}
                          type="button"
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>

                  {activeQuestionPart?.question ? (
                    <p className="rounded-xl bg-secondary/50 p-3 text-sm leading-6 text-muted-foreground">
                      <span className="mr-2 font-medium text-foreground">
                        Question {activeQuestionLabel}:
                      </span>
                      {activeQuestionPart.question}
                    </p>
                  ) : null}
                </CardContent>
              </Card>
            ) : null}

            {activeStep ? (
              <Card className="border-border/70">
                <CardHeader className="space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">
                        {hasQuestionSwitcher
                          ? `Question ${activeQuestionLabel} · `
                          : ""}
                        Step {safeActiveStepIndex + 1} of {steps.length}
                      </Badge>
                      <CheckCircle2 className="h-4 w-4 text-primary" />
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        disabled={safeActiveStepIndex === 0}
                        onClick={() =>
                          setActiveStepIndex((index) => Math.max(0, index - 1))
                        }
                        size="sm"
                        type="button"
                        variant="outline"
                      >
                        <ChevronLeft className="mr-1 h-4 w-4" />
                        Previous
                      </Button>
                      <Button
                        disabled={safeActiveStepIndex === steps.length - 1}
                        onClick={() =>
                          setActiveStepIndex((index) =>
                            Math.min(index + 1, steps.length - 1),
                          )
                        }
                        size="sm"
                        type="button"
                        variant="outline"
                      >
                        Next
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {steps.map((step, index) => (
                      <button
                        aria-label={`Go to step ${index + 1}`}
                        className={cn(
                          "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                          safeActiveStepIndex === index
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border text-muted-foreground hover:border-primary/40 hover:bg-primary/5 hover:text-primary",
                        )}
                        key={`${step.index}-${step.title}`}
                        onClick={() => setActiveStepIndex(index)}
                        type="button"
                      >
                        {index + 1}
                      </button>
                    ))}
                  </div>

                  <CardTitle className="text-lg">{activeStep.title}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <MathText
                    latex={activeStep.latex}
                    text={activeStep.explanation}
                  />

                  {activeStep.why_it_happens ? (
                    <div className="rounded-xl bg-secondary/60 p-4 text-sm leading-6 text-muted-foreground">
                      <p className="mb-1 font-medium text-foreground">
                        Why it works
                      </p>
                      {activeStep.why_it_happens}
                    </div>
                  ) : null}

                  {activeStepCommonMistakes.length ||
                  activeStepSupportItems.length ? (
                    <div className="rounded-xl border border-border bg-background/50 p-4">
                      <div
                        className={cn(
                          "grid gap-4",
                          activeStepHasBothSupportSections && "md:grid-cols-2",
                        )}
                      >
                        {activeStepCommonMistakes.length ? (
                          <section>
                            <p className="flex items-center gap-2 text-sm font-medium text-foreground">
                              <AlertTriangle className="h-4 w-4 text-amber-500" />
                              Watch for
                            </p>
                            <ul className="mt-3 space-y-2 text-sm leading-6 text-muted-foreground">
                              {activeStepCommonMistakes.map((mistake) => (
                                <li key={mistake}>• {mistake}</li>
                              ))}
                            </ul>
                          </section>
                        ) : null}

                        {activeStepSupportItems.length ? (
                          <section>
                            <p className="flex items-center gap-2 text-sm font-medium text-foreground">
                              <BookOpenCheck className="h-4 w-4 text-primary" />
                              Helpful notes
                            </p>
                            <ul className="mt-3 space-y-2 text-sm leading-6 text-muted-foreground">
                              {activeStepSupportItems.map((item) => (
                                <li key={item}>• {item}</li>
                              ))}
                            </ul>
                          </section>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            ) : null}

            {result.warnings.length ? (
              <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/20 dark:text-amber-200">
                <p className="mb-2 flex items-center gap-2 font-medium">
                  <AlertTriangle className="h-4 w-4" />
                  Backend notes
                </p>
                <ul className="space-y-1">
                  {result.warnings.map((warning) => (
                    <li key={warning}>• {warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </>
        ) : (
          <Card className="min-h-[420px] border-dashed border-border/80">
            <CardContent className="flex min-h-[420px] flex-col items-center justify-center p-10 text-center">
              <Sparkles className="h-8 w-8 text-primary/70" />
              <h2 className="mt-4 text-xl font-semibold">
                Your answer will appear here
              </h2>
              <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
                Start with a clear typed problem. If the backend can generate a
                graph or construction, it will appear in the visualization
                panel.
              </p>
            </CardContent>
          </Card>
        )}
      </section>

      {hasVisualization ? (
        <aside className="space-y-4 lg:col-span-2 xl:col-span-1 xl:sticky xl:top-24 xl:self-start">
          <GeoGebraApplet key={visualizationKey} commands={commands} />
        </aside>
      ) : null}
    </div>
  );
}

function toQuestionLabel(index: number) {
  let label = "";
  let cursor = index;

  while (cursor >= 0) {
    label = `${String.fromCharCode(97 + (cursor % 26))}${label}`;
    cursor = Math.floor(cursor / 26) - 1;
  }

  return label;
}
