"use client";

import {
  type ChangeEvent,
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "next/navigation";
import {
  AlertTriangle,
  BookOpenCheck,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ImagePlus,
  LoaderCircle,
  Sparkles,
  X,
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
import { cn } from "@/lib/utils";

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
    loadProblem,
    clearAttachment: clearStoredAttachment,
  } = useSolveWorkspaceStore();
  const solveMutation = useSolveProblem();
  const searchParams = useSearchParams();
  const promptFromUrl = searchParams.get("prompt");
  const result = solveMutation.data;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileReaderRef = useRef<FileReader | null>(null);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [activeQuestionIndex, setActiveQuestionIndex] = useState(0);

  const activePrompt = useMemo(() => input.trim(), [input]);
  const geogebra = result?.visualization.geogebra;
  const commands = geogebra?.commands ?? [];
  const visualizationEnvironment =
    geogebra?.environment ??
    result?.visualization.dsl?.environment ??
    (result?.visualization.kind === "graph" ? "graphing" : "geometry_2d");
  const renderHints = result?.visualization.dsl?.render_hints;
  const visualizationKey = useMemo(
    () =>
      JSON.stringify({
        commands,
        environment: visualizationEnvironment,
        renderHints,
      }),
    [commands, renderHints, visualizationEnvironment],
  );
  const hasVisualization = commands.length > 0;
  const warnings = result?.warnings ?? [];
  const visualizationWarnings =
    geogebra?.validation_passed === false &&
    !warnings.some((warning) =>
      warning.toLowerCase().includes("visualization plan failed validation"),
    )
      ? [
          ...warnings,
          "The model-generated visualization plan failed validation, so no shape could be constructed.",
        ]
      : warnings;
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
    if (!promptFromUrl) return;
    fileReaderRef.current?.abort();
    fileReaderRef.current = null;
    loadProblem(promptFromUrl);
  }, [loadProblem, promptFromUrl]);

  useEffect(
    () => () => {
      fileReaderRef.current?.abort();
    },
    [],
  );

  useEffect(() => {
    setActiveStepIndex(0);
  }, [safeActiveQuestionIndex]);

  function handleFileUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    event.currentTarget.value = "";

    fileReaderRef.current?.abort();
    const reader = new FileReader();
    fileReaderRef.current = reader;
    reader.onload = () => {
      if (fileReaderRef.current !== reader) return;
      const base64 =
        typeof reader.result === "string"
          ? (reader.result.split(",")[1] ?? null)
          : null;
      setImageBase64(base64);
      setImageMimeType(file.type || null);
      fileReaderRef.current = null;
    };
    reader.onerror = () => {
      if (fileReaderRef.current === reader) fileReaderRef.current = null;
    };
    reader.onabort = () => {
      if (fileReaderRef.current === reader) fileReaderRef.current = null;
    };
    reader.readAsDataURL(file);
  }

  function clearAttachment() {
    fileReaderRef.current?.abort();
    fileReaderRef.current = null;
    clearStoredAttachment();
  }

  function selectExample(example: string) {
    fileReaderRef.current?.abort();
    fileReaderRef.current = null;
    loadProblem(example);
  }

  function clearProblemBox() {
    setInput("");
    clearAttachment();
  }

  function handlePromptKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key !== "Enter" ||
      (!event.metaKey && !event.ctrlKey) ||
      event.nativeEvent.isComposing
    ) {
      return;
    }

    event.preventDefault();
    if (!solveMutation.isPending) void onSolve();
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
        result
          ? hasVisualization
            ? "max-w-[100rem] lg:grid-cols-[minmax(320px,0.85fr)_minmax(0,1.25fr)] xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.25fr)_minmax(440px,0.9fr)]"
            : "max-w-7xl lg:grid-cols-[minmax(320px,0.85fr)_minmax(0,1.35fr)]"
          : "max-w-3xl",
      )}
    >
      <section className="space-y-4">
        <Card className="border-border/70">
          <CardHeader>
            <CardTitle id="problem-input-label">What do you want to solve?</CardTitle>
            <CardDescription id="problem-input-help">
              Type a problem or attach a photo. Clear, specific questions work
              best.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="overflow-hidden rounded-xl border border-input bg-background shadow-sm transition-shadow focus-within:ring-2 focus-within:ring-ring">
              <Textarea
                aria-describedby="problem-input-help"
                aria-labelledby="problem-input-label"
                className={cn(
                  "resize-none rounded-none border-0 bg-transparent shadow-none focus-visible:ring-0",
                  result ? "min-h-[180px]" : "min-h-[220px]",
                )}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handlePromptKeyDown}
                placeholder="Type or paste your math problem here…"
                value={input}
              />
              <div className="flex items-center justify-between border-t border-input px-2 py-2">
                <Button
                  aria-label={
                    imageBase64 ? "Replace attached image" : "Add an image"
                  }
                  className={cn(
                    "gap-2 text-muted-foreground",
                    imageBase64 &&
                      "bg-primary/10 text-primary hover:bg-primary/15 hover:text-primary",
                  )}
                  onClick={() => fileInputRef.current?.click()}
                  size="sm"
                  type="button"
                  variant="ghost"
                >
                  <ImagePlus className="h-4 w-4" />
                  {imageBase64 ? "Replace image" : "Attach image"}
                </Button>
                <Input
                  accept="image/jpeg,image/png,image/webp,image/gif"
                  className="hidden"
                  onChange={handleFileUpload}
                  ref={fileInputRef}
                  type="file"
                />
                <div className="flex items-center gap-2">
                  <span className="hidden text-xs text-muted-foreground sm:inline">
                    Ctrl/⌘ + Enter to solve
                  </span>
                  {input || imageBase64 ? (
                    <Button
                      aria-label="Clear problem and attached image"
                      className="text-muted-foreground hover:text-foreground"
                      onClick={clearProblemBox}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  ) : null}
                </div>
              </div>
            </div>

            {imageBase64 ? (
              <div className="flex items-center justify-between gap-3 rounded-xl border border-primary/20 bg-primary/5 px-3 py-2 text-sm">
                <span className="flex min-w-0 items-center gap-2 text-primary">
                  <ImagePlus className="h-4 w-4 shrink-0" />
                  <span className="truncate">
                    Image attached{imageMimeType ? ` · ${imageMimeType}` : ""}
                  </span>
                </span>
                <Button
                  aria-label="Remove attached image"
                  className="h-8 w-8 shrink-0"
                  onClick={clearAttachment}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : null}

            <details className="group">
              <summary className="cursor-pointer text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                Try an example
              </summary>
              <div className="mt-3 flex flex-wrap gap-2">
                {examples.map((example) => (
                  <button
                    key={example}
                    className="rounded-full border border-border px-3 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
                    onClick={() => selectExample(example)}
                    type="button"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </details>

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
                <p className="font-medium">We couldn&apos;t solve that yet.</p>
                <p className="mt-1">
                  Try rewording the problem, checking your connection, or using
                  a clearer image.
                </p>
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

            {visualizationWarnings.length ? (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/20 dark:text-amber-200">
                <p className="flex items-center gap-2 font-medium">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  Please note
                </p>
                <ul className="mt-2 space-y-1">
                  {visualizationWarnings.map((warning, index) => (
                    <li key={`${warning}-${index}`}>• {warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}

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
                            "rounded-full border px-4 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card",
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
                          "rounded-full border px-3 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card",
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
          </>
        ) : null}
      </section>

      {hasVisualization ? (
        <aside className="space-y-4 lg:col-span-2 xl:col-span-1 xl:sticky xl:top-24 xl:self-start">
          <GeoGebraApplet
            commands={commands}
            environment={visualizationEnvironment}
            key={visualizationKey}
            renderHints={renderHints}
          />
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
