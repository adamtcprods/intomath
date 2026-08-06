"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Construction,
  ExternalLink,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";

import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type {
  GeoGebraRenderHints,
  VisualizationEnvironment,
} from "@/features/solver/types";
import {
  applyGeoGebraRenderHints,
  configureGeoGebraView,
  executeGeoGebraCommands,
  type CommandExecutionFailure,
  type GeoGebraApi,
} from "@/components/visualization/geogebra-runtime";

type GeoGebraAppletInstance = {
  inject: (container: string) => void;
  getAppletObject?: () => GeoGebraApi | undefined;
};

declare global {
  interface Window {
    GGBApplet?: new (
      config: Record<string, unknown>,
      ready: boolean,
    ) => GeoGebraAppletInstance;
  }
}

const GEOGEBRA_SCRIPT_URLS = [
  "https://www.geogebra.org/apps/deployggb.js",
  "https://cdn.geogebra.org/apps/deployggb.js",
] as const;
const GEOGEBRA_SCRIPT_TIMEOUT_MS = 10_000;
const GEOGEBRA_APPLET_TIMEOUT_MS = 15_000;

let geogebraScriptPromise: Promise<void> | null = null;

type LoadStatus = "idle" | "loading" | "ready" | "error";

interface GeoGebraAppletProps {
  commands: string[];
  environment?: VisualizationEnvironment;
  renderHints?: GeoGebraRenderHints;
}

function geogebraLoadError() {
  return new Error(
    "GeoGebra could not be loaded. Check your network connection or try opening GeoGebra in a new tab and paste the commands below.",
  );
}

function loadGeoGebraScriptFrom(index: number): Promise<void> {
  if (window.GGBApplet) {
    return Promise.resolve();
  }

  const src = GEOGEBRA_SCRIPT_URLS[index];
  if (!src) {
    return Promise.reject(geogebraLoadError());
  }

  return new Promise((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>(
      `script[src="${src}"]`,
    );

    if (
      existingScript?.dataset.geogebraStatus === "ready" &&
      window.GGBApplet
    ) {
      resolve();
      return;
    }

    if (
      existingScript?.dataset.geogebraStatus === "error" ||
      existingScript?.dataset.geogebraStatus === "timeout" ||
      existingScript?.dataset.geogebraStatus === "ready"
    ) {
      existingScript.remove();
    }

    const script =
      existingScript?.isConnected === true
        ? existingScript
        : document.createElement("script");
    const shouldAppend = !script.isConnected;
    let settled = false;

    const cleanup = () => {
      window.clearTimeout(timeoutId);
      script.removeEventListener("load", handleLoad);
      script.removeEventListener("error", handleError);
    };

    const tryNextSource = () => {
      cleanup();
      if (script.dataset.geogebraStatus !== "ready") {
        script.remove();
      }
      loadGeoGebraScriptFrom(index + 1)
        .then(resolve)
        .catch(reject);
    };

    const handleLoad = () => {
      if (settled) return;
      settled = true;

      if (window.GGBApplet) {
        script.dataset.geogebraStatus = "ready";
        cleanup();
        resolve();
        return;
      }

      script.dataset.geogebraStatus = "error";
      tryNextSource();
    };

    const handleError = () => {
      if (settled) return;
      settled = true;
      script.dataset.geogebraStatus = "error";
      tryNextSource();
    };

    const handleTimeout = () => {
      if (settled) return;
      settled = true;
      script.dataset.geogebraStatus = "timeout";
      tryNextSource();
    };

    script.src = src;
    script.async = true;
    script.dataset.geogebraStatus = "loading";
    script.addEventListener("load", handleLoad, { once: true });
    script.addEventListener("error", handleError, { once: true });
    const timeoutId = window.setTimeout(
      handleTimeout,
      GEOGEBRA_SCRIPT_TIMEOUT_MS,
    );

    if (shouldAppend) {
      document.body.appendChild(script);
    }
  });
}

function loadGeoGebraScript() {
  if (window.GGBApplet) {
    return Promise.resolve();
  }

  geogebraScriptPromise ??= loadGeoGebraScriptFrom(0).catch(
    (error: unknown) => {
      geogebraScriptPromise = null;
      throw error;
    },
  );

  return geogebraScriptPromise;
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : geogebraLoadError().message;
}

export function GeoGebraApplet({
  commands,
  environment = "geometry_2d",
  renderHints,
}: GeoGebraAppletProps) {
  const reactId = useId();
  const containerId = useMemo(
    () => `geogebra-${reactId.replace(/[^a-zA-Z0-9_-]/g, "")}`,
    [reactId],
  );
  const appletCommands = useMemo(
    () => commands.map((command) => command.trim()).filter(Boolean),
    [commands],
  );
  const commandString = useMemo(
    () => appletCommands.join("\n"),
    [appletCommands],
  );

  const [scriptStatus, setScriptStatus] = useState<LoadStatus>("idle");
  const [appletStatus, setAppletStatus] = useState<LoadStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [runtimeFailure, setRuntimeFailure] =
    useState<CommandExecutionFailure | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const injectedRef = useRef(false);
  const apiRef = useRef<GeoGebraApi | null>(null);

  useEffect(() => {
    let cancelled = false;
    injectedRef.current = false;
    setErrorMessage(null);
    setRuntimeFailure(null);

    if (appletCommands.length === 0) {
      setScriptStatus("idle");
      setAppletStatus("idle");
      return;
    }

    setScriptStatus(window.GGBApplet ? "ready" : "loading");
    setAppletStatus("loading");

    loadGeoGebraScript()
      .then(() => {
        if (!cancelled) {
          setScriptStatus("ready");
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setScriptStatus("error");
          setAppletStatus("error");
          setErrorMessage(getErrorMessage(error));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [appletCommands.length, retryNonce]);

  useEffect(() => {
    if (
      appletCommands.length === 0 ||
      scriptStatus !== "ready" ||
      !window.GGBApplet ||
      injectedRef.current
    ) {
      return;
    }

    let cancelled = false;
    setAppletStatus("loading");
    setErrorMessage(null);
    setRuntimeFailure(null);
    apiRef.current?.remove?.();
    apiRef.current = null;
    document.getElementById(containerId)?.replaceChildren();

    const timeoutId = window.setTimeout(() => {
      if (!cancelled) {
        setAppletStatus("error");
        setErrorMessage(
          "GeoGebra loaded, but the applet did not finish initializing. Try again or open GeoGebra and paste the commands below.",
        );
      }
    }, GEOGEBRA_APPLET_TIMEOUT_MS);

    const markReady = (api: GeoGebraApi) => {
      if (cancelled) return;
      apiRef.current = api;

      try {
        configureGeoGebraView(api, environment, renderHints);
        const execution = executeGeoGebraCommands(api, appletCommands);
        if (!execution.success && execution.failure) {
          window.clearTimeout(timeoutId);
          setRuntimeFailure(execution.failure);
          setAppletStatus("error");
          setErrorMessage(
            `Construction command ${execution.failure.index + 1} failed: ${execution.failure.command}`,
          );
          return;
        }
        applyGeoGebraRenderHints(api, environment, renderHints);
        window.clearTimeout(timeoutId);
        setAppletStatus("ready");
      } catch (error) {
        window.clearTimeout(timeoutId);
        setAppletStatus("error");
        setErrorMessage(getErrorMessage(error));
      }
    };

    const applet = new window.GGBApplet(
      {
        appName: "classic",
        width: 420,
        height: 320,
        showToolBar: false,
        showAlgebraInput: false,
        showMenuBar: false,
        showResetIcon: false,
        enableShiftDragZoom: true,
        enableLabelDrags: true,
        useBrowserForJS: false,
        borderColor: null,
        scaleContainerClass: "geogebra-responsive",
        appletOnLoad: markReady,
      },
      true,
    );

    try {
      applet.inject(containerId);
      injectedRef.current = true;
    } catch (error) {
      window.clearTimeout(timeoutId);
      setAppletStatus("error");
      setErrorMessage(getErrorMessage(error));
    }

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
      apiRef.current?.remove?.();
      apiRef.current = null;
    };
  }, [
    appletCommands,
    containerId,
    environment,
    renderHints,
    retryNonce,
    scriptStatus,
  ]);

  const isLoading =
    appletCommands.length > 0 &&
    (scriptStatus === "loading" || appletStatus === "loading");
  const hasError = scriptStatus === "error" || appletStatus === "error";

  return (
    <Card className="h-fit overflow-hidden border-border/70">
      <CardHeader className="p-5">
        <CardTitle className="flex items-center gap-2 text-base">
          <Construction className="h-4 w-4 text-primary" />
          Visualization
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 p-5 pt-0">
        <div className="overflow-hidden rounded-2xl border border-border/70 bg-secondary/40">
          {appletCommands.length === 0 ? (
            <div className="flex h-[320px] items-center justify-center p-6 text-center text-sm text-muted-foreground">
              No visualization was generated for this problem.
            </div>
          ) : hasError ? (
            <div className="flex min-h-[320px] flex-col items-center justify-center gap-4 p-6 text-center text-sm text-muted-foreground">
              <AlertCircle className="h-6 w-6 text-amber-500" />
              <div className="space-y-2">
                <p className="font-medium text-foreground">
                  {runtimeFailure
                    ? "GeoGebra rejected this construction."
                    : "GeoGebra is unavailable right now."}
                </p>
                <p>{errorMessage ?? geogebraLoadError().message}</p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                <Button
                  onClick={() => setRetryNonce((value) => value + 1)}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Try again
                </Button>
                <a
                  className={buttonVariants({ size: "sm" })}
                  href="https://www.geogebra.org/classic"
                  rel="noreferrer"
                  target="_blank"
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open GeoGebra
                </a>
              </div>
            </div>
          ) : (
            <div className="relative min-h-[320px]">
              <div
                className="geogebra-responsive h-[320px] w-full"
                id={containerId}
              />
              {isLoading ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-secondary/80 text-sm text-muted-foreground backdrop-blur-sm">
                  <LoaderCircle className="h-5 w-5 animate-spin text-primary" />
                  Loading GeoGebra…
                </div>
              ) : null}
            </div>
          )}
        </div>
        {appletCommands.length > 0 ? (
          <details className="rounded-2xl border border-border/70 bg-background p-4 text-sm">
            <summary className="cursor-pointer font-medium text-muted-foreground transition-colors hover:text-foreground">
              View construction details
            </summary>
            <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-muted-foreground">
              {commandString}
            </pre>
            {runtimeFailure ? (
              <div className="mt-3 border-t border-border/70 pt-3 text-xs leading-5 text-amber-700 dark:text-amber-400">
                <p>
                  Failed at command {runtimeFailure.index + 1}: {runtimeFailure.command}
                </p>
                <p>{runtimeFailure.detail}</p>
                <p>Rollback: {runtimeFailure.rollback}</p>
              </div>
            ) : null}
          </details>
        ) : null}
      </CardContent>
    </Card>
  );
}
