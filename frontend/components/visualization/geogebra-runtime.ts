import type {
  GeoGebraRenderHints,
  VisualizationEnvironment,
} from "@/features/solver/types";

export interface GeoGebraApi {
  evalCommand: (command: string) => boolean;
  getXML?: () => string;
  setXML?: (xml: string) => void;
  newConstruction?: () => void;
  remove?: () => void;
  setErrorDialogsActive?: (active: boolean) => void;
  enableCAS?: (enable: boolean) => void;
  enable3D?: (enable: boolean) => void;
  setPerspective?: (perspective: string) => void;
  setColor?: (label: string, red: number, green: number, blue: number) => void;
  setLineThickness?: (label: string, thickness: number) => void;
  setLineStyle?: (label: string, style: number) => void;
  setPointSize?: (label: string, size: number) => void;
  setLabelVisible?: (label: string, visible: boolean) => void;
  setVisible?: (label: string, visible: boolean) => void;
  setFixed?: (
    label: string,
    fixed: boolean,
    selectionAllowed: boolean,
  ) => void;
  setCaption?: (label: string, caption: string) => void;
  setCoordSystem?: (...bounds: Array<number | boolean>) => void;
  setAxesVisible?: (...values: Array<number | boolean>) => void;
  setGridVisible?: (...values: Array<number | boolean>) => void;
  showAllObjects?: () => void;
  setAnimating?: (label: string, animate: boolean) => void;
  startAnimation?: () => void;
  stopAnimation?: () => void;
}

export interface CommandExecutionFailure {
  index: number;
  command: string;
  reason: "returned_false" | "exception";
  detail: string;
  rollback: "restored" | "cleared" | "unavailable" | "failed";
}

export interface CommandExecutionResult {
  success: boolean;
  executed: number;
  failure?: CommandExecutionFailure;
}

const SAFE_LABEL = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
const SAFE_COLOR = /^#[0-9A-Fa-f]{6}$/;

export const PERSPECTIVES: Record<VisualizationEnvironment, string> = {
  geometry_2d: "2",
  graphing: "1",
  // "5" is the 3D preset, which includes Algebra ("AT"). "T" is 3D only.
  graphics_3d: "T",
  cas: "4",
  probability: "6",
  statistics: "6",
  spreadsheet: "3",
};

function errorDetail(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function rollback(
  api: GeoGebraApi,
  snapshot: string | null,
): CommandExecutionFailure["rollback"] {
  try {
    if (snapshot !== null && api.setXML) {
      api.setXML(snapshot);
      return "restored";
    }
    if (api.newConstruction) {
      api.newConstruction();
      return "cleared";
    }
    return "unavailable";
  } catch {
    return "failed";
  }
}

export function executeGeoGebraCommands(
  api: GeoGebraApi,
  commands: readonly string[],
): CommandExecutionResult {
  let snapshot: string | null = null;
  try {
    snapshot = api.getXML?.() ?? null;
  } catch {
    snapshot = null;
  }

  api.setErrorDialogsActive?.(false);
  for (let index = 0; index < commands.length; index += 1) {
    const command = commands[index];
    try {
      if (api.evalCommand(command) !== true) {
        return {
          success: false,
          executed: index,
          failure: {
            index,
            command,
            reason: "returned_false",
            detail: "GeoGebra rejected the command.",
            rollback: rollback(api, snapshot),
          },
        };
      }
    } catch (error) {
      return {
        success: false,
        executed: index,
        failure: {
          index,
          command,
          reason: "exception",
          detail: errorDetail(error),
          rollback: rollback(api, snapshot),
        },
      };
    }
  }

  return { success: true, executed: commands.length };
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isBoundedInteger(
  value: unknown,
  minimum: number,
  maximum: number,
): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= minimum &&
    value <= maximum
  );
}

export function configureGeoGebraView(
  api: GeoGebraApi,
  environment: VisualizationEnvironment,
  hints?: GeoGebraRenderHints,
) {
  const perspectiveEnvironment = hints?.perspective ?? environment;
  api.enableCAS?.(perspectiveEnvironment === "cas");
  api.enable3D?.(perspectiveEnvironment === "graphics_3d");
  api.setPerspective?.(PERSPECTIVES[perspectiveEnvironment]);
}

export function applyGeoGebraRenderHints(
  api: GeoGebraApi,
  environment: VisualizationEnvironment,
  hints?: GeoGebraRenderHints,
) {
  for (const style of hints?.styles ?? []) {
    if (!SAFE_LABEL.test(style.label)) continue;
    if (typeof style.color === "string" && SAFE_COLOR.test(style.color)) {
      const red = Number.parseInt(style.color.slice(1, 3), 16);
      const green = Number.parseInt(style.color.slice(3, 5), 16);
      const blue = Number.parseInt(style.color.slice(5, 7), 16);
      api.setColor?.(style.label, red, green, blue);
    }
    if (isBoundedInteger(style.line_thickness, 1, 13)) {
      api.setLineThickness?.(style.label, style.line_thickness);
    }
    if (isBoundedInteger(style.line_style, 0, 4)) {
      api.setLineStyle?.(style.label, style.line_style);
    }
    if (isBoundedInteger(style.point_size, 1, 9)) {
      api.setPointSize?.(style.label, style.point_size);
    }
    if (typeof style.label_visible === "boolean") {
      api.setLabelVisible?.(style.label, style.label_visible);
    }
    if (typeof style.visible === "boolean") {
      api.setVisible?.(style.label, style.visible);
    }
    if (typeof style.fixed === "boolean") {
      api.setFixed?.(style.label, style.fixed, !style.fixed);
    }
    if (
      typeof style.caption === "string" &&
      style.caption.length <= 120 &&
      !/[\n\r\0]/.test(style.caption)
    ) {
      api.setCaption?.(style.label, style.caption);
    }
  }

  const viewport = hints?.viewport;
  if (viewport) {
    const twoDimensionalBounds = [
      viewport.x_min,
      viewport.x_max,
      viewport.y_min,
      viewport.y_max,
    ];
    const threeDimensionalBounds = [
      ...twoDimensionalBounds,
      viewport.z_min,
      viewport.z_max,
    ];
    if (
      environment === "graphics_3d" &&
      threeDimensionalBounds.every(isFiniteNumber) &&
      viewport.x_min! < viewport.x_max! &&
      viewport.y_min! < viewport.y_max! &&
      viewport.z_min! < viewport.z_max!
    ) {
      api.setCoordSystem?.(...(threeDimensionalBounds as number[]), true);
    } else if (
      environment !== "graphics_3d" &&
      twoDimensionalBounds.every(isFiniteNumber) &&
      viewport.x_min! < viewport.x_max! &&
      viewport.y_min! < viewport.y_max!
    ) {
      api.setCoordSystem?.(...(twoDimensionalBounds as number[]));
    }
    if (typeof viewport.axes_visible === "boolean") {
      if (environment === "graphics_3d") {
        api.setAxesVisible?.(
          3,
          viewport.axes_visible,
          viewport.axes_visible,
          viewport.axes_visible,
        );
      } else {
        api.setAxesVisible?.(viewport.axes_visible, viewport.axes_visible);
      }
    }
    if (typeof viewport.grid_visible === "boolean") {
      api.setGridVisible?.(
        ...(environment === "graphics_3d"
          ? [3, viewport.grid_visible]
          : [viewport.grid_visible]),
      );
    }
  }

  const interaction = hints?.interaction;
  for (const label of interaction?.movable_points ?? []) {
    if (SAFE_LABEL.test(label)) api.setFixed?.(label, false, true);
  }
  for (const label of interaction?.animated_objects ?? []) {
    if (SAFE_LABEL.test(label)) api.setAnimating?.(label, true);
  }
  if (interaction?.animation === "start") api.startAnimation?.();
  if (interaction?.animation === "stop") api.stopAnimation?.();
}
