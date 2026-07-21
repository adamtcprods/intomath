export interface SolveStep {
  index: number;
  title: string;
  explanation: string;
  why_it_happens?: string;
  common_mistakes?: string[];
  alternative_approaches?: string[];
  hints?: string[];
  exam_tip?: string;
  latex?: string[];
}

export interface SolveAnswer {
  text: string;
  latex?: string | null;
}

export interface SolvePart {
  label: string;
  question: string;
  answer: SolveAnswer;
  steps: SolveStep[];
}

export interface GeometryAction {
  action: string;
  label?: string | null;
  output?: string | null;
  object_type?: GeoGebraDefinitionObjectType | null;
  value?: GeoGebraDefinitionValue | null;
  command?: string | null;
  arguments?: GeoGebraArgument[];
  points?: string[];
  coordinates?: [number, number] | null;
  center?: string | null;
  radius?: number | null;
  through?: string[] | null;
  equation?: string | null;
  line?: string | null;
}

export type VisualizationEnvironment =
  | "geometry_2d"
  | "graphing"
  | "graphics_3d"
  | "cas"
  | "probability"
  | "statistics"
  | "spreadsheet";

export type GeoGebraDefinitionObjectType =
  | "function"
  | "equation"
  | "expression"
  | "number"
  | "point"
  | "vector"
  | "list"
  | "text"
  | "boolean"
  | "interval";

export type GeoGebraArgument =
  | { kind: "reference"; value: string }
  | { kind: "number"; value: number }
  | { kind: "angle"; value: number; unit: "degree" | "radian" }
  | { kind: "point"; x: number; y: number; z?: number | null }
  | { kind: "vector"; x: number; y: number; z?: number | null }
  | { kind: "text"; value: string }
  | { kind: "boolean"; value: boolean }
  | { kind: "expression"; value: string }
  | { kind: "equation"; value: string }
  | { kind: "list"; items: GeoGebraArgument[] }
  | {
      kind: "interval";
      lower: number;
      upper: number;
      lower_inclusive: boolean;
      upper_inclusive: boolean;
    };

export type GeoGebraDefinitionValue = Exclude<
  GeoGebraArgument,
  { kind: "reference" } | { kind: "angle" }
>;

export interface GeoGebraObjectStyle {
  label: string;
  color?: string | null;
  line_thickness?: number | null;
  line_style?: number | null;
  point_size?: number | null;
  label_visible?: boolean | null;
  visible?: boolean | null;
  fixed?: boolean | null;
  caption?: string | null;
}

export interface GeoGebraRenderHints {
  perspective?: VisualizationEnvironment | null;
  styles?: GeoGebraObjectStyle[];
  viewport?: {
    x_min?: number | null;
    x_max?: number | null;
    y_min?: number | null;
    y_max?: number | null;
    z_min?: number | null;
    z_max?: number | null;
    axes_visible?: boolean | null;
    grid_visible?: boolean | null;
  } | null;
  interaction?: {
    movable_points?: string[];
    animated_objects?: string[];
    animation?: "start" | "stop" | null;
  } | null;
}

export interface GeometryDsl {
  version: "1.1";
  space: "euclidean_2d" | "euclidean_3d";
  environment: VisualizationEnvironment;
  actions: GeometryAction[];
  render_hints: GeoGebraRenderHints;
}

export interface GeoGebraValidationIssue {
  code: string;
  action_index?: number | null;
  command?: string | null;
  output_label?: string | null;
  message: string;
  severity: "error" | "warning";
}

export interface SolveVisualization {
  kind: "geogebra" | "graph" | "none";
  summary?: string;
  dsl?: GeometryDsl | null;
  geogebra?: {
    commands: string[];
    command_string: string;
    validation_passed?: boolean;
    issues?: string[];
    validation_issues?: GeoGebraValidationIssue[];
    environment?: VisualizationEnvironment;
    retrieved_commands?: Array<{
      name: string;
      score: number;
      signatures: string[];
    }>;
  } | null;
}

export interface RoutingDecision {
  parser_model: string;
  solver_model: string;
  vision_model?: string | null;
  visualization_environment?: VisualizationEnvironment | null;
  reason: string;
}

export interface SolveResponse {
  request_id: string;
  status: "ok" | "error";
  problem_type: string;
  difficulty: string;
  answer: SolveAnswer;
  steps: SolveStep[];
  parts: SolvePart[];
  visualization: SolveVisualization;
  confidence: number;
  routing: RoutingDecision;
  cached: boolean;
  warnings: string[];
}

export interface SolveRequest {
  input: {
    text: string;
    image_base64?: string | null;
    image_mime_type?: string | null;
    language?: string;
  };
  options?: {
    include_visualization?: boolean;
  };
}
