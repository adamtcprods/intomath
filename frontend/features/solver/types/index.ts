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
  points?: string[];
  coordinates?: [number, number] | null;
  center?: string | null;
  radius?: number | null;
  through?: string[] | null;
  equation?: string | null;
  line?: string | null;
}

export interface GeometryDsl {
  version: string;
  actions: GeometryAction[];
  render_hints?: Record<string, unknown>;
}

export interface SolveVisualization {
  kind: "geogebra" | "graph" | "none";
  summary?: string;
  dsl?: GeometryDsl | null;
  geogebra?: {
    commands: string[];
    command_string: string;
  } | null;
}

export interface RoutingDecision {
  parser_model: string;
  solver_model: string;
  vision_model?: string | null;
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
