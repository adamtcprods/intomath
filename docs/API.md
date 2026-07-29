# IntoMath API

## Base URL

Local default:

```text
http://localhost:8000/api/v1
```

## Endpoints

### `GET /health`
Basic health check.

### `POST /solve`
Primary structured solving endpoint.

## Request shape

```json
{
  "input": {
    "text": "Solve 2x + 5 = 17 and explain each step.",
    "image_base64": null,
    "image_mime_type": null,
    "language": "en"
  },
  "options": {
    "include_visualization": true
  }
}
```

## Request fields

### `input`
- `text: string`
- `image_base64: string | null`
- `image_mime_type: string | null`
- `language: string`

### `options`
- `include_visualization: boolean`

## Response shape

```json
{
  "request_id": "9e71e8a1-2c7c-4d75-8b7d-94d2f4e15c3e",
  "status": "ok",
  "problem_type": "algebra",
  "difficulty": "easy",
  "answer": {
    "text": "The solution is x = 6.",
    "latex": "x = 6"
  },
  "steps": [
    {
      "index": 1,
      "title": "Isolate the variable term",
      "explanation": "Move constants to the other side so the x-term is alone.",
      "why_it_happens": "Equivalent operations preserve the equality while simplifying the equation.",
      "common_mistakes": [
        "Changing the sign incorrectly when moving a term across the equals sign."
      ],
      "alternative_approaches": [],
      "hints": [
        "Undo addition or subtraction before dividing."
      ],
      "exam_tip": "Check the result by substitution.",
      "latex": [
        "x = 6"
      ]
    }
  ],
  "parts": [],
  "visualization": {
    "kind": "none",
    "summary": null,
    "dsl": null,
    "geogebra": null
  },
  "confidence": 0.92,
  "routing": {
    "parser_model": "openai/gpt-oss-20b",
    "solver_model": "openai/gpt-oss-20b",
    "vision_model": null,
    "visualization_environment": null,
    "reason": "classified as algebra; difficulty assessed as easy; kept on the lower-latency model"
  },
  "cached": false,
  "warnings": []
}
```

## Response fields

### Top-level
- `request_id: string`
- `status: string`
- `problem_type: string`
- `difficulty: string`
- `answer: SolveAnswer`
- `steps: SolveStep[]`
- `parts: SolvePart[]` (per-question answers and steps for multi-part prompts; labels are normalized to `a`, `b`, `c`, ...)
- `visualization: VisualizationPayload`
- `confidence: number`
- `routing: RoutingPayload`
- `cached: boolean`
- `warnings: string[]`

When NVIDIA NIM directly serves the structured solve, `routing.solver_model` is
prefixed with `nvidia-direct:` followed by the exact NVIDIA-native model ID. This
keeps the serving provider visible in the response.

`routing.visualization_environment` is the local model's semantic selection and is
one of `geometry_2d`, `graphing`, `graphics_3d`, `cas`, `probability`, `statistics`,
`spreadsheet`, or `null` when the model selected no interactive view.

### `answer`
- `text: string`
- `latex: string | null`

### `steps[]`
- `index: number`
- `title: string`
- `explanation: string`
- `why_it_happens?: string`
- `common_mistakes: string[]`
- `alternative_approaches: string[]`
- `hints: string[]`
- `exam_tip?: string`
- `latex: string[]`

### `parts[]`
- `label: string` (`a`, `b`, `c`, ... by order)
- `question: string`
- `answer: SolveAnswer`
- `steps: SolveStep[]`

### `visualization`
- `kind: "none" | "graph" | "geogebra"`
- `summary?: string | null`
- `dsl?: GeometryDSL | null`
- `geogebra?: GeoGebraPayload | null`

### `geogebra`
- `commands: string[]`
- `command_string: string`
- `validation_passed: boolean`
- `issues: string[]` (backward-compatible human-readable messages)
- `validation_issues: GeoGebraValidationIssue[]`
- `environment: "geometry_2d" | "graphing" | "graphics_3d" | "cas" | "probability" | "statistics" | "spreadsheet"`
- `retrieved_commands: {name, score, signatures}[]` (development only when `APP_DEBUG=true`)

Each structured validation issue contains:

```json
{
  "code": "undefined_reference",
  "action_index": 3,
  "command": "Tangent",
  "output_label": "t",
  "message": "Reference 'c' is never produced by this construction.",
  "severity": "error"
}
```

### Geometry DSL `1.1`

DSL `1.1` is the only accepted visualization format. Every payload must declare
`version`, `space`, `environment`, `actions`, and typed `render_hints`; explicit
DSL `1.0` payloads are rejected. The existing high-level actions,
`DEFINE_OBJECT`, and `EXECUTE_COMMAND` are represented in `1.1`:

```json
{
  "version": "1.1",
  "space": "euclidean_2d",
  "environment": "geometry_2d",
  "actions": [
    {"action": "CREATE_POINT", "label": "A", "coordinates": [0, 0]},
    {"action": "CREATE_POINT", "label": "B", "coordinates": [4, 0]},
    {
      "action": "EXECUTE_COMMAND",
      "output": "lAB",
      "command": "Line",
      "arguments": [
        {"kind": "reference", "value": "A"},
        {"kind": "reference", "value": "B"}
      ]
    }
  ],
  "render_hints": {
    "perspective": "geometry_2d",
    "styles": [{"label": "lAB", "color": "#2563EB", "line_thickness": 3}],
    "viewport": {"x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5}
  }
}
```

Generic argument kinds are `reference`, `number`, `angle`, `point`, `vector`,
`text`, `boolean`, `expression`, `equation`, `list`, and `interval`. The backend
may reorder actions by explicit references. Generic commands without `output`
are allowed only as unreferenceable terminal results and produce an
`untracked_output` warning.

Definitions that are not command calls use `DEFINE_OBJECT`:

```json
{
  "action": "DEFINE_OBJECT",
  "output": "f",
  "object_type": "function",
  "value": {"kind": "equation", "value": "f(x) = x^2"}
}
```

Generic `EXECUTE_COMMAND` actions are restricted to the 10–20 safe command
names retrieved for that prompt. Both accepted and experimental safe overloads
can be translated. Experimental use produces a validation warning and remains
protected by browser rejection handling and rollback. Overloads marked
`blocked` cannot be emitted or translated.

## OCR flow

If `image_base64` is present:
1. the image is sent through the configured OCR model
2. extracted text is merged into the normalized prompt
3. the normalized prompt is routed to the appropriate solver

## Visualization flow

If `include_visualization` is true:
1. the tiny router model selects the visualization environment; `none` stops the visualization flow here
2. the tiny model expands the request into semantic GeoGebra search terms and a bounded set of at most 10 relevant commands is retrieved from the local catalog
3. a model emits typed DSL; if every model-backed parser fails, no visualization is emitted
4. schema, command selection, signature/type/environment and dependency validation run
5. a remote plan with action-scoped validation errors gets at most one compact repair turn, followed by full re-validation
6. trusted code translates the sorted DSL into GeoGebra commands
7. both DSL and commands are returned; the browser validates each command again at runtime

Remote solve and geometry requests supply strict JSON schemas in their prompts and
validate each response locally. A provider failure is logged and
warned separately from a model plan that parsed but failed deterministic validation.
Geometry then tries the validated local llama.cpp parser, NVIDIA direct gpt-oss-120b
and gpt-oss-20b in that order. Every proposed DSL—including NVIDIA direct output—runs through the same
authoritative validators. Invalid output follows this chain too; it does not stop at an
invalid but parseable DSL.

Remote attempts have a 25-second default hard timeout. Only direct gpt-oss-120b
receives a model-specific 50-second cold-start allowance. Structured solve tries the
preferred and alternate explicit NVIDIA gpt-oss models. NVIDIA calls
are non-streaming. Their hosted request contracts omit `response_format`, so responses
are logged as unenforced proposals; gpt-oss reasoning effort is low. Each structured
step logs whether `latex` is empty; math notation
without a matching formula is surfaced in `warnings`. Scratch-work-style explanation
fields receive at most one bounded cleanup turn on either provider path.

If visualization validation fails, the solve response remains `status: "ok"`
when the mathematical solution succeeded. The visualization has no executable
commands (and normally `kind: "none"`), while `warnings` and
`validation_issues` explain the visualization failure.

## Notes

- The frontend currently posts to this API via `frontend/lib/api-client.ts`.
- The API is intentionally structured for UI rendering rather than chat transcript playback.
- `commands: string[]` and `command_string` remain available for older clients.
- Raw model-generated GeoGebra command mode is not exposed by this API.
