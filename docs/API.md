# IntoMath 2.0 API

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
    "parser_model": "nvidia/nemotron-3-nano-30b-a3b:free",
    "solver_model": "nvidia/nemotron-3-nano-30b-a3b:free",
    "vision_model": null,
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
- `issues: string[]`

## OCR flow

If `image_base64` is present:
1. the image is sent through the configured OCR model
2. extracted text is merged into the normalized prompt
3. the normalized prompt is routed to the appropriate solver

## Visualization flow

If `include_visualization` is true:
1. geometry or graph intent is extracted
2. the system produces a Geometry DSL payload
3. the translator converts it into GeoGebra commands
4. both DSL and commands are returned in the response

## Notes

- The frontend currently posts to this API via `frontend/lib/api-client.ts`.
- The API is intentionally structured for UI rendering rather than chat transcript playback.
