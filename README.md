# IntoMath 2.0

Math help that actually solves.

Instead of presenting a generic AI chat interface, IntoMath is organized around a simple solve flow:

- **Problem input**
- **Structured solution**
- **Interactive visualization**

Students can type a prompt and receive:

- a structured answer
- step-by-step reasoning
- hints and common mistakes
- an interactive GeoGebra visualization when the backend can generate one

Image upload uses local OCR. For supported typed prompts, the backend tries the local deterministic solver first. It handles arithmetic, one-variable linear equations with basic parentheses/division, quadratic graphs, and simple geometry constructions. A local llama.cpp model can normalize borderline prompts and extract validated visualization DSL, while proof-style geometry can use NVIDIA NIM routing.

## Tech stack

### Frontend
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- shadcn-style component primitives
- React Query
- Zustand
- KaTeX / `react-katex`
- GeoGebra embed API

### Backend
- FastAPI
- Python 3.12+
- Pydantic v2
- SQLAlchemy
- NVIDIA NIM API

### Database
- PostgreSQL-ready via SQLAlchemy
- SQLite default for local development

## Product surfaces

### Marketing site
Located in `frontend/app/(marketing)`.

Includes:
- hero
- focused feature cards
- how it works
- direct solver CTA
- footer

### Learning workspace
Located in `frontend/app/(dashboard)`.

The solve experience is built around two focused areas:
- **Input:** prompt, optional image attachment, and examples
- **Result:** answer, steps, backend notes, and visualization when available

## Core backend architecture

### 1. Structured solving
All solver responses follow the same schema:

```json
{
  "request_id": "",
  "status": "ok",
  "problem_type": "",
  "difficulty": "",
  "answer": {
    "text": "",
    "latex": ""
  },
  "steps": [],
  "visualization": {},
  "confidence": 0.0,
  "routing": {},
  "cached": false,
  "warnings": []
}
```

This prevents the UI from depending on unpredictable free-form model output.

### 2. Model routing
The routing layer lives in `backend/app/services/model_router.py`.

Current models and local routes:
- **Deterministic local solving first:** `local:deterministic-solver`
- **Local visualization DSL extraction:** `local:llama-geometry-parser`, backed by `unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL` through llama-server with JSON-schema constraints
- **Local routing and solver normalization:** the same llama.cpp model
- **Easy / lower-latency solving via NVIDIA NIM:** `openai/gpt-oss-20b`
- **Hard / proof-heavy solving via NVIDIA NIM:** `openai/gpt-oss-120b`
- **Explicit JSON fallback routing:** NVIDIA-hosted gpt-oss models with no opaque router alias
- **OCR / visual extraction locally:** `deepseek-ai/deepseek-ocr-2`

Examples:
- supported arithmetic → `local:deterministic-solver`
- supported linear equations, including `ax+b=cx+d`, `2(x+3)=14`, and `x/2+3=7` → `local:deterministic-solver`
- supported quadratic graphs → `local:deterministic-solver`
- geometry proofs → `openai/gpt-oss-120b`
- proof-style calculus → `openai/gpt-oss-120b`
- image input → OCR first, then local deterministic solving when supported, otherwise normal model routing

### 3. Catalog-driven GeoGebra DSL
The model is not allowed to emit arbitrary GeoGebra syntax. DSL `1.1` is the
only accepted visualization format and includes both the existing high-level
actions and the typed generic `EXECUTE_COMMAND` action.

Instead, visualization intent is represented as structured actions such as:
- `CREATE_POINT`
- `CREATE_LINE`
- `CREATE_CIRCLE`
- `CREATE_POLYGON`
- `INTERSECT`
- `MIDPOINT`
- `PERPENDICULAR`
- `PARALLEL`
- `ANGLE_BISECTOR`
- `CREATE_FUNCTION`
- `EXECUTE_COMMAND`

Generic command arguments use discriminated kinds: `reference`, `number`,
`angle`, `point`, `vector`, `text`, `boolean`, `expression`, `equation`, `list`,
and `interval`. IntoMath classifies the visualization environment, retrieves a
small relevant command set from the local registry, validates signatures,
types and dependencies, and only then translates it. The full catalog is never
placed in a model prompt.

The production rollout currently covers the existing high-level actions plus
the core 2D generic set (points/lines/circles/conics, intersections,
transformations, tangents, and common measurements). Catalog presence alone is
not a support claim. Graphing retains the deterministic `CREATE_FUNCTION` path;
advanced geometry, calculus, statistics/probability, 3D, CAS, and spreadsheet
families remain gated pending family-specific signature and web-runtime tests.

### 4. Deterministic translation
`backend/app/services/geogebra_translator.py` converts DSL actions into GeoGebra commands in code.

Example:

```json
{
  "action": "CREATE_CIRCLE",
  "label": "c",
  "center": "O",
  "radius": 5
}
```

becomes:

```text
c = Circle(O, 5.0)
```

This keeps constructions stable and auditable. The browser executes commands
sequentially, treats GeoGebra's boolean `evalCommand` result as authoritative,
stops on failure, restores the pre-run XML snapshot when possible, and shows
the failed command. Browser JavaScript execution is disabled.

### GeoGebra command source

GeoGebra command definitions are derived from the official GeoGebra manual.
The manual is not vendored in this repository. See:
https://github.com/geogebra/manual/tree/main/en/modules/ROOT/pages/commands

`backend/geogebra_commands.json` is a pinned, development-time generated
artifact (currently 502 command names and 1,052 overloads). Runtime startup and
requests use only this local registry and never fetch GitHub. Regeneration is
documented in `docs/SETUP.md`.

## Database tables

Current SQLAlchemy models:
- `problem_attempts`
- `solver_runs`
- `visualization_artifacts`

These store normalized prompts, routing decisions, confidence, and generated visualization payloads.

## Local development

See `docs/SETUP.md` for full instructions.

Quick start:

### Frontend
```bash
bun install --cwd frontend
bun run --cwd frontend dev
```

The frontend uses Bun as its package manager. `frontend/bun.lock` is the canonical lockfile.

### Backend
```bash
python3 -m venv .venv-local
.venv-local/bin/pip install -r backend/requirements.txt
.venv-local/bin/uvicorn app.main:app --app-dir backend --reload
```

Frontend default URL: `http://localhost:3000`

Backend default URL: `http://localhost:8000`

## Environment variables

### Frontend
- `NEXT_PUBLIC_API_URL` — defaults to `http://localhost:8000/api/v1`

### Backend
- `APP_NAME`
- `APP_ENV`
- `APP_DEBUG`
- `REMOTE_MODEL_ATTEMPT_TIMEOUT_SECONDS` — defaults to `25.0`; hard per-attempt remote timeout
- `NVIDIA_LARGE_MODEL_ATTEMPT_TIMEOUT_SECONDS` — defaults to `50.0`; applies only to direct gpt-oss-120b cold starts
- `NVIDIA_API_KEY` — NVIDIA NIM API key for remote model-backed solving
- `NVIDIA_DIRECT_ENABLED` — defaults to `true`; active when `NVIDIA_API_KEY` is configured
- `NVIDIA_BASE_URL` — defaults to `https://integrate.api.nvidia.com/v1`
- `DATABASE_URL`
- `CORS_ORIGINS`
- `LOCAL_SOLVER_FIRST` — defaults to `true`; tries deterministic solving before model-backed solving
- `LOCAL_LLAMA_ENABLED` — defaults to `true`; master switch for the local llama.cpp integration
- `LOCAL_SOLVER_LLAMA_DETECTION_ENABLED` — defaults to `true`; asks local llama-server to detect/normalize supported local-solver prompts when direct deterministic matching fails
- `LOCAL_SOLVER_LLAMA_TRIVIA_ENABLED` — defaults to `true`; enables the local concept/trivia fallback
- `LOCAL_LLAMA_GEOMETRY_EXTRACTION_ENABLED` — defaults to `true`; uses the local model to produce validated visualization DSL for local solve routes
- `LOCAL_SOLVER_LLAMA_BASE_URL` — defaults to `http://localhost:8080`
- `LOCAL_SOLVER_LLAMA_MODEL` — defaults to `unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL`
- `LOCAL_SOLVER_LLAMA_TIMEOUT_SECONDS` — defaults to `20.0`
- `LOCAL_LLAMA_STARTUP_PROBE_TIMEOUT_SECONDS` — defaults to `1.0`; bounds the startup `/health` probe
- `LOCAL_LLAMA_UNAVAILABLE_COOLDOWN_SECONDS` — defaults to `60.0`; skips repeated dead local hops after a connectivity failure
- `LOCAL_LLAMA_GEOMETRY_TIMEOUT_SECONDS` — defaults to `30.0`
- `LOCAL_LLAMA_GEOMETRY_MAX_TOKENS` — defaults to `1200`

## Validation

Recommended checks:
- `python3 -m compileall backend/app backend/tests`
- `.venv-local/bin/pytest backend/tests`
- `bun run --cwd frontend typecheck`
- `bun test --cwd frontend`
- `bun run --cwd frontend build`

The frontend commands require Bun and installed frontend dependencies.

## Additional docs

- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/SETUP.md`
