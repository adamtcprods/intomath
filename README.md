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

Image upload is available for configured OCR/model backends. For supported typed prompts, the backend now tries the local deterministic solver first, even when `OPENROUTER_API_KEY` is configured. It handles arithmetic, one-variable linear equations with basic parentheses/division, quadratic graphs, and simple geometry constructions. A local llama.cpp model can normalize borderline prompts and extract validated visualization DSL, while proof-style geometry stays on model-backed routing.

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
- OpenRouter API

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
- **Local visualization DSL extraction:** `local:llama-geometry-parser`, backed by `unsloth/LFM2.5-8B-A1B-GGUF` (`UD-Q4_K_XL`) through llama-server with JSON-schema constraints
- **Local routing and solver normalization:** the same llama.cpp model
- **Easy / lower-latency solving via OpenRouter:** `nvidia/nemotron-3-nano-30b-a3b:free`
- **Hard / proof-heavy solving via OpenRouter:** `nvidia/nemotron-3-super-120b-a12b:free`
- **JSON fallback routing via OpenRouter:** `nvidia/nemotron-3-nano-30b-a3b:free`, then `openrouter/free`
- **OCR / visual extraction locally:** `deepseek-ai/deepseek-ocr-2`

Examples:
- supported arithmetic → `local:deterministic-solver`
- supported linear equations, including `ax+b=cx+d`, `2(x+3)=14`, and `x/2+3=7` → `local:deterministic-solver`
- supported quadratic graphs → `local:deterministic-solver`
- geometry proofs → `nvidia/nemotron-3-super-120b-a12b:free`
- proof-style calculus → `nvidia/nemotron-3-super-120b-a12b:free`
- image input → OCR first, then local deterministic solving when supported, otherwise normal model routing

### 3. Geometry DSL
The model is not allowed to emit arbitrary GeoGebra syntax.

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

This keeps constructions stable and auditable.

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
- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`
- `OPENROUTER_APP_NAME`
- `OPENROUTER_SITE_URL`
- `DATABASE_URL`
- `CORS_ORIGINS`
- `LOCAL_SOLVER_FIRST` — defaults to `true`; tries deterministic solving before model-backed solving
- `LOCAL_LLAMA_ENABLED` — defaults to `true`; master switch for the local llama.cpp integration
- `LOCAL_SOLVER_LLAMA_DETECTION_ENABLED` — defaults to `true`; asks local llama-server to detect/normalize supported local-solver prompts when direct deterministic matching fails
- `LOCAL_SOLVER_LLAMA_TRIVIA_ENABLED` — defaults to `true`; enables the local concept/trivia fallback
- `LOCAL_LLAMA_GEOMETRY_EXTRACTION_ENABLED` — defaults to `true`; uses the local model to produce validated visualization DSL for local solve routes
- `LOCAL_SOLVER_LLAMA_BASE_URL` — defaults to `http://localhost:8080`
- `LOCAL_SOLVER_LLAMA_MODEL` — defaults to `unsloth/LFM2.5-8B-A1B-GGUF`
- `LOCAL_SOLVER_LLAMA_TIMEOUT_SECONDS` — defaults to `20.0`
- `LOCAL_LLAMA_GEOMETRY_TIMEOUT_SECONDS` — defaults to `30.0`
- `LOCAL_LLAMA_GEOMETRY_MAX_TOKENS` — defaults to `1200`

## Validation

Recommended checks:
- `python3 -m compileall backend/app backend/tests`
- `.venv-local/bin/pytest backend/tests`
- `bun run --cwd frontend typecheck`
- `bun run --cwd frontend build`

The frontend commands require Bun and installed frontend dependencies.

## Additional docs

- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/SETUP.md`
