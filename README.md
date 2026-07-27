# IntoMath

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

Image upload uses local OCR. For text requests, the exact parser gets the first
execution attempt. Unsupported prompts then pass through an optional multilingual
sentence-embedding router for independent problem-type, difficulty, and
visualization classification. The local llama.cpp classifier is retained only as
an uncertainty or availability fallback. Embedding labels never authorize
execution; deterministic solvers still re-parse the original prompt. The local
llama.cpp model also extracts validated visualization DSL, while proof-style
geometry can use NVIDIA NIM routing.

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
Routing behavior lives in `backend/app/services/model_router.py`; shared model
names, endpoint order, timeout policy, and failure labels live in the neutral
`backend/app/core/model_policy.py` module.

Current models and local routes:
- **Exact deterministic execution:** `local:deterministic-solver`, entered only after the original prompt passes the exact parser
- **Primary semantic routing:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, loaded once per API worker from local files and scored against independent multilingual prototype banks
- **Uncertain semantic-routing fallback:** `unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL` through llama-server with a constrained problem-type/difficulty/visualization schema
- **Local visualization DSL extraction:** the same llama.cpp model through llama-server with JSON-schema constraints
- **Easy / lower-latency solving via NVIDIA NIM:** `openai/gpt-oss-20b`
- **Hard / proof-heavy solving via NVIDIA NIM:** `openai/gpt-oss-120b`
- **Explicit JSON fallback routing:** NVIDIA-hosted gpt-oss models with no opaque router alias
- **OCR / visual extraction locally:** `deepseek-ai/deepseek-ocr-2`

The embedding router supports English, Vietnamese, and mixed-language starter
examples. Its problem type, difficulty, and visualization scores have separate
confidence and margin gates. Uncertain difficulty safely defaults to `medium`
without forcing an LLM call; uncertain required routing axes, OOD inputs, an
unavailable local embedding model, and overlong inputs can use the retained LLM
classifier. If both classifiers fail, the response remains explicitly
unclassified. GeoGebra term retrieval is a separate nearest-example operation
and cannot turn a failed classification into an accepted one. The 216-row starter
corpus is AI-authored and every example is marked `needs_review`; its evaluation
results are development measurements, not production accuracy claims.

Visualization extraction is local-first whenever the llama.cpp parser is enabled,
healthy, and the prompt is within its 4,000-character limit. Local unavailability,
timeout, or invalid output falls back to NVIDIA gpt-oss-20b and then, only if policy
allows another source, gpt-oss-120b. A 429, timeout, connectivity failure, or rejected
proposal never repeats the same remote extraction. Valid-schema DSL failures may receive
one action-scoped repair; every provider/model/operation tuple is attempted at most once
and all attempts share the overall solve deadline.

Examples:
- exact arithmetic → `local:deterministic-solver` before semantic routing
- exactly supported linear equations, including `ax+b=cx+d`, `2(x+3)=14`, and `x/2+3=7` → `local:deterministic-solver`
- exactly supported quadratic graph analysis → `local:deterministic-solver`
- unsupported syntax → embedding classification, then structured solving or the LLM routing fallback when uncertain
- geometry proofs → `openai/gpt-oss-120b`
- proof-style calculus → `openai/gpt-oss-120b`
- image input → OCR first, then the same exact-first routing flow

### 3. Catalog-driven GeoGebra DSL
The model is not allowed to emit arbitrary GeoGebra syntax. DSL `1.1` is the
only accepted visualization format and includes both the existing high-level
actions and the typed generic `EXECUTE_COMMAND` action.

Visualization environments are selected by the semantic router (or its LLM
fallback), not by object-name or command-name branches. The selected environment
bounds catalog retrieval; the DSL model then chooses among only the relevant
retrieved commands.

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
- `DEFINE_OBJECT`
- `EXECUTE_COMMAND`

Generic command arguments use discriminated kinds: `reference`, `number`,
`angle`, `point`, `vector`, `text`, `boolean`, `expression`, `equation`, `list`,
and `interval`. IntoMath classifies the visualization environment, retrieves
separate dense nearest-example GeoGebra search terms when available, retrieves at
most 10 relevant commands from the local registry, validates signatures,
types and dependencies, and only then translates it. The full catalog is never
placed in a model prompt.

`DEFINE_OBJECT` handles safe definitions that are not command calls, including
`f(x) = x^2`; it is the preferred path for new function definitions while
`CREATE_FUNCTION` remains compatible with persisted DSL 1.1 payloads.

Catalog presence alone is not an acceptance claim. Every exact overload is
marked `supported`, `experimental`, or `blocked`, and separately records
whether it is runtime-eligible. All 933 overloads across the 434 non-dangerous
command names can participate in bounded retrieval and typed translation in
their applicable environments. Experimental overloads emit a warning and rely
on GeoGebra's boolean runtime result plus construction rollback. The remaining
119 overloads across 68 scripting/state/network/media command names are
permanently blocked. Four exact overloads currently carry prior browser
acceptance evidence; that label no longer limits catalog utilization.

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
.venv-local/bin/pip install -e "backend[dev]"
cd backend
../.venv-local/bin/alembic upgrade head
../.venv-local/bin/uvicorn app.main:app --reload
```

Install `backend[semantic-router]` on workers that use embedding routing and
provision the configured model into the local Hugging Face cache (or set
`SEMANTIC_ROUTER_MODEL_PATH`). Normal tests, startup, and requests never download
model files. Fine-tuning is an explicit, optional `backend[semantic-router-train]`
development operation; generated weights under `backend/models/semantic_router/`
are ignored and never selected by runtime configuration automatically.

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
- `LOCAL_SOLVER_FIRST` — defaults to `true`; enables the exact deterministic attempt before semantic routing
- `LOCAL_LLAMA_ENABLED` — defaults to `true`; master switch for the local llama.cpp integration
- `LOCAL_SOLVER_LLAMA_DETECTION_ENABLED` — defaults to `true`; enables the retained LLM semantic-classification fallback
- `LOCAL_SOLVER_LLAMA_TRIVIA_ENABLED` — defaults to `true`; enables the local concept/trivia fallback
- `LOCAL_LLAMA_GEOMETRY_EXTRACTION_ENABLED` — defaults to `true`; makes the healthy local model the primary validated visualization DSL parser
- `LOCAL_SOLVER_LLAMA_BASE_URL` — defaults to `http://localhost:8080`
- `LOCAL_SOLVER_LLAMA_MODEL` — defaults to `unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL`
- `LOCAL_SOLVER_LLAMA_TIMEOUT_SECONDS` — defaults to `20.0`
- `LOCAL_LLAMA_STARTUP_PROBE_TIMEOUT_SECONDS` — defaults to `1.0`; bounds the startup `/health` probe
- `LOCAL_LLAMA_UNAVAILABLE_COOLDOWN_SECONDS` — defaults to `60.0`; skips repeated dead local hops after a connectivity failure
- `LOCAL_LLAMA_GEOMETRY_TIMEOUT_SECONDS` — defaults to `30.0`
- `LOCAL_LLAMA_GEOMETRY_MAX_TOKENS` — defaults to `1200`
- `SEMANTIC_ROUTER_ENABLED` — defaults to `true`; unavailable local files degrade to the LLM fallback
- `SEMANTIC_ROUTER_MODEL` — base model identifier; runtime loading is always local-files-only
- `SEMANTIC_ROUTER_MODEL_PATH` / `SEMANTIC_ROUTER_ARTIFACT_PATH` — optional local exported model and prototype paths
- `SEMANTIC_ROUTER_DEVICE` — defaults to `cpu`
- `SEMANTIC_ROUTER_MAX_TEXT_CHARS` — defaults to `4000`
- `SEMANTIC_ROUTER_MIN_CONFIDENCE`, `SEMANTIC_ROUTER_MIN_MARGIN`, and `SEMANTIC_ROUTER_MIN_RAW_SIMILARITY` — deployment floors; artifact recommendations can be stricter
- `SEMANTIC_ROUTER_TERM_MIN_SIMILARITY` — independent dense search-term retrieval floor
- `SEMANTIC_ROUTER_FALLBACK_TO_LLM` — defaults to `true`

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
