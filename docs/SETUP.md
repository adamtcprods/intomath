# IntoMath Setup

## Prerequisites

- Bun 1.3+
- Python 3.12+
- PostgreSQL (optional for local dev; SQLite works by default)

## Frontend setup

From the repository root:

```bash
bun install --cwd frontend
bun run --cwd frontend dev
```

This starts the Next.js app on:

```text
http://localhost:3000
```

### Frontend environment

Create `frontend/.env.local` with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

## Backend setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv-local
.venv-local/bin/pip install -r backend/requirements.txt
```

### Geometry DSL `1.1` artifact migration

Before deploying the `1.1`-only visualization schema over an existing database,
inspect persisted DSL `1.0` artifacts from the `backend` directory:

```bash
python scripts/migrate_geometry_dsl_1_1.py
```

The command is a dry run by default. Apply validated upgrades with:

```bash
python scripts/migrate_geometry_dsl_1_1.py --apply
```

Use `--database-url` to override `DATABASE_URL` and `--failure-report PATH` to
export malformed artifact IDs and validation errors. The migration preserves
stored GeoGebra commands and leaves invalid legacy DSL records unchanged.

Run the API:

```bash
.venv-local/bin/uvicorn app.main:app --app-dir backend --reload
```

This starts FastAPI on:

```text
http://localhost:8000
```

## Backend environment

Create `backend/.env` with values like:

```env
APP_NAME=IntoMath API
APP_ENV=development
APP_DEBUG=true
REMOTE_MODEL_ATTEMPT_TIMEOUT_SECONDS=25.0
NVIDIA_LARGE_MODEL_ATTEMPT_TIMEOUT_SECONDS=50.0
NVIDIA_API_KEY=
NVIDIA_DIRECT_ENABLED=true
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
DATABASE_URL=sqlite:///./intomath.db
CORS_ORIGINS=http://localhost:3000
LOCAL_SOLVER_FIRST=true
LOCAL_LLAMA_ENABLED=true
LOCAL_SOLVER_LLAMA_DETECTION_ENABLED=true
LOCAL_SOLVER_LLAMA_TRIVIA_ENABLED=true
LOCAL_LLAMA_GEOMETRY_EXTRACTION_ENABLED=true
LOCAL_SOLVER_LLAMA_BASE_URL=http://localhost:8080
LOCAL_SOLVER_LLAMA_MODEL=unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL
LOCAL_SOLVER_LLAMA_TIMEOUT_SECONDS=20.0
LOCAL_LLAMA_STARTUP_PROBE_TIMEOUT_SECONDS=1.0
LOCAL_LLAMA_UNAVAILABLE_COOLDOWN_SECONDS=60.0
LOCAL_LLAMA_GEOMETRY_TIMEOUT_SECONDS=30.0
LOCAL_LLAMA_GEOMETRY_MAX_TOKENS=1200
```

### PostgreSQL option

For PostgreSQL, set `DATABASE_URL` to something like:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/intomath
```

## Solver model configuration

IntoMath expects the following model policy:

- AI-selected deterministic arithmetic/algebra execution: `local:deterministic-solver`
- easy solving via NVIDIA NIM: `openai/gpt-oss-20b`
- hard solving via NVIDIA NIM: `openai/gpt-oss-120b`
- NVIDIA NIM fallback order: gpt-oss-120b, gpt-oss-20b
- OCR / vision locally: `deepseek-ai/deepseek-ocr-2`

For local-first routing, normalization, trivia fallback, and visualization DSL extraction, run the local model through llama-server:

```bash
./llama-server \
  --hf-repo unsloth/LFM2.5-8B-A1B-GGUF \
  --hf-file LFM2.5-8B-A1B-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --threads 6 \
  --threads-batch 12 \
  --parallel 1 \
  --reasoning auto
```

`LOCAL_LLAMA_ENABLED=true` does not start this process. Confirm the operational
dependency before running the API:

```bash
curl --fail http://localhost:8080/health
curl --fail http://localhost:8080/v1/models
```

Set `LOCAL_SOLVER_LLAMA_MODEL` to the exact `id` reported by `/v1/models`. For the
installed `LFM2.5-8B-A1B` Q4_K_XL model, use
`unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL`; the repository name without `:Q4_K_XL` is not
a valid OpenAI-compatible model ID.

A connection-refused result means local routing and local geometry extraction are
unavailable until `llama-server` is started. API startup now probes this health URL and
logs the full causal exception chain (including `Connection refused`), HTTP status, and
bounded response body. A failed probe opens a process-wide 60-second circuit keyed by
URL and model, so classification, trivia, and geometry do not repeat the same dead hop.
Start the server before the API, restart the API after starting it, or wait for the
cooldown to expire; a successful request closes the circuit. Geometry continues through
the validated NVIDIA direct model fallbacks while the circuit is open.

The model proposes problem type, difficulty, visualization environment, and DSL;
there are no object-name branches that decide 2D versus 3D. Deterministic code
remains the trust boundary: it rejects unsupported normalization hints and validates
labels, dependencies, retrieved command membership, overload argument types,
environment compatibility, numeric values, expressions, and action count before
producing GeoGebra commands. Explicit visualization requests may use neutral finite
placement/scale defaults solely to make an under-specified object visible. If every
model-backed extractor is unavailable or invalid, no visualization is generated.

`LFM2.5-8B-A1B` is reasoning-tuned. Leave the server reasoning budget unrestricted
so IntoMath can set `thinking_budget_tokens` per request; routing and bounded DSL
extraction use zero hidden-reasoning tokens with strict output schemas. The model
uses several GiB of RAM, so at least 8–10 GiB of available memory is recommended.

The current code routes solver requests in `backend/app/services/solver_service.py`, `backend/app/services/local_solver_selector.py`, and `backend/app/services/model_router.py`, and uses local DeepSeek OCR for image extraction in `backend/app/services/ocr_service.py`.

NVIDIA NIM request contracts omit `response_format`, so structured solve and geometry
requests supply their schemas in the prompt and validate every response locally.
Geometry goes directly to the local llama.cpp parser, then the ordered NVIDIA list.
Invalid model output has different warnings and follows the same model-backed fallback
chain; no regex construction parser replaces it.

Structured solve candidates are explicit and ordered:

1. `openai/gpt-oss-120b`
2. `openai/gpt-oss-20b`

Geometry uses the same two-entry NVIDIA order after its local fallback. `REMOTE_MODEL_ATTEMPT_TIMEOUT_SECONDS=25.0` is
the hard budget for ordinary remote attempts.
`NVIDIA_LARGE_MODEL_ATTEMPT_TIMEOUT_SECONDS=50.0` applies only to direct gpt-oss-120b
to allow for free-tier cold starts; gpt-oss-20b remains at 25s.
HTTP 429s are not hidden or retried inside an opaque SDK: logs include attempt number,
model, provider, operation, request ID, status/body, and `failure_code=rate_limited`.
Ordering should only change after those logs provide representative rate-limit data.

Set `NVIDIA_API_KEY` to a key from build.nvidia.com. `NVIDIA_DIRECT_ENABLED=true`
enables remote solving when the key is present. The published
`ChatRequest` schemas on both configured model pages are closed with
`additionalProperties: false` and omit OpenAI's `response_format` field. The gpt-oss
cards advertise Structured Output as a model capability, but NVIDIA's hosted Chat
Completions contract still does not expose `json_schema`; the client therefore does not
send an unsupported field or claim enforcement. Calls are complete and non-streaming.
gpt-oss uses `reasoning_effort=low`. Every response remains an unenforced proposal subject to the
same strict local payload parser, deterministic geometry validator, and bounded repair
loop. The local parser also records empty step-level `latex`, warns when math notation
has no matching KaTeX expression, and sends scratch-work-style explanation fields
through at most one cleanup turn. Set `NVIDIA_DIRECT_ENABLED=false` to disable this
fallback without removing the key.

To run the bounded live smoke test (it starts and always terminates its own server):

```bash
PYTHONPATH=backend .venv-local/bin/python backend/scripts/test_lfm_geometry_parser.py
```

## Local validation commands

### Frontend
```bash
bun test --cwd frontend
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

### Backend
```bash
python3 -m compileall backend/app backend/tests
.venv-local/bin/pytest backend/tests
PYTHONPATH=backend .venv-local/bin/python backend/scripts/evaluate_geogebra_fixtures.py
```

## Regenerating the GeoGebra command catalog

GeoGebra command definitions are derived from the official GeoGebra manual.
The manual is not vendored in IntoMath. The upstream command pages are:
https://github.com/geogebra/manual/tree/main/en/modules/ROOT/pages/commands

The running API never downloads or scrapes this source. Regeneration is an
explicit development operation. From an existing local checkout:

```bash
PYTHONPATH=backend .venv-local/bin/python \
  backend/scripts/generate_geogebra_catalog.py \
  --manual-path /path/to/geogebra-manual \
  --repository https://github.com/geogebra/manual \
  --output backend/geogebra_commands.json
```

Or explicitly fetch a pinned ref into a temporary checkout:

```bash
PYTHONPATH=backend .venv-local/bin/python \
  backend/scripts/generate_geogebra_catalog.py \
  --repository https://github.com/geogebra/manual \
  --ref 83d180f2469c47f13f6884fce3cc1926b58c2681 \
  --output backend/geogebra_commands.json
```

The script accepts `--manual-path`, `--repository`, `--ref`, `--commit`, and
`--output`. It parses every page containing one or more formal command
signatures, preserves categories from upstream category pages, and maps those
overlapping documentation categories into stable product-facing `families`.
Each overload can belong to multiple families; for example, a command listed by
both Geometry and 3D receives `geometry_2d` and `graphics_3d`. The generator
also records command-name and overload counts per family in catalog metadata.
It attaches source path/repository/commit to every overload, validates a
temporary registry, and atomically replaces the catalog only after validation.
Catalog JSON is stably sorted and deterministic for a given commit. The sidecar
`backend/geogebra_commands.json.metadata.json` records the generation timestamp,
content SHA-256, generator/schema versions, source commit, pages, names, and
overload, family, acceptance-status, and runtime-eligibility counts. Catalog
schema `1.4` adds generated environment capabilities and acceptance-backed output
metadata alongside the exact
`support_status`, `support_requirements`, and
`runtime_accepted_environments` plus `runtime_eligible` to every overload.
Fixture tests do not require network access.

Safe catalog commands are runtime-eligible automatically; catalog membership
never overrides the permanent denylist. Exact overload acceptance records live
in `backend/geogebra_runtime_acceptance.json`. The registry derives acceptance
status independently and refuses stale catalog metadata. To promote an
experimental overload to accepted:

1. verify that the manual signature normalizes without uncertainty;
2. map every argument to a representable typed DSL value;
3. add the known output type and correct environment metadata;
4. add serializer, invalid-input, dependency, and bounded-retrieval fixtures;
5. run the exact translated fixture against the browser applet; and
6. record the accepted signature, environment, output type/strategy, and fixture ID in the runtime
   acceptance manifest, then regenerate the catalog.

The planned order is transformations and measurements; advanced graphing and
calculus; statistics and probability; 3D solids and surfaces; CAS; then
spreadsheet. Scripting, network/file behavior, and unsafe state mutation remain
permanently blocked rather than entering this rollout.

## GeoGebra browser validation

Unit tests cover sequential success, `evalCommand=false`, exceptions,
stop-on-failure, XML restore/fresh-construction fallback, documented perspective
selection, styling, viewport, and interaction API calls. A real web applet still
requires browser network access to GeoGebra's deployment script. Runtime
acceptance is the final syntax check, but it is not evidence that a construction
is mathematically correct.

The checked-in acceptance manifest is consumed as prior browser evidence. The
offline fixture evaluator verifies that every record still points to a reviewed
DSL fixture and exact catalog overload; it deliberately reports that it did not
re-run the live browser applet during an offline test run.

Raw AI-generated command mode is not available. If a trusted-user diagnostic
mode is ever added, it must be disabled by default, count/length limited,
single-command only, catalog and environment checked, scripting/JavaScript
denied, and executed one command at a time with boolean result inspection.

## Notes about local development

- The frontend uses Bun as its package manager. Keep `frontend/bun.lock` committed and do not regenerate `package-lock.json`.
- The backend runs deterministic arithmetic/algebra only after the tiny model selects and normalizes that execution path.
- GeoGebra is loaded lazily in the browser from the GeoGebra deployment script.
- The registry validates selected names, signatures, approximate types,
  environment, rollout status, and dependencies against
  `backend/geogebra_commands.json` before the translator emits syntax.
