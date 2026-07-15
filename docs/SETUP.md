# IntoMath 2.0 Setup

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
APP_NAME=IntoMath 2.0 API
APP_ENV=development
APP_DEBUG=true
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_NAME=IntoMath 2.0
OPENROUTER_SITE_URL=http://localhost:3000
DATABASE_URL=sqlite:///./intomath.db
CORS_ORIGINS=http://localhost:3000
LOCAL_SOLVER_FIRST=true
LOCAL_LLAMA_ENABLED=true
LOCAL_SOLVER_LLAMA_DETECTION_ENABLED=true
LOCAL_SOLVER_LLAMA_TRIVIA_ENABLED=true
LOCAL_LLAMA_GEOMETRY_EXTRACTION_ENABLED=true
LOCAL_SOLVER_LLAMA_BASE_URL=http://localhost:8080
LOCAL_SOLVER_LLAMA_MODEL=unsloth/LFM2.5-8B-A1B-GGUF
LOCAL_SOLVER_LLAMA_TIMEOUT_SECONDS=20.0
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

- supported deterministic solving first: `local:deterministic-solver`
- easy solving via OpenRouter: `nvidia/nemotron-3-nano-30b-a3b:free`
- hard solving via OpenRouter: `nvidia/nemotron-3-super-120b-a12b:free`
- OCR / vision locally: `deepseek-ai/deepseek-ocr-2`

For local-first routing, normalization, trivia fallback, and visualization DSL extraction, run the local model through llama-server:

```bash
./llama-server \
  --hf-repo unsloth/LFM2.5-8B-A1B-GGUF \
  --hf-file LFM2.5-8B-A1B-UD-Q4_K_XL.gguf \
  --ctx-size 4096 \
  --threads 6 \
  --threads-batch 12 \
  --parallel 1 \
  --reasoning on \
  --reasoning-budget 64
```

The model proposes routing decisions and visualization DSL, but deterministic code remains the authority: the solver rejects unsupported normalization hints, and the geometry extractor uses constrained JSON decoding plus validation of labels, dependencies, requested intent, numeric values, expressions, and action count before producing GeoGebra commands. Invented point coordinates are removed unless they are explicit in the prompt, and conservative normalization repairs unambiguous triangle/circle labeling. If local extraction is unavailable, semantically mismatched, or invalid, a small deterministic parser handles only basic constructions.

`LFM2.5-8B-A1B` is reasoning-tuned. Keep `--reasoning-budget 64` so it reaches the final structured response promptly; unrestricted reasoning exhausted a 1,200-token response budget in live testing. On an Intel i5-12400 CPU, the tested `UD-Q4_K_XL` quantization produced approximately 22–24 output tokens/second, with simple geometry extraction taking about 14–18 seconds. The model uses several GiB of RAM, so at least 8–10 GiB of available memory is recommended.

The current code routes solver requests in `backend/app/services/solver_service.py`, `backend/app/services/local_solver_selector.py`, and `backend/app/services/model_router.py`, and uses local DeepSeek OCR for image extraction in `backend/app/services/ocr_service.py`.

To run the bounded live smoke test (it starts and always terminates its own server):

```bash
PYTHONPATH=backend .venv-local/bin/python backend/scripts/test_lfm_geometry_parser.py
```

## Local validation commands

### Frontend
```bash
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

### Backend
```bash
python3 -m compileall backend/app backend/tests
.venv-local/bin/pytest backend/tests
```

## Notes about local development

- The frontend uses Bun as its package manager. Keep `frontend/bun.lock` committed and do not regenerate `package-lock.json`.
- The backend tries deterministic local solving for supported typed prompts before model-backed solving, even when `OPENROUTER_API_KEY` is configured.
- GeoGebra is loaded lazily in the browser from the GeoGebra deployment script.
- The translator validates emitted command names against `backend/geogebra_commands.json`.
