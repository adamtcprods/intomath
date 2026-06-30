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
LOCAL_SOLVER_LLAMA_DETECTION_ENABLED=true
LOCAL_SOLVER_LLAMA_BASE_URL=http://localhost:8080
LOCAL_SOLVER_LLAMA_MODEL=hf.co/unsloth/LiquidAI/LFM2.5-350M-GGUF
LOCAL_SOLVER_LLAMA_TIMEOUT_SECONDS=4.0
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

For stronger local-first routing, run a tiny Llama-server detector locally:

```bash
./llama-server -m hf.co/unsloth/LiquidAI/LFM2.5-350M-GGUF --chat-template-kwargs '{"enable_thinking":true}'
```

The detector only decides whether a prompt can be normalized into a deterministic-solver shape; the deterministic solver still produces the answer and rejects unsupported hints.

The current code routes solver requests in `backend/app/services/solver_service.py`, `backend/app/services/local_solver_selector.py`, and `backend/app/services/model_router.py`, and uses local DeepSeek OCR for image extraction in `backend/app/services/ocr_service.py`.

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
