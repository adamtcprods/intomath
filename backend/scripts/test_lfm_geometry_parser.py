from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.integrations.llama_client import LlamaClient  # noqa: E402
from app.services.geogebra_translator import GeoGebraTranslator  # noqa: E402
from app.services.geometry_extractor import GeometryExtractor  # noqa: E402
from app.services.model_router import ModelRouter  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_BINARY = (
    PROJECT_DIR / ".tools" / "llama.cpp" / "llama-b9987" / "llama-server"
)
DEFAULT_HF_REPO = "unsloth/LFM2.5-8B-A1B-GGUF"
DEFAULT_HF_FILE = "LFM2.5-8B-A1B-UD-Q4_K_XL.gguf"

SMOKE_TEST_PROMPTS = [
    "Make ABC a triangle and put M halfway between A and B.",
    "Draw a circle with center O at (0, 0) and radius 5.",
    "Graph y = x^2 - 4x + 3.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a bounded LFM llama-server and test IntoMath geometry extraction."
    )
    parser.add_argument("--server-binary", type=Path, default=DEFAULT_SERVER_BINARY)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    return parser.parse_args()


def wait_until_healthy(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{base_url}/health"

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"llama-server exited during startup with code {exit_code}.")
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)

    raise TimeoutError(f"llama-server did not become healthy within {timeout:.0f}s.")


async def run_smoke_tests(
    base_url: str, request_timeout: float, model_name: str
) -> None:
    client = LlamaClient()
    client.settings.local_llama_enabled = True
    client.settings.local_solver_llama_base_url = base_url
    client.settings.local_solver_llama_model = model_name
    client.settings.local_llama_geometry_extraction_enabled = True
    client.settings.local_llama_geometry_timeout_seconds = request_timeout
    client.settings.local_llama_geometry_max_tokens = 1_200

    extractor = GeometryExtractor(client, settings=client.settings)
    router = ModelRouter(client)
    translator = GeoGebraTranslator()

    failures: list[str] = []

    for index, prompt in enumerate(SMOKE_TEST_PROMPTS, start=1):
        started_at = time.monotonic()
        routing = await router.route_async(prompt, has_image=False)
        environment = routing.visualization_environment
        if environment is None:
            failure = "Tiny router selected no visualization environment."
            failures.append(f"{prompt}: {failure}")
            print(f"\n[{index}/{len(SMOKE_TEST_PROMPTS)}] {prompt}")
            print(f"FAILED: {failure}")
            continue

        result = await extractor.extract(
            prompt,
            "local:llama-geometry-parser",
            environment=environment,
            semantic_query_terms=routing.visualization_search_terms,
        )
        elapsed = time.monotonic() - started_at
        translation = translator.translate(
            result.dsl, allowed_command_names=result.allowed_commands
        )

        print(f"\n[{index}/{len(SMOKE_TEST_PROMPTS)}] {prompt}")
        print(f"Latency: {elapsed:.2f}s")
        print(f"Environment: {environment.value}")
        print(f"Search terms: {', '.join(routing.visualization_search_terms)}")
        print(f"Summary: {result.summary or '(none)'}")
        print("DSL:")
        print(json.dumps(result.dsl.model_dump(mode="json"), indent=2))
        print("GeoGebra commands:")
        for command in translation.commands:
            print(f"  {command}")

        errors = [
            issue.message
            for issue in translation.issues
            if issue.severity.value == "error"
        ]
        if not result.dsl.actions:
            errors.extend(result.warnings or ["Model returned no visualization actions."])
        if errors:
            failures.append(f"{prompt}: {'; '.join(errors)}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(SMOKE_TEST_PROMPTS)} prompts failed:\n- "
            + "\n- ".join(failures)
        )


def print_log_tail(log_path: Path, line_count: int = 80) -> None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if lines:
        print("\nllama-server log tail:", file=sys.stderr)
        print("\n".join(lines[-line_count:]), file=sys.stderr)


def main() -> int:
    args = parse_args()
    server_binary = args.server_binary.resolve()
    if not server_binary.is_file():
        print(f"llama-server binary not found: {server_binary}", file=sys.stderr)
        return 2

    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(server_binary),
        "--hf-repo",
        args.hf_repo,
        "--hf-file",
        args.hf_file,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        "4096",
        "--threads",
        "6",
        "--threads-batch",
        "12",
        "--parallel",
        "1",
        "--reasoning",
        "on",
        "--reasoning-budget",
        "64",
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="intomath-lfm-", suffix=".log", delete=False
    ) as log_file:
        log_path = Path(log_file.name)
        process = subprocess.Popen(
            command,
            cwd=server_binary.parent,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        print(f"Starting llama-server with {args.hf_repo}/{args.hf_file}...")
        wait_until_healthy(base_url, process, args.startup_timeout)
        print(f"llama-server is healthy at {base_url}.")
        asyncio.run(run_smoke_tests(base_url, args.request_timeout, args.hf_repo))
        print("\nAll live LFM geometry smoke tests passed.")
        return 0
    except Exception as exc:
        print(f"\nSmoke test failed: {exc}", file=sys.stderr)
        print_log_tail(log_path)
        return 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
