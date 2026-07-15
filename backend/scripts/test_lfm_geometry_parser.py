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

from app.integrations.llama_client import LlamaClient
from app.integrations.openrouter_client import OpenRouterClient
from app.services.geogebra_translator import GeoGebraTranslator
from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import GeometryDSL
from app.services.geometry_extractor import (
    LOCAL_GEOMETRY_EXTRACTION_PROMPT,
    LOCAL_GEOMETRY_RESPONSE_SCHEMA,
    GeometryExtractor,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_BINARY = (
    PROJECT_DIR / ".tools" / "llama.cpp" / "llama-b9987" / "llama-server"
)
DEFAULT_HF_REPO = "unsloth/LFM2.5-8B-A1B-GGUF"
DEFAULT_HF_FILE = "LFM2.5-8B-A1B-UD-Q4_K_XL.gguf"

SMOKE_TEST_PROMPTS = [
    ("Make ABC a triangle and put M halfway between A and B.", ProblemType.geometry),
    ("Draw a circle with center O at (0, 0) and radius 5.", ProblemType.geometry),
    ("Graph y = x^2 - 4x + 3.", ProblemType.algebra),
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


    extractor = GeometryExtractor(OpenRouterClient(), client, client.settings)
    translator = GeoGebraTranslator()

    failures: list[str] = []

    for index, (prompt, problem_type) in enumerate(SMOKE_TEST_PROMPTS, start=1):
        started_at = time.monotonic()
        if problem_type is ProblemType.algebra:
            result = await extractor.extract(
                prompt, problem_type, "local:llama-geometry-parser"
            )
            translation = translator.translate(result.dsl)
            print(f"\n[{index}/{len(SMOKE_TEST_PROMPTS)}] {prompt}")
            print(f"Deterministic graph fast path latency: {time.monotonic() - started_at:.3f}s")
            print("GeoGebra commands:")
            for command in translation.commands:
                print(f"  {command}")
            if translation.issues:
                failures.append(
                    f"{prompt}: translation issues: {'; '.join(translation.issues)}"
                )
            continue

        payload = await client.generate_json(
            prompt=f"{LOCAL_GEOMETRY_EXTRACTION_PROMPT}\n\nProblem:\n{prompt}",
            max_tokens=client.settings.local_llama_geometry_max_tokens,
            timeout_seconds=request_timeout,
            json_schema=LOCAL_GEOMETRY_RESPONSE_SCHEMA,
        )
        elapsed = time.monotonic() - started_at

        print(f"\n[{index}/{len(SMOKE_TEST_PROMPTS)}] {prompt}")
        print(f"Latency: {elapsed:.2f}s")
        print("Raw model JSON:")
        print(json.dumps(payload, indent=2))

        dsl = GeometryDSL.model_validate(payload.get("dsl", {}))
        dsl = extractor._sanitize_local_dsl(prompt, dsl)
        issues = extractor._validate_local_dsl(dsl)
        issues.extend(extractor._validate_intent_alignment(prompt, dsl))
        if issues:
            repair_prompt = (
                f"{LOCAL_GEOMETRY_EXTRACTION_PROMPT}\n\n"
                f"Original problem:\n{prompt}\n\n"
                "Your previous response was invalid. Return a complete corrected response. "
                "Create every referenced point or object before using it.\n"
                f"Validation errors: {'; '.join(issues)}\n"
                f"Previous response:\n{json.dumps(payload)}"
            )
            repair_started_at = time.monotonic()
            payload = await client.generate_json(
                prompt=repair_prompt,
                max_tokens=client.settings.local_llama_geometry_max_tokens,
                timeout_seconds=request_timeout,
                json_schema=LOCAL_GEOMETRY_RESPONSE_SCHEMA,
            )
            print(f"Repair latency: {time.monotonic() - repair_started_at:.2f}s")
            print("Repaired model JSON:")
            print(json.dumps(payload, indent=2))
            dsl = GeometryDSL.model_validate(payload.get("dsl", {}))
            dsl = extractor._sanitize_local_dsl(prompt, dsl)
            issues = extractor._validate_local_dsl(dsl)
            issues.extend(extractor._validate_intent_alignment(prompt, dsl))
            if issues:
                failure = "Invalid repaired geometry DSL: " + "; ".join(issues)
                failures.append(f"{prompt}: {failure}")
                print(f"FAILED: {failure}")
                continue

        translation = translator.translate(dsl)
        print(f"Summary: {extractor._coerce_optional_string(payload.get('summary')) or '(none)'}")
        print("DSL:")
        print(json.dumps(dsl.model_dump(mode="json"), indent=2))
        print("GeoGebra commands:")
        for command in translation.commands:
            print(f"  {command}")
        if translation.issues:
            print("Translation issues:")
            for issue in translation.issues:
                print(f"  - {issue}")
            failures.append(
                f"{prompt}: translation issues: {'; '.join(translation.issues)}"
            )

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
