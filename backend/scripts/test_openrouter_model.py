from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_PROMPT = (
    "Return a JSON object with keys answer and explanation for: What is 12 * 13?"
)


def _extract_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output")
    if not isinstance(output, list):
        raise ValueError("Response does not include output/output_text")

    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if isinstance(content, list):
            text_parts.extend(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    text = "".join(text_parts).strip()
    if not text:
        raise ValueError("Response output did not include text content")
    return text


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test an OpenRouter Responses API model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        print(
            "OPENROUTER_API_KEY is not configured. Add it to backend/.env or export it in your shell.",
            file=sys.stderr,
        )
        return 2

    payload = {
        "model": args.model,
        "temperature": 0.0,
        "instructions": "You are a concise math assistant. Return valid JSON only.",
        "input": args.prompt,
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": 2000,
        "store": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.openrouter_base_url}/responses",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
                "X-OpenRouter-Metadata": "enabled",
            },
            json=payload,
        )

    print(f"HTTP {response.status_code}")
    if response.status_code >= 400:
        print(response.text)
        return 1

    data = response.json()
    print(f"model: {data.get('model', args.model)}")
    if data.get("error"):
        print("OpenRouter returned an error:")
        print(json.dumps(data.get("error"), indent=2, ensure_ascii=False))
        return 1
    print(f"status: {data.get('status')}")

    try:
        text = _extract_text(data)
    except ValueError:
        print("unexpected response body:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 1
    print("content:")
    try:
        print(json.dumps(json.loads(text), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
