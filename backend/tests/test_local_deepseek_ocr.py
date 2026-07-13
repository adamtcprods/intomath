import asyncio
import base64
import threading
from pathlib import Path

import pytest

from app.integrations import local_deepseek_ocr as ocr_module
from app.integrations.local_deepseek_ocr import LocalDeepSeekOCR


def test_cancellation_waits_for_ocr_worker_before_deleting_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "problem.png"
    image_path.write_bytes(b"temporary image")
    started = threading.Event()
    release = threading.Event()
    runner = LocalDeepSeekOCR()

    monkeypatch.setattr(
        ocr_module,
        "_write_temp_image",
        lambda image_bytes, mime_type: str(image_path),
    )

    def blocking_extract(path: str, prompt: str) -> str:
        assert path == str(image_path)
        assert image_path.exists()
        started.set()
        release.wait(timeout=5)
        assert image_path.exists()
        return "x + 1"

    monkeypatch.setattr(runner, "_extract_text_sync", blocking_extract)

    async def run_scenario() -> None:
        task = asyncio.create_task(
            runner.extract_text(
                image_base64=base64.b64encode(b"image bytes").decode("ascii")
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)

            task.cancel()
            await asyncio.sleep(0.05)

            assert image_path.exists()
            assert not task.done()
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert not image_path.exists()

    asyncio.run(run_scenario())


def test_parent_cancellation_wins_if_ocr_worker_later_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "problem.png"
    image_path.write_bytes(b"temporary image")
    started = threading.Event()
    release = threading.Event()
    runner = LocalDeepSeekOCR()

    monkeypatch.setattr(
        ocr_module,
        "_write_temp_image",
        lambda image_bytes, mime_type: str(image_path),
    )

    def failing_extract(path: str, prompt: str) -> str:
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("worker failed after cancellation")

    monkeypatch.setattr(runner, "_extract_text_sync", failing_extract)

    async def run_scenario() -> None:
        task = asyncio.create_task(
            runner.extract_text(
                image_base64=base64.b64encode(b"image bytes").decode("ascii")
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.05)
            assert image_path.exists()
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert not image_path.exists()

    asyncio.run(run_scenario())
