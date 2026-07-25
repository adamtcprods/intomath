from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient
from starlette.requests import Request

import app.main as main_module
from app.dependencies import SharedModelClients, get_solver_service


class FakeNvidiaClient:
    enabled = False

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeLlamaClient:
    enabled = False
    available = False
    model = "fake-llama"

    def __init__(self) -> None:
        self.closed = False
        self.probe_calls = 0

    async def probe_health(self, **_: Any) -> bool:
        self.probe_calls += 1
        return True

    async def aclose(self) -> None:
        self.closed = True


def test_lifespan_closes_shared_clients_and_request_services_only_borrow_them(
    monkeypatch: Any,
) -> None:
    nvidia = FakeNvidiaClient()
    llama = FakeLlamaClient()
    shared = SharedModelClients(nvidia=nvidia, llama=llama)  # type: ignore[arg-type]
    monkeypatch.setattr(
        main_module,
        "create_shared_model_clients",
        lambda settings: shared,
    )

    with TestClient(main_module.app):
        assert main_module.app.state.model_clients is shared
        request = Request({"type": "http", "app": main_module.app})
        first_db = object()
        second_db = object()

        first_service = get_solver_service(request, db=first_db)  # type: ignore[arg-type]
        second_service = get_solver_service(request, db=second_db)  # type: ignore[arg-type]

        assert first_service is not second_service
        assert first_service.db is first_db
        assert second_service.db is second_db
        assert first_service.nvidia_client is second_service.nvidia_client is nvidia
        assert first_service.llama_client is second_service.llama_client is llama

        asyncio.run(first_service.aclose())
        assert nvidia.closed is False
        assert llama.closed is False

    assert nvidia.closed is True
    assert llama.closed is True
    assert not hasattr(main_module.app.state, "model_clients")
