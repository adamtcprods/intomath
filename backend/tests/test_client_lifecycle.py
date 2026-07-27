from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from starlette.requests import Request

import app.main as main_module
from app.dependencies import SharedModelClients, get_solver_service
from app.repositories.result_repository import ResultRepository


class FakeNvidiaClient:
    enabled = False

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeSemanticRouter:
    artifact_identity = "fake-semantic-artifact"

    def __init__(self) -> None:
        self.initialize_calls = 0

    async def initialize_async(self) -> SimpleNamespace:
        self.initialize_calls += 1
        return SimpleNamespace(
            enabled=True,
            state="ready",
            model_source="fake-embedding",
            error_category=None,
        )


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
    semantic_router = FakeSemanticRouter()
    shared = SharedModelClients(
        nvidia=nvidia,
        llama=llama,
        semantic_router=semantic_router,  # type: ignore[arg-type]
    )  # type: ignore[arg-type]
    monkeypatch.setattr(
        main_module,
        "create_shared_model_clients",
        lambda settings: shared,
    )

    with TestClient(main_module.app):
        assert main_module.app.state.model_clients is shared
        request = Request({"type": "http", "app": main_module.app})

        first_service = get_solver_service(request)
        second_service = get_solver_service(request)

        assert first_service is not second_service
        assert not hasattr(first_service, "db")
        assert not hasattr(second_service, "db")
        assert isinstance(first_service.result_repository, ResultRepository)
        assert isinstance(second_service.result_repository, ResultRepository)
        assert first_service.nvidia_client is second_service.nvidia_client is nvidia
        assert first_service.llama_client is second_service.llama_client is llama
        assert (
            first_service.semantic_router
            is second_service.semantic_router
            is semantic_router
        )
        assert semantic_router.initialize_calls == 1

        asyncio.run(first_service.aclose())
        assert nvidia.closed is False
        assert llama.closed is False

    assert nvidia.closed is True
    assert llama.closed is True
    assert not hasattr(main_module.app.state, "model_clients")
