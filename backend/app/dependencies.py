"""Application and request dependency composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from app.db.session import SessionLocal
from app.integrations.llama_client import LlamaClient
from app.integrations.nvidia_client import NvidiaClient
from app.repositories.result_repository import ResultRepository
from app.services.solver_service import SolverService


@dataclass(frozen=True)
class SharedModelClients:
    """Process-local model clients owned by one FastAPI application lifespan."""

    nvidia: NvidiaClient
    llama: LlamaClient

    async def aclose(self) -> None:
        # Attempt both closes even if one integration reports a shutdown error.
        results = await asyncio.gather(
            self.nvidia.aclose(),
            self.llama.aclose(),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]


def create_shared_model_clients(settings: Any) -> SharedModelClients:
    """Create lifespan-owned integrations without import-time side effects."""
    return SharedModelClients(
        nvidia=NvidiaClient(settings),
        llama=LlamaClient(settings),
    )


def get_solver_service(
    request: Request,
) -> SolverService:
    """Combine shared model clients with stateless persistence dependencies."""
    clients: SharedModelClients = request.app.state.model_clients
    return SolverService(
        result_repository=ResultRepository(SessionLocal),
        nvidia_client=clients.nvidia,
        llama_client=clients.llama,
    )
