from __future__ import annotations

import asyncio

from core.embeddings import EmbeddingEngine


class SlowRemoteClient:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_remote_embedding_requests_are_serialized() -> None:
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.client = SlowRemoteClient()
    engine.model_name = "bge-m3"
    engine.backend = "remote"
    engine.dimension = 0

    async def run() -> None:
        results = await asyncio.gather(
            engine.aget_embeddings(["first"]),
            engine.aget_embeddings(["second"]),
            engine.aget_embeddings(["third"]),
        )
        assert all(result and len(result[0]) == 2 for result in results)

    asyncio.run(run())
    assert engine.client.maximum_active == 1
