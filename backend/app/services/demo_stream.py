import asyncio
from collections.abc import AsyncIterator

from app.schemas.live_result import LiveResult
from app.services.demo_incident import get_demo_snapshots


async def stream_demo_snapshots(
    incident_id: str,
    *,
    start_index: int,
    interval_ms: int,
    loop: bool,
) -> AsyncIterator[LiveResult]:
    snapshots = get_demo_snapshots(incident_id)
    index = min(start_index, len(snapshots) - 1)

    while True:
        for snapshot in snapshots[index:]:
            yield snapshot
            if interval_ms:
                await asyncio.sleep(interval_ms / 1_000)
        if not loop:
            return
        index = 0
