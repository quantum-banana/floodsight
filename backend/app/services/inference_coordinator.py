import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.inference.pipeline import InferencePipeline
from app.schemas.ingestion import FrameIntelligence, FrameMetadata

IntelligenceCallback = Callable[[FrameIntelligence], Awaitable[None]]


@dataclass(slots=True)
class _PendingFrame:
    frame: NDArray[np.uint8]
    metadata: FrameMetadata
    callback: IntelligenceCallback


class InferenceCoordinator:
    """One bounded latest-frame worker per session; ingestion acknowledgements never wait."""

    def __init__(self, pipeline: InferencePipeline) -> None:
        self.pipeline = pipeline
        self._pending: dict[str, _PendingFrame] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._latest: dict[str, FrameIntelligence] = {}
        self._sequence: dict[str, int] = {}

    def submit(
        self,
        *,
        session_id: str,
        frame: NDArray[np.uint8],
        metadata: FrameMetadata,
        callback: IntelligenceCallback,
    ) -> bool:
        if not self.pipeline.should_process(metadata.frame_id):
            return False
        replaced = session_id in self._pending
        self._pending[session_id] = _PendingFrame(frame=frame, metadata=metadata, callback=callback)
        worker = self._workers.get(session_id)
        if worker is None or worker.done():
            self._workers[session_id] = asyncio.create_task(self._run(session_id))
        return not replaced

    async def _run(self, session_id: str) -> None:
        try:
            while pending := self._pending.pop(session_id, None):
                result = await asyncio.to_thread(
                    self.pipeline.process,
                    session_id=session_id,
                    frame_bgr=pending.frame,
                    frame_id=pending.metadata.frame_id,
                    timestamp_ms=pending.metadata.captured_at_ms,
                    source_mode=pending.metadata.source_mode,
                )
                if result is None:
                    continue
                sequence = self._sequence.get(session_id, 0)
                self._sequence[session_id] = sequence + 1
                message = FrameIntelligence(
                    session_id=session_id,
                    frame_id=pending.metadata.frame_id,
                    sequence=sequence,
                    result=result,
                )
                self._latest[session_id] = message
                try:
                    await pending.callback(message)
                except (RuntimeError, ConnectionError):
                    break
        finally:
            self._workers.pop(session_id, None)

    def latest(self, session_id: str) -> FrameIntelligence | None:
        return self._latest.get(session_id)

    def close(self, session_id: str) -> None:
        self.disconnect(session_id)
        self._latest.pop(session_id, None)
        self._sequence.pop(session_id, None)
        self.pipeline.close_session(session_id)

    def disconnect(self, session_id: str) -> None:
        """Stop transient work but retain the latest reportable session intelligence."""
        self._pending.pop(session_id, None)
        task = self._workers.pop(session_id, None)
        if task is not None:
            task.cancel()
