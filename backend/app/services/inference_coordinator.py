import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from anyio import BrokenResourceError, ClosedResourceError
from numpy.typing import NDArray
from starlette.websockets import WebSocketDisconnect

from app.inference.contracts import DetectorInferenceMode
from app.inference.pipeline import InferencePipeline
from app.schemas.ingestion import (
    FrameIntelligence,
    FrameMetadata,
    IngestionSessionState,
    VideoAnalysisComplete,
)
from app.services.video_analysis import VideoAnalysisAggregator

IntelligenceCallback = Callable[[FrameIntelligence], Awaitable[None]]


@dataclass(slots=True)
class _PendingFrame:
    frame: NDArray[np.uint8]
    metadata: FrameMetadata
    callback: IntelligenceCallback
    detector_mode: DetectorInferenceMode


class InferenceCoordinator:
    """One bounded latest-frame worker per session; ingestion acknowledgements never wait."""

    def __init__(self, pipeline: InferencePipeline) -> None:
        self.pipeline = pipeline
        self._pending: dict[str, _PendingFrame] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._latest: dict[str, FrameIntelligence] = {}
        self._sequence: dict[str, int] = {}
        self._aggregators: dict[str, VideoAnalysisAggregator] = {}
        self._completed: dict[str, VideoAnalysisComplete] = {}
        self._finalizing: set[str] = set()
        self._finalize_locks: dict[str, asyncio.Lock] = {}

    def submit(
        self,
        *,
        session_id: str,
        frame: NDArray[np.uint8],
        metadata: FrameMetadata,
        callback: IntelligenceCallback,
        detector_mode: DetectorInferenceMode,
    ) -> bool:
        if not self.pipeline.should_process(metadata.frame_id):
            return False
        if session_id in self._finalizing or session_id in self._completed:
            return False
        replaced = session_id in self._pending
        self._pending[session_id] = _PendingFrame(
            frame=frame,
            metadata=metadata,
            callback=callback,
            detector_mode=detector_mode,
        )
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
                    detector_mode=pending.detector_mode,
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
                self._aggregators.setdefault(
                    session_id, VideoAnalysisAggregator(session_id)
                ).add(result, media_time_ms=pending.metadata.media_time_ms)
                try:
                    await pending.callback(message)
                except (
                    RuntimeError,
                    ConnectionError,
                    BrokenResourceError,
                    ClosedResourceError,
                    WebSocketDisconnect,
                ):
                    # Transport loss must not discard already accepted video work.
                    # Continue aggregating the bounded pending frame; explicit close
                    # (DELETE/TTL expiry) remains the destructive cancellation path.
                    continue
        finally:
            self._workers.pop(session_id, None)

    def latest(self, session_id: str) -> FrameIntelligence | None:
        return self._latest.get(session_id)

    def completed(self, session_id: str) -> VideoAnalysisComplete | None:
        return self._completed.get(session_id)

    async def finalize(
        self,
        session_id: str,
        *,
        frames_accepted: int,
        frames_dropped: int,
    ) -> VideoAnalysisComplete:
        lock = self._finalize_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            completed = self._completed.get(session_id)
            if completed is not None:
                return completed
            self._finalizing.add(session_id)
            try:
                await self._drain(session_id)
                aggregator = self._aggregators.setdefault(
                    session_id, VideoAnalysisAggregator(session_id)
                )
                completed = VideoAnalysisComplete(
                    session_id=session_id,
                    state=IngestionSessionState.COMPLETE,
                    summary=aggregator.build_summary(
                        frames_accepted=frames_accepted,
                        frames_dropped=frames_dropped,
                        model_status=self.pipeline.status(),
                    ),
                    latest_result=aggregator.latest_result,
                )
                self._completed[session_id] = completed
                return completed
            finally:
                self._finalizing.discard(session_id)

    async def _drain(self, session_id: str) -> None:
        while True:
            task = self._workers.get(session_id)
            if task is None and session_id in self._pending:
                task = asyncio.create_task(self._run(session_id))
                self._workers[session_id] = task
            if task is None:
                return
            await asyncio.shield(task)

    def close(self, session_id: str) -> None:
        self._pending.pop(session_id, None)
        task = self._workers.pop(session_id, None)
        if task is not None:
            task.cancel()
        self._latest.pop(session_id, None)
        self._sequence.pop(session_id, None)
        self._aggregators.pop(session_id, None)
        self._completed.pop(session_id, None)
        self._finalizing.discard(session_id)
        self._finalize_locks.pop(session_id, None)
        self.pipeline.close_session(session_id)

    def disconnect(self, session_id: str) -> None:
        """Detach transport while bounded accepted work continues for later finalization."""
