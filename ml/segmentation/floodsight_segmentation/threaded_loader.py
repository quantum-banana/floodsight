"""Bounded threaded loading of individual segmentation samples.

Only the static, no-RNG ``read_pair`` operation runs in producer threads.
``transform_pair`` and collation run in sampler order on the consuming thread,
so stochastic transforms retain a single, checkpoint-restorable Torch RNG
stream.  Unlike a multiprocessing ``DataLoader``, this loader never transfers
whole batches through shared-memory IPC.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sized
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import Any, Protocol, TypeVar

from torch.utils.data import Sampler, default_collate
from torch.utils.data._utils.pin_memory import pin_memory as pin_memory_batch

ImageT = TypeVar("ImageT")
MaskT = TypeVar("MaskT")
SampleT = TypeVar("SampleT")


class ThreadedPairDataset(Protocol[ImageT, MaskT, SampleT]):
    """Dataset operations required by :class:`ThreadedSampleLoader`."""

    def read_pair(self, index: int) -> tuple[ImageT, MaskT]:
        """Decode one image/mask pair without consuming any RNG state."""

        ...

    def transform_pair(self, index: int, image: ImageT, mask: MaskT) -> SampleT:
        """Transform one decoded pair into the dataset's returned sample."""

        ...


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return value


class ThreadedSampleLoader:
    """Load decoded pairs concurrently and transform them in sampler order.

    A new iterator (and therefore a new bounded thread pool) is created for
    each pass over the loader.  ``prefetch_samples`` is a hard upper bound on
    submitted-but-not-consumed sample reads, including reads that have already
    completed.  Call ``close()`` on an explicitly retained iterator when
    abandoning it early; errors and normal exhaustion close it automatically.
    """

    def __init__(
        self,
        dataset: ThreadedPairDataset[Any, Any, Any],
        *,
        sampler: Sampler[int],
        batch_size: int,
        num_threads: int,
        prefetch_samples: int,
        drop_last: bool = False,
        pin_memory: bool = False,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = _positive_int("batch_size", batch_size)
        self.num_threads = _positive_int("num_threads", num_threads)
        self.prefetch_samples = _positive_int("prefetch_samples", prefetch_samples)
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last must be a boolean; received {drop_last!r}.")
        self.drop_last = drop_last
        if not isinstance(pin_memory, bool):
            raise ValueError(f"pin_memory must be a boolean; received {pin_memory!r}.")
        self.pin_memory = pin_memory

        if not callable(getattr(dataset, "read_pair", None)):
            raise TypeError("dataset must expose a callable read_pair(index).")
        if not callable(getattr(dataset, "transform_pair", None)):
            raise TypeError(
                "dataset must expose a callable transform_pair(index, image, mask)."
            )
        if not isinstance(sampler, Sized):
            raise TypeError("sampler must have a deterministic __len__ implementation.")
        try:
            sampler_length = len(sampler)
        except (NotImplementedError, TypeError) as exc:
            raise TypeError("sampler must have a deterministic __len__ implementation.") from exc
        if sampler_length < 0:
            raise ValueError("sampler length must not be negative.")
        self._sampler_length = sampler_length
        if drop_last:
            self._sample_count = (sampler_length // self.batch_size) * self.batch_size
        else:
            self._sample_count = sampler_length

    def __len__(self) -> int:
        full_batches, remainder = divmod(self._sampler_length, self.batch_size)
        if self.drop_last or remainder == 0:
            return full_batches
        return full_batches + 1

    def __iter__(self) -> _ThreadedSampleIterator:
        return _ThreadedSampleIterator(self)


class _ThreadedSampleIterator(Iterator[Any]):
    """One closable traversal of a :class:`ThreadedSampleLoader`."""

    def __init__(self, loader: ThreadedSampleLoader) -> None:
        self._loader = loader
        self._indices = iter(loader.sampler)
        self._executor = ThreadPoolExecutor(
            max_workers=loader.num_threads,
            thread_name_prefix="floodsight-segmentation-reader",
        )
        self._pending: deque[tuple[int, Future[tuple[Any, Any]]]] = deque()
        self._submitted = 0
        self._source_exhausted = False
        self._closed = False
        try:
            self._fill_prefetch()
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> _ThreadedSampleIterator:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration

        samples: list[Any] = []
        try:
            while len(samples) < self._loader.batch_size and self._pending:
                index, future = self._pending.popleft()
                try:
                    image, mask = future.result()
                except StopIteration as exc:
                    raise RuntimeError(
                        f"dataset.read_pair({index}) raised StopIteration."
                    ) from exc

                # Refill before transforming so decoding overlaps the consuming
                # thread's transform while never changing transform order.
                self._fill_prefetch()
                try:
                    sample = self._loader.dataset.transform_pair(index, image, mask)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"dataset.transform_pair({index}, ...) raised StopIteration."
                    ) from exc
                samples.append(sample)

            if not samples:
                self.close()
                raise StopIteration

            if len(samples) < self._loader.batch_size and self._loader.drop_last:
                self.close()
                raise StopIteration

            if not self._pending and self._source_exhausted:
                self.close()
            batch = default_collate(samples)
            if self._loader.pin_memory:
                batch = pin_memory_batch(batch)
            return batch
        except BaseException:
            self.close()
            raise

    def _fill_prefetch(self) -> None:
        while (
            not self._source_exhausted
            and len(self._pending) < self._loader.prefetch_samples
        ):
            if self._submitted == self._loader._sample_count:
                self._consume_dropped_tail_and_assert_sampler_exhausted()
                return
            try:
                index = next(self._indices)
            except StopIteration as exc:
                raise RuntimeError(
                    "sampler yielded fewer indices than its declared length: "
                    f"expected {self._loader._sampler_length}, got {self._submitted}."
                ) from exc
            future = self._executor.submit(self._loader.dataset.read_pair, index)
            self._pending.append((index, future))
            self._submitted += 1

    def _consume_dropped_tail_and_assert_sampler_exhausted(self) -> None:
        dropped_tail = self._loader._sampler_length - self._loader._sample_count
        for offset in range(dropped_tail):
            try:
                next(self._indices)
            except StopIteration as exc:
                seen = self._submitted + offset
                raise RuntimeError(
                    "sampler yielded fewer indices than its declared length: "
                    f"expected {self._loader._sampler_length}, got {seen}."
                ) from exc
        try:
            extra_index = next(self._indices)
        except StopIteration:
            self._source_exhausted = True
            return
        raise RuntimeError(
            "sampler yielded more indices than its declared length: "
            f"expected {self._loader._sampler_length}; next extra index is {extra_index!r}."
        )

    def close(self) -> None:
        """Cancel queued reads and join producer threads; safe to call repeatedly."""

        if self._closed:
            return
        self._closed = True
        while self._pending:
            _, future = self._pending.popleft()
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> _ThreadedSampleIterator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # ``__del__`` covers the ordinary ``for ...: break`` case on CPython;
        # retained iterators should use ``close`` or the context-manager API.
        with suppress(Exception):
            self.close()


__all__ = ["ThreadedPairDataset", "ThreadedSampleLoader"]
