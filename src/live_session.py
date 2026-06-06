"""Small helpers for starting and stopping live worker sessions safely."""

from __future__ import annotations

import queue
from typing import Any, Callable


def new_status_queue() -> queue.Queue:
    return queue.Queue()


def start_worker(
    worker_factory: Callable[..., Any],
    model_type: str,
    *,
    status_queue: queue.Queue | None = None,
    **kwargs: Any,
) -> tuple[Any, queue.Queue]:
    q = status_queue or new_status_queue()
    worker = worker_factory(model_type, status_queue=q, **kwargs)
    return worker, q


def request_stop(worker: Any) -> None:
    if worker is None:
        return
    if hasattr(worker, "stop"):
        try:
            worker.stop()
        except Exception:
            pass


def force_cleanup(worker: Any) -> None:
    if worker is None:
        return
    if hasattr(worker, "force_cleanup"):
        try:
            worker.force_cleanup()
        except Exception:
            pass


def join_worker(worker: Any, timeout: float = 0.2) -> bool:
    if worker is None:
        return False
    if hasattr(worker, "join"):
        try:
            worker.join(timeout=max(0.0, float(timeout)))
        except Exception:
            pass
    if hasattr(worker, "is_alive"):
        try:
            return bool(worker.is_alive())
        except Exception:
            return False
    return False


def stop_worker(worker: Any, *, join_timeout: float = 0.2, force: bool = True) -> bool:
    request_stop(worker)
    alive = join_worker(worker, timeout=join_timeout)
    if alive and force:
        force_cleanup(worker)
        alive = join_worker(worker, timeout=join_timeout)
    return alive
