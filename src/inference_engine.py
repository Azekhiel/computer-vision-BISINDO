"""Compatibility shim for the new fast GRU live inference engine."""

from live_gru_fast import FastGRULiveWorker, start_live_inference

__all__ = ["FastGRULiveWorker", "start_live_inference"]
