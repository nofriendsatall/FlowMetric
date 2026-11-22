"""Domain objects with no dependency on pipeline or provider modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

TRACK_SMOOTH_SECONDS = 0.5



@dataclass
class Track:
    gid: str
    role: str
    frames: np.ndarray
    boxes: np.ndarray
    fps: float
    frame_times: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int)
        self.boxes = np.asarray(self.boxes, dtype=float)

        if self.frame_times is not None:
            self.frame_times = np.asarray(self.frame_times, dtype=float)
            if len(self.frame_times) != len(self.frames):
                raise ValueError("frame_times and frames must have equal lengths")
            if len(self.frame_times) > 1 and np.any(np.diff(self.frame_times) <= 0):
                raise ValueError("frame_times must be strictly increasing")

    @property
    def times(self) -> np.ndarray:
        if self.frame_times is not None:
            return self.frame_times
        return self.frames / self.fps

    @property
    def t0(self) -> float:
        return float(self.times[0])

    @property
    def t1(self) -> float:
        return float(self.times[-1])

    @property
    def life(self) -> float:
        return self.t1 - self.t0

    @property
    def centers(self) -> np.ndarray:
        boxes = self.boxes
        return np.c_[
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ]

    @property
    def heights(self) -> np.ndarray:
        return self.boxes[:, 3] - self.boxes[:, 1]

    def travel(self, a: int = 0, b: int | None = None) -> float:
        """Net displacement between indices, measured in body heights."""
        centers, heights = self.centers, self.heights
        b = len(centers) - 1 if b is None else b
        scale = max(np.median(heights[a : b + 1]), 1.0)
        return float(np.linalg.norm(centers[b] - centers[a]) / scale)

    def reach(self) -> float:
        """Farthest displacement from the first observation in body heights."""
        centers = self.centers
        scale = max(np.median(self.heights), 1.0)
        return float(np.max(np.linalg.norm(centers - centers[0], axis=1)) / scale)

    def path(self) -> float:
        """Total smoothed path length in body heights."""
        return float(np.sum(self.speed()[1:] * np.diff(self.times)))
    def speed(self) -> np.ndarray:
        """Smoothed center speed in body heights per second."""
        centers = self.centers
        times = self.times
        heights = self.heights

        if len(centers) < 2:
            return np.zeros(len(centers))

        half_window = TRACK_SMOOTH_SECONDS / 2.0
        smoothed = np.empty_like(centers)
        smoothed_heights = np.empty_like(heights)

        for i, t in enumerate(times):
            mask = np.abs(times - t) <= half_window
            smoothed[i] = np.mean(centers[mask], axis=0)
            smoothed_heights[i] = np.mean(heights[mask])

        dt = np.diff(times)
        velocity = (
            np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
            / np.maximum(dt, 1e-6)
        )

        return np.r_[
            0.0,
            velocity / np.maximum(smoothed_heights[1:], 1.0),
        ]

@dataclass(frozen=True)
class ServiceInterval:
    """A confirmed service episode; censored sides lack a point event."""

    staff: str
    customer: str
    start_frame: int
    end_frame: int
    left_censored: bool
    right_censored: bool
    confidence: float


@dataclass(frozen=True)
class ServiceCandidate:
    """A locally proposed interval and optional table context."""

    interval: ServiceInterval
    table_box: tuple[float, float, float, float] | None
@dataclass(frozen=True)
class ServiceDecision:
    """A verifier verdict without authority to change IDs or boundaries."""

    is_service: bool
    staff_id: str
    customer_id: str
    confidence: float
    source: str
    sampled_frames: tuple[int, ...] = ()
    model: str | None = None
    details: dict[str, Any] | None = None


ServiceVerifier = Callable[[list[ServiceCandidate]], list[ServiceDecision]]
