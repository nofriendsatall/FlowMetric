"""Serialization and annotated-video rendering."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flowmetric.tracking import LocalPass

# ---------------------------------------------------------------------------
# Pass 3: render
# ---------------------------------------------------------------------------

ROLE_BGR = {"staff": (219, 152, 52), "customer": (113, 204, 46)}


def id_color(gid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(gid * 9973)
    c = rng.integers(64, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


EVENT_LABEL = {
    "enter": "entered",
    "exit": "left frame",
    "leave_kitchen": "left kitchen",
    "seated": "seated",
    "service_start": "service start",
    "service_end": "service end",
}


def event_text(e: dict[str, Any]) -> str:
    who = f"ID {e['staff']} -> ID {e['customer']}" if "staff" in e else f"ID {e['person']}"
    return f"{e['t']:5.1f}s  {EVENT_LABEL.get(e['type'], e['type'])}: {who}"


def render(
    source: Path,
    dest: Path,
    lp: LocalPass,
    tid_to_gid: dict[int, int],
    gid_role: dict[int, str],
    *,
    mode: str,
    events: dict[str, Any] | None = None,
    trail_len: int = 30,
    event_hold_s: float = 3.0,
) -> None:
    import av

    if len(lp.frame_times) != lp.total:
        raise ValueError(
            f"frame_times length ({len(lp.frame_times)}) "
            f"does not match total frames ({lp.total})"
        )
    if lp.total <= 0:
        raise ValueError("cannot render video with no frames")
    if not np.all(np.isfinite(lp.frame_times)):
        raise ValueError("frame_times must be finite")
    if np.any(np.diff(lp.frame_times) <= 0):
        raise ValueError("frame_times must be strictly increasing")

    dest.parent.mkdir(parents=True, exist_ok=True)

    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen: set[int] = set()
    sampled = set(lp.sample_idx) if mode.startswith("hybrid") else set()
    max_jump = 0.08 * max(lp.width, lp.height)
    ev_list = (events or {}).get("events", [])
    waiting = [
        s for s in (events or {}).get("states", [])
        if s["type"] == "waiting"
    ]
    roles = {
        **{str(k): v for k, v in gid_role.items()},
        **(events or {}).get("roles", {}),
    }

    try:
        with av.open(str(source)) as inp:
            src = next(
                (s for s in inp.streams if s.type == "video"),
                None,
            )
            if src is None:
                raise RuntimeError(f"source has no video stream: {source}")

            with av.open(str(dest), "w") as out:
                dst = out.add_stream("libx264")
                dst.width = lp.width
                dst.height = lp.height
                dst.pix_fmt = "yuv420p"
                dst.time_base = src.time_base
                dst.codec_context.time_base = src.time_base
                dst.options = {"preset": "ultrafast", "crf": "23"}

                frame_count = 0

                first_source_time = None
                timestamp_tolerance = 1e-6

                if src.time_base is None:
                    raise RuntimeError(
                        f"source video stream has no time_base: {source}"
                    )

                for idx, decoded_frame in enumerate(inp.decode(src)):
                    if idx >= lp.total:
                        raise RuntimeError(
                            f"source contains more frames than LocalPass: "
                            f"at least {idx + 1}, expected {lp.total}"
                        )

                    if decoded_frame.pts is None:
                        raise RuntimeError(
                            f"source frame {idx} has no PTS"
                        )
                    source_time = float(decoded_frame.pts * src.time_base)

                    if first_source_time is None:
                        first_source_time = source_time

                    relative_time = source_time - first_source_time
                    expected_time = lp.frame_times[idx]

                    if abs(relative_time - expected_time) > timestamp_tolerance:
                        raise RuntimeError(
                            f"source timestamp mismatch at frame {idx}: "
                            f"source={relative_time:.9f}s, "
                            f"LocalPass={expected_time:.9f}s"
                        )

                    frame = decoded_frame.to_ndarray(format="bgr24")
                    now = lp.frame_times[idx]

                    dets = lp.dets.get(idx, [])
                    drawn: set[int] = set()

                    for tid, box, score in dets:
                        gid = tid_to_gid.get(tid, tid)
                        if gid in drawn:
                            continue
                        drawn.add(gid)
                        seen.add(gid)

                        x1, y1, x2, y2 = map(int, box)
                        role = roles.get(str(gid), "")
                        color = ROLE_BGR.get(role, id_color(gid))

                        if any(
                            s["person"] == str(gid)
                            and s["start"] <= now <= s["end"]
                            for s in waiting
                        ):
                            role = f"{role} waiting".strip()

                        center = (
                            (x1 + x2) // 2,
                            (y1 + y2) // 2,
                        )
                        prev = trails[gid][-1] if trails[gid] else None
                        if prev and np.hypot(
                            center[0] - prev[0],
                            center[1] - prev[1],
                        ) > max_jump:
                            trails[gid] = []

                        trails[gid].append(center)
                        trails[gid] = trails[gid][-trail_len:]
                        pts = trails[gid]

                        for i in range(1, len(pts)):
                            cv2.line(
                                frame, pts[i - 1], pts[i], color, 2
                            )

                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), color, 2
                        )
                        label = f"ID {gid}" + (
                            f" {role}" if role else ""
                        )
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                        )
                        cv2.rectangle(
                            frame,
                            (x1, max(0, y1 - th - 8)),
                            (x1 + tw + 6, y1),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 3, max(th + 2, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    hud = (
                        f"FlowMetric [{mode}]  frame {idx + 1}/{lp.total}"
                        f"  people {len(drawn)}  ids {len(seen)}"
                    )
                    if idx in sampled:
                        hud += "  * astra"

                    cv2.putText(
                        frame, hud, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 4, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, hud, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )

                    recent = [
                        e for e in ev_list
                        if now - event_hold_s <= e["t"] <= now
                    ][-6:]

                    for k, e in enumerate(reversed(recent)):
                        y = lp.height - 20 - 34 * k
                        text = event_text(e)
                        cv2.putText(
                            frame, text, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 5, cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame, text, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 220, 255), 2, cv2.LINE_AA,
                        )

                    output_frame = av.VideoFrame.from_ndarray(
                        frame, format="bgr24"
                    )
                    output_frame.pts = decoded_frame.pts
                    output_frame.time_base = decoded_frame.time_base

                    for packet in dst.encode(output_frame):
                        out.mux(packet)

                    frame_count += 1

                if frame_count != lp.total:
                    raise RuntimeError(
                        f"decoded {frame_count} frames, "
                        f"but LocalPass contains {lp.total}"
                    )

                for packet in dst.encode():
                    out.mux(packet)

    except (av.error.FFmpegError, OSError) as exc:
        raise RuntimeError(
            f"cannot render annotated video: {source} -> {dest}"
        ) from exc

def build_tracks(
    lp: LocalPass,
    tid_to_gid: dict[int, int],
    gid_role: dict[int, str],
    *,
    mode: str,
) -> dict[str, Any]:
    """Per-person trajectories keyed by final ID (input for events + benchmark)."""
    per_frame: dict[int, dict[int, tuple[int, np.ndarray, float]]] = defaultdict(dict)
    for fi, dets in lp.dets.items():
        for tid, box, score in dets:
            gid = tid_to_gid.get(tid, tid)
            prev = per_frame[gid].get(fi)
            if prev is None or score > prev[2]:
                per_frame[gid][fi] = (tid, box, score)

    tracks: dict[str, Any] = {}
    for gid, frames in per_frame.items():
        fis = sorted(frames)
        tracks[str(gid)] = {
            "role": gid_role.get(gid, ""),
            "tids": sorted({frames[f][0] for f in fis}),
            "frames": fis,
            "times": [
                round(float(lp.frame_times[f]), 6)
                for f in fis
            ],
            "boxes": [[round(float(v), 1) for v in frames[f][1]] for f in fis],
            "conf": [round(frames[f][2], 3) for f in fis],
        }
    return {
        "mode": mode,
        "fps": lp.fps,
        "width": lp.width,
        "height": lp.height,
        "n_frames": lp.total,
        "frame_times": [
            round(float(t), 6)
            for t in lp.frame_times
        ],
        "tracks": tracks,
    }
