"""Full presentation-frame coverage, independent of CFR/VFR and worker count."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .cli import fingerprint
from .media import (
    detect_segment,
    digest,
    extract_segment,
    finish_segment,
    frame_times,
    probe,
    video_start_offset,
    write_json,
)


def candidate(index, frames, origin="detector"):
    if not 0 < index < len(frames):
        raise ValueError("Boundary must have an adjacent frame on both sides")
    return {
        "id": f"c{index:08d}",
        "index": index,
        "time": frames[index]["time"],
        "origin": origin,
        "left": frames[index - 1],
        "right": frames[index],
    }


def window(frames, start, stop, candidates, duration, offset, context=8):
    lo, hi = max(0, start - context), min(len(frames), stop + context)
    selected = frames[lo:hi]
    local = [
        {
            "id": c["id"],
            "time": c["time"],
            "left_ref": c["index"] - lo,
            "right_ref": c["index"] - lo + 1,
        }
        for c in candidates
    ]
    return {
        "frames": selected,
        "core_start_index": start,
        "core_stop_index": stop,
        "core_start": frames[start]["time"],
        "core_end": frames[stop]["time"] if stop < len(frames) else offset + duration,
        "candidates": local,
    }


def make_windows(frames, candidates, duration, offset, core_frames=48, context=8):
    return [
        {
            "id": f"w{start:08d}",
            **window(
                frames,
                start,
                min(len(frames), start + core_frames),
                [c for c in candidates if start <= c["index"] < start + core_frames],
                duration,
                offset,
                context,
            ),
        }
        for start in range(0, len(frames), core_frames)
    ]


def review_window(plan, boundary, radius=6):
    i = boundary["index"]
    return window(
        plan["frames"],
        i,
        i + 1,
        [boundary],
        plan["metadata"]["duration"],
        plan["settings"]["source_offset"],
        radius,
    )


def verify(plan, output, digests=True):
    paths = [(output / frame["path"]).resolve() for frame in plan["frames"]]
    for index, (frame, path) in enumerate(zip(plan["frames"], paths, strict=True)):
        if frame["index"] != index or not path.is_relative_to(output.resolve()):
            raise ValueError("Invalid evidence catalog")
    if digests:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            if list(pool.map(digest, paths)) != [f["sha256"] for f in plan["frames"]]:
                raise ValueError("Evidence changed; choose a fresh output directory")
    owned = [i for w in plan["windows"] for i in range(w["core_start_index"], w["core_stop_index"])]
    if owned != list(range(len(plan["frames"]))):
        raise ValueError("Primary windows must cover every input frame exactly once")


class ReadyPlan:
    """Window feed for a plan whose evidence is already on disk."""

    def __init__(self, plan):
        self.plan, self.error, self.done = plan, None, True
        self._fresh = list(range(len(plan["windows"])))

    def take(self, timeout=None):
        fresh, self._fresh = self._fresh, []
        return fresh


class PlanBuilder:
    """Decodes, hashes and scans the video in ordered segments on background workers,
    publishing each primary window once its frames (plus a review margin) are final.
    Analysis starts on the first minutes while later minutes are still being decoded."""

    SEGMENT = 1500
    WARMUP = 32  # frames replayed before each detection segment
    LOOKAHEAD = 8  # AdaptiveDetector decides on a frame two frames later
    MARGIN = 16  # focused reviews look up to ten frames past a boundary

    def __init__(self, plan, source, times, output, args, stop):
        self.plan, self.source, self.times = plan, source, times
        self.output, self.args, self.stop = output, args, stop
        self.error, self.done = None, False
        self.cond = threading.Condition()
        self._fresh, self._published = [], 0
        self.thread = threading.Thread(target=self._run, name="plan-builder", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def take(self, timeout=None):
        """Newly publishable window indices; waits up to timeout for at least one."""
        with self.cond:
            if not self._fresh and not self.done:
                self.cond.wait(timeout)
            fresh, self._fresh = self._fresh, []
            return fresh

    def _publish(self, upto):
        frames, plan = self.plan["frames"], self.plan
        with self.cond:
            windows = plan["windows"]
            while self._published < len(windows):
                w = windows[self._published]
                if upto < len(frames) and w["core_stop_index"] + self.MARGIN > upto:
                    break
                core = [
                    c
                    for c in plan["candidates"]
                    if w["core_start_index"] <= c["index"] < w["core_stop_index"]
                ]
                windows[self._published] = {
                    "id": w["id"],
                    **window(
                        frames,
                        w["core_start_index"],
                        w["core_stop_index"],
                        core,
                        plan["metadata"]["duration"],
                        plan["settings"]["source_offset"],
                        self.args.context_frames,
                    ),
                }
                self._fresh.append(self._published)
                self._published += 1
            self.cond.notify_all()

    def _run(self):
        frames, dest = self.plan["frames"], self.output / "frames"
        n, workers = len(frames), os.cpu_count() or 4
        segments = [(a, min(n, a + self.SEGMENT)) for a in range(0, n, self.SEGMENT)]
        try:
            (dest / "seams").mkdir(parents=True, exist_ok=True)
            base = video_start_offset(self.source)
            fps = n / self.plan["metadata"]["duration"]
            pattern = str(dest / "f%08d.jpg")
            with (
                ThreadPoolExecutor(max_workers=workers) as decoders,
                ThreadPoolExecutor(max_workers=workers) as hashers,
                # spawn, not fork: a worker forked while another thread is inside Popen
                # inherits that child's exec-status pipe, and Popen then waits forever.
                ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                ) as scanners,
            ):
                decoded = [
                    decoders.submit(
                        extract_segment,
                        (str(self.source), self.times, base, a, b, dest, self.args.frame_size),
                    )
                    for a, b in segments
                ]
                scans, waiting = [], list(segments)
                for (a, b), job in zip(segments, decoded, strict=True):
                    if self.stop.is_set() or job.exception():
                        for later in decoded:
                            later.cancel()
                        job.result()
                        raise RuntimeError("Run stopped during preparation")
                    finish_segment(dest, a, b)
                    for i, sha in zip(
                        range(a, b),
                        hashers.map(digest, [dest / f"f{i:08d}.jpg" for i in range(a, b)]),
                        strict=True,
                    ):
                        frames[i]["sha256"] = sha
                    # A segment is scanned once the frames it looks ahead into are final.
                    while waiting and min(n, waiting[0][1] + self.LOOKAHEAD) <= b:
                        a0, b0 = waiting.pop(0)
                        start, end = max(0, a0 - self.WARMUP), min(n, b0 + self.LOOKAHEAD)
                        job = (pattern, fps, start, a0, b0, end, self.args.threshold)
                        scans.append(((a0, b0), scanners.submit(detect_segment, job)))
                    while scans and scans[0][1].done():
                        self._scanned(*scans.pop(0))
                for scan in scans:
                    self._scanned(*scan)
            (dest / "seams").rmdir()
            verify(self.plan, self.output, digests=False)
            write_json(self.output / "precision-plan.json", self.plan)
        except BaseException as exc:
            self.error = exc
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def _scanned(self, bounds, future):
        cuts = future.result()
        with self.cond:
            self.plan["candidates"].extend(candidate(i, self.plan["frames"]) for i in cuts)
        self._publish(bounds[1])


def prepare(args):
    """The complete plan, blocking until every frame is decoded, hashed and scanned."""
    plan, feed = start_plan(args)
    while not feed.done:
        feed.take(1)
    if feed.error is not None:
        raise feed.error
    return plan


def start_plan(args, stop=None):
    """The plan plus a window feed: ReadyPlan when evidence exists, else a running
    PlanBuilder whose plan fills in while analysis proceeds."""
    source = args.video.expanduser().resolve(strict=True)
    metadata = probe(source)
    if metadata["duration"] > args.max_duration:
        raise ValueError("Input exceeds --max-duration; choose a clip or explicitly raise limit")
    settings = {
        "version": "precision-plan-2",
        "source_sha256": digest(source),
        "source_offset": args.source_offset,
        "frame_size": args.frame_size,
        "core_frames": args.core_frames,
        "context_frames": args.context_frames,
        "threshold": args.threshold,
    }
    key = fingerprint(settings)
    destination = args.output / "precision-plan.json"
    if destination.exists():
        plan = json.loads(destination.read_text())
        if plan["plan_key"] != key:
            raise ValueError("Different input/preparation parameters: use a new output directory")
        verify(plan, args.output)
        return plan, ReadyPlan(plan)
    times = frame_times(source)
    if len(times) > args.max_frames_total:
        raise ValueError("Frame count exceeds --max-frames-total; no API requests made")
    if any(a >= b for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Frame PTS must be strictly increasing")
    frames = [
        {"index": i, "time": t + args.source_offset, "path": f"frames/f{i:08d}.jpg", "sha256": None}
        for i, t in enumerate(times)
    ]
    plan = {
        "plan_key": key,
        "settings": settings,
        "metadata": metadata,
        "source": str(source),
        "frames": frames,
        "candidates": [],
        "windows": make_windows(
            frames,
            [],
            metadata["duration"],
            args.source_offset,
            args.core_frames,
            args.context_frames,
        ),
        "clock": (
            "Presentation timestamps relative to first input frame plus explicit source offset"
        ),
    }
    builder = PlanBuilder(plan, source, times, args.output, args, stop or threading.Event())
    return plan, builder.start()
