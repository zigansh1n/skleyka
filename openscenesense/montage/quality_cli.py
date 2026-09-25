from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
from pathlib import Path

from .cli import load_transcript, lock
from .media import run
from .provider import DEFAULT_CONFIG, load_config
from .quality_plan import start_plan
from .quality_report import render
from .quality_runner import Runner


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="video-montage precision",
        description="All source frames, parallel analysis and independent boundary checks",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=800,
        help="Simultaneous requests; DashScope kimi-k3 starts returning 429 near 1000",
    )
    parser.add_argument("--core-frames", type=int, default=48)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=27)
    parser.add_argument(
        "--source-offset",
        type=float,
        default=0,
        help="Seconds to add to input presentation times; never resamples frames",
    )
    parser.add_argument("--max-duration", type=float, default=14400)
    parser.add_argument("--max-frames-total", type=int, default=400000)
    parser.add_argument(
        "--max-calls",
        type=int,
        help="Cumulative across resumes in this output directory; default 150 per window",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Cumulative known tokens plus conservative reservations; not a provider billing "
        "cap; default never binds before --max-calls",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", choices=["none", "low", "high", "max"])
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument(
        "--transcript",
        type=Path,
        help="Timed transcript on the output/source clock, including --source-offset",
    )
    speech.add_argument(
        "--transcribe",
        action="store_true",
        help="Local Whisper; adjusts clip-relative timestamps by --source-offset",
    )
    args = parser.parse_args(argv)
    if not (
        1 <= args.workers <= 2000
        and 4 <= args.core_frames <= 60
        and 1 <= args.context_frames <= 16
        and args.core_frames + 2 * args.context_frames <= 64
        and 160 <= args.frame_size <= 2048
        and math.isfinite(args.source_offset)
        and args.source_offset >= 0
        and math.isfinite(args.threshold)
        and 0 < args.threshold <= 255
        and math.isfinite(args.max_duration)
        and args.max_duration > 0
        and (args.max_calls is None or args.max_calls > 0)
        and (args.max_tokens is None or args.max_tokens > 0)
        and args.max_frames_total > 0
    ):
        parser.error("Invalid limits; core frames plus both context margins must not exceed 64")
    args.output = args.output.expanduser().resolve()
    try:
        with lock(args.output):
            stop = threading.Event()
            plan, feed = start_plan(args, stop)
            print(
                f"Preparing {len(plan['frames'])} source frames in {len(plan['windows'])} "
                "windows; analysis starts as windows become ready",
                flush=True,
            )
            if args.prepare_only:
                while not feed.done:
                    feed.take(1)
                if feed.error is not None:
                    raise feed.error
                print(f"Prepared {len(plan['candidates'])} boundary candidates", flush=True)
                return 0
            config = load_config(args.config)
            if args.model:
                config["MONTAGE_MODEL"] = args.model
            if args.reasoning:
                config["MONTAGE_REASONING_EFFORT"] = args.reasoning
            if args.transcribe:
                if not plan["metadata"]["has_audio"]:
                    raise ValueError("--transcribe requested but input has no audio")
                python, script = config.get("MONTAGE_ASR_PYTHON"), config.get("MONTAGE_ASR_SCRIPT")
                if not python or not script:
                    raise ValueError("Local ASR is not configured")
                args.transcript = args.output / "precision-transcript.txt"
                if not args.transcript.exists():
                    temp = args.output / "precision-transcript.pending.txt"
                    run([python, script, plan["source"], "-o", str(temp)], timeout=14400)
                    temp.replace(args.transcript)
            transcript = load_transcript(args.transcript)
            if args.transcribe:
                transcript = [
                    {
                        **s,
                        "start": s["start"] + args.source_offset,
                        "end": s["end"] + args.source_offset,
                    }
                    for s in transcript
                ]
            if args.max_calls is None:
                # Candidates are still being detected; 150 covers a cut on every frame of a
                # 48-frame window (one primary plus two reviews each) with retries to spare.
                args.max_calls = 150 * len(plan["windows"])
            if args.max_tokens is None:
                per_call = (args.core_frames + 2 * args.context_frames) * 4096 + 65536
                args.max_tokens = args.max_calls * per_call
            runner = Runner(args, plan, config, transcript)
            runner.stop = stop
            previous = {}
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, lambda *_: stop.set())
            try:
                result = runner.analyze(feed)
                render(plan, result, args.output)
            finally:
                runner.store.close()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            print(
                f"{result['status']}; boundaries {result['quality_status']}; "
                f"{args.output / 'report.html'}",
                flush=True,
            )
            return (
                1
                if result["status"] != "complete"
                else (2 if result["quality_status"] == "needs_review" else 0)
            )
    except Exception as exc:
        print(f"video-montage precision: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
