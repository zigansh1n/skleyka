"""Bounded parallel map, independent review, explicit unresolved outcomes."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import httpx

from .cli import fingerprint
from .provider import TransientAPIError, VisionAPI
from .quality_plan import ReadyPlan, candidate, review_window
from .quality_schema import (
    PROMPT_VERSION,
    reconcile,
    review_prompt,
    validate_review,
    validate_window,
    window_prompt,
)
from .quality_store import Store

TRANSIENT_RETRIES = 8


class Runner:
    def __init__(self, args, plan, config, transcript, api_factory=VisionAPI):
        self.args, self.plan, self.config = args, plan, config
        self.transcript = transcript
        self.api_factory = api_factory
        self.identity = api_factory(config).identity
        self.run_key = fingerprint(
            {
                "plan": plan["plan_key"],
                "provider": self.identity,
                "prompt": PROMPT_VERSION,
                "transcript": transcript,
                "output_tokens": config.get("MONTAGE_MAX_OUTPUT_TOKENS", "8192"),
            }
        )
        self.store = Store(
            args.output / "precision.sqlite", self.run_key, args.max_calls, args.max_tokens
        )
        self.stop = threading.Event()
        self.failures = []
        self.started = time.monotonic()
        self.context = {
            "duration": plan["metadata"]["duration"],
            "source_offset": plan["settings"]["source_offset"],
            "purpose": "повторить монтаж референса",
            "all_input_frames": True,
        }

    def task(self, name, window, prompt, validator):
        key = fingerprint({"name": name, "window": window, "prompt": prompt, "run": self.run_key})
        return {
            "name": name,
            "key": key,
            "window": window,
            "prompt": prompt,
            "validator": validator,
        }

    def execute(self, task):
        cached = self.store.cached(task["key"])
        if cached is not None:
            return task["validator"](cached, task["window"])
        last_error = None
        invalid = transient = 0
        # Up to three answers per task; overload and network drops are retried separately
        # with jittered backoff, so a burst of 429s at high concurrency does not end the run.
        while invalid < 3 and transient <= TRANSIENT_RETRIES:
            if self.stop.is_set():
                raise RuntimeError("Run stopped; completed requests retained")
            # A conservative planning reservation, not a provider-enforced billing cap.
            reserve = (
                len(task["window"]["frames"]) * 4096
                + int(self.config.get("MONTAGE_MAX_OUTPUT_TOKENS", "8192"))
                + len(task["prompt"]) * 4
            )
            attempt = self.store.begin(task["key"], reserve)
            api = self.api_factory(self.config, max_attempts=1)
            value, error = None, None
            try:
                raw = api.request(
                    task["prompt"],
                    [(self.args.output / f["path"], f["time"]) for f in task["window"]["frames"]],
                )
                value = task["validator"](raw, task["window"])
            except Exception as exc:
                # Provider exceptions contain only sanitized errors, never request bodies/keys.
                error = f"{type(exc).__name__}: {exc}"[:300]
                last_error = exc
            finally:
                self.store.finish(
                    attempt, task["key"], value, api.usage[-1] if api.usage else {}, error
                )
            if error is None:
                return value
            if isinstance(last_error, (TransientAPIError, httpx.TransportError)):
                transient += 1
                delay = min(30, 2**transient) * random.uniform(0.5, 1.5)
            else:
                invalid += 1
                delay = min(2 ** (invalid - 1), 4)
            if self.stop.wait(delay):
                break
        raise RuntimeError(f"{task['name']}: {last_error}")

    def stage(self, tasks, label, follow=None, feed=None):
        """Run tasks on one pool; feed() adds windows as they become ready and
        follow(task, value) queues dependent reviews as soon as their inputs exist.

        Windows go first: they are the long requests and every review waits on one."""
        results = {}
        windows, reviews = deque(tasks), deque()
        done_count = {}
        total = len(self.plan["windows"])
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            active = {}
            while True:
                # Snapshot before draining: the feed publishes everything before it is done.
                waiting = feed is not None and not self.feed.done
                if feed is not None:
                    windows.extend(feed(0))
                while (windows or reviews) and len(active) < self.args.workers:
                    if self.stop.is_set():
                        break
                    task = (windows or reviews).popleft()
                    active[pool.submit(self.execute, task)] = task
                if not active:
                    if not waiting or self.stop.is_set():
                        break
                    windows.extend(feed(1))
                    continue
                done, _ = wait(
                    active, timeout=0.5 if waiting else None, return_when=FIRST_COMPLETED
                )
                for future in done:
                    task = active.pop(future)
                    try:
                        results[task["name"]] = value = future.result()
                    except Exception as exc:
                        self.failures.append({"task": task["name"], "reason": str(exc)[:300]})
                        self.stop.set()
                        continue
                    kind = task["name"].split(":")[0] if ":" in task["name"] else label
                    done_count[kind] = done_count.get(kind, 0) + 1
                    of = f"/{total}" if kind == label else ""
                    print(f"{kind}: {done_count[kind]}{of} ({task['name']})", flush=True)
                    if follow is not None and not self.stop.is_set():
                        reviews.extend(follow(task, value))
        return results

    def analyze(self, feed=None):
        """feed: ReadyPlan or PlanBuilder from prepare(); windows are analyzed as it
        publishes them. A plan passed without a feed must be complete."""
        self.feed = feed or ReadyPlan(self.plan)
        primary = []
        candidates, votes = {}, {}

        def fresh(timeout):
            tasks = []
            for index in self.feed.take(timeout):
                w = self.plan["windows"][index]
                for local in w["candidates"]:
                    c = candidate(w["frames"][local["right_ref"] - 1]["index"], self.plan["frames"])
                    candidates[c["id"]] = c
                    votes[c["id"]] = []
                speech = [
                    s
                    for s in self.transcript
                    if s["end"] > w["frames"][0]["time"] and s["start"] <= w["frames"][-1]["time"]
                ]
                prompt = window_prompt(w, speech, {}, self.context)
                tasks.append(self.task(w["id"], w, prompt, validate_window))
            primary.extend(tasks)
            if self.feed.error is not None:
                self.stop.set()
                raise self.feed.error
            return tasks

        # Fail fast on an unavailable model before opening the pool.
        pilot = []
        while not pilot and not self.feed.done:
            pilot = fresh(1)
        pilot = pilot or fresh(0)
        self.stage(pilot[:1], "Pilot")
        ready = pilot

        def review(cid):
            # At most three votes; omitted primary verdicts still receive two reviews.
            if (
                len(votes[cid]) >= 3
                or reconcile(candidates[cid], votes[cid])["status"] == "reviewed"
            ):
                return []
            round_number = 1 + sum(v["stage"] != "primary" for v in votes[cid])
            w = review_window(self.plan, candidates[cid], radius=6 if round_number == 1 else 10)
            prompt = review_prompt(w, third=round_number > 1)
            return [self.task(f"review{round_number}:{cid}", w, prompt, validate_review)]

        def follow(task, value):
            w = task["window"]
            if task["name"].startswith("review"):
                v = value["boundary"]
                votes[v["id"]].append(self.attach(v, w, task["name"].split(":")[0]))
                return review(v["id"])
            for verdict in value["boundaries"]:
                votes[verdict["id"]].append(self.attach(verdict, w, "primary"))
            owned = [c["id"] for c in w["candidates"]]
            for proposed in value["extra_boundaries"]:
                i = w["frames"][proposed["after_ref"] - 1]["index"]
                if not w["core_start_index"] <= i < w["core_stop_index"]:
                    continue
                c = candidate(i, self.plan["frames"], "model_proposal")
                candidates.setdefault(c["id"], c)
                votes.setdefault(c["id"], [])
                owned.append(c["id"])
            return [t for cid in sorted(set(owned)) for t in review(cid)]

        first = {}
        if not self.stop.is_set():
            # The pilot answer comes from the ledger here, at no cost.
            first = self.stage(ready, "Windows", follow, fresh)
        primary.sort(key=lambda t: t["window"]["core_start_index"])
        events, windows = [], []
        for task in primary:
            value = first.get(task["name"])
            if value is None:
                continue
            w = task["window"]
            windows.append(
                {"id": task["name"], "start": w["core_start"], "end": w["core_end"], **value}
            )
            for e in value["events"]:
                a, b = w["frames"][e["start_ref"] - 1], w["frames"][e["end_ref"] - 1]
                if b["index"] < w["core_start_index"] or a["index"] >= w["core_stop_index"]:
                    continue
                events.append(
                    {
                        **e,
                        "start": a["time"],
                        "end": b["time"],
                        "window": w["id"],
                        "first_frame": a,
                        "last_frame": b,
                        "status": "model_observation",
                    }
                )
        # Candidates of windows never analyzed (stopped run) still belong in the report.
        for c in list(self.plan["candidates"]):
            candidates.setdefault(c["id"], c)
            votes.setdefault(c["id"], [])
        boundaries = [reconcile(c, votes[cid]) for cid, c in sorted(candidates.items())]
        indices = [0] + [c["index"] for c in boundaries if c["kind"] in ("hard_cut", "crop_jump")]
        indices.append(len(self.plan["frames"]))
        shots = []
        for number, (start, stop) in enumerate(zip(indices, indices[1:], strict=False), 1):
            first_frame = self.plan["frames"][start]
            end = (
                self.plan["frames"][stop]["time"]
                if stop < len(self.plan["frames"])
                else self.plan["settings"]["source_offset"] + self.plan["metadata"]["duration"]
            )
            shots.append(
                {
                    "id": f"s{number:05d}",
                    "start": first_frame["time"],
                    "end": end,
                    "duration": end - first_frame["time"],
                    "first_frame_index": start,
                    "last_frame_index": stop - 1,
                }
            )
        complete = len(windows) == len(self.plan["windows"]) and not self.stop.is_set()
        unresolved = [c["id"] for c in boundaries if c["status"] != "reviewed"]
        return {
            "version": PROMPT_VERSION,
            "run_key": self.run_key,
            "provider": self.identity,
            "status": "complete" if complete else "partial",
            "quality_status": "needs_review"
            if unresolved
            else ("reviewed" if complete else "incomplete"),
            "completed_windows": len(windows),
            "total_windows": len(self.plan["windows"]),
            "source_frames": len(self.plan["frames"]),
            "analyzed_frames": sum(
                t["window"]["core_stop_index"] - t["window"]["core_start_index"]
                for t in primary
                if t["name"] in first
            ),
            "boundaries": boundaries,
            "shots": shots,
            "events": events,
            "windows": windows,
            "unresolved_candidates": unresolved,
            "failures": self.failures,
            "usage": self.store.summary(),
            "elapsed_seconds_this_run": time.monotonic() - self.started,
            "limitations": [
                "Согласие двух проверок не доказывает правильность классификации.",
                "Все кадры переданы модели; внимание модели к каждому кадру не гарантируется.",
                "Звук не классифицируется. Транскрипт содержит только речь.",
                "События внутри планов — наблюдения одного прохода; "
                "границы проверяются независимо.",
                "Разрешение JPEG ограничено frame-size; "
                "исходные пиксели доступны в исходном видео.",
            ],
        }

    @staticmethod
    def attach(verdict, window, stage):
        return {
            **verdict,
            "stage": stage,
            "evidence": [window["frames"][i - 1] for i in verdict["evidence_refs"]],
        }
