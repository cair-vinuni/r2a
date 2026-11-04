# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
"""Websocket policy server: one process, one GPU-resident policy, many sim clients.

Every client sends one observation per request and blocks for the answer, so with max_batch above
one the server folds the requests that arrive while an inference runs into the next call to
policy.predict_action, whose contract is already a batch (lists of length B), and splits the
answer back per request. A policy that loops per item inside predict_action is served with
max_batch 1, which keeps the original one-request-per-call path.
"""

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import websockets.asyncio.server
import websockets.frames

from . import image_tools, msgpack_numpy

# Request fields that are per-item lists; a batch is their concatenation, in request order.
BATCH_FIELDS = ("batch_images", "instructions", "state", "episode_step", "worker_id",
                "obj_pos", "place_pos", "grasped", "seed")


def merge_payloads(payloads: list[dict]) -> tuple[dict, list[int]]:
    """Concatenate per-request payloads into one batched payload.

    Args:
        payloads: One infer payload per request, each with batch_images of length B_i.

    Returns:
        The merged payload and the per-request sizes B_i. A field that is not a per-item list
        takes the first request's value.
    """
    sizes = [len(p["batch_images"]) for p in payloads]
    merged: dict[str, Any] = {}
    for p in payloads:
        for key, value in p.items():
            if key in BATCH_FIELDS:
                merged.setdefault(key, []).extend(list(value))
            else:
                merged.setdefault(key, value)
    return merged, sizes


def split_output(output: dict, sizes: list[int]) -> list[dict]:
    """Cut a batched predict_action answer back into one dict per request.

    Args:
        output: The policy's answer; every array or list whose leading dimension is sum(sizes)
            is split, anything else is shared.
        sizes: Per-request item counts, in request order.

    Returns:
        One answer per request.
    """
    total = sum(sizes)
    outs: list[dict] = [{} for _ in sizes]
    for key, value in output.items():
        is_batched = (isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total) \
            or (isinstance(value, (list, tuple)) and len(value) == total)
        start = 0
        for i, n in enumerate(sizes):
            outs[i][key] = value[start:start + n] if is_batched else value
            start += n
    return outs


class BatchStats:
    """Rolling batch statistics, logged every `every` calls so a long run shows its steady state."""

    def __init__(self, every: int = 500) -> None:
        self.every = every
        self.calls = self.items = 0
        self.seconds = 0.0
        self.started = time.monotonic()

    def add(self, batch_size: int, seconds: float) -> None:
        self.calls += 1
        self.items += batch_size
        self.seconds += seconds
        if self.calls % self.every == 0:
            elapsed = time.monotonic() - self.started
            logging.info("batch stats: %d calls, mean batch %.1f, mean call %.0f ms, %.1f items/s over %.0f s",
                         self.calls, self.items / self.calls, 1000 * self.seconds / self.calls,
                         self.items / elapsed if elapsed else 0.0, elapsed)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        workers: int = 1,
        max_batch: int = 1,
        batch_wait_ms: float = 3.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_batch = max(1, int(max_batch))
        self._batch_wait = max(0.0, float(batch_wait_ms)) / 1000.0
        # Requests run on this pool, not on the event loop. Unbatched, two workers overlap one
        # request's CPU preprocessing with another's GPU forward; batched, one worker owns the GPU.
        n_workers = 1 if self._max_batch > 1 else max(1, int(workers))
        self._pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="policy")
        self._queue: asyncio.Queue | None = None
        self._stats = BatchStats()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._queue = asyncio.Queue()
        batcher = asyncio.create_task(self._batch_loop()) if self._max_batch > 1 else None
        if batcher is not None:
            logging.info("dynamic batching: up to %d requests per call, %.1f ms gather window",
                         self._max_batch, self._batch_wait * 1000)
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                # Disable keepalive ping: a slow synchronous inference (e.g. the flow-matching head's
                # multi-step ODE sampling, esp. on a shared GPU) blocks the event loop longer than the
                # default 20s ping timeout, which would otherwise drop the client mid-episode.
                ping_interval=None,
                ping_timeout=None,
            ) as server:
                await server.serve_forever()
        finally:
            if batcher is not None:
                batcher.cancel()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                if self._max_batch > 1 and msg.get("type", "infer") == "infer":
                    future = asyncio.get_running_loop().create_future()
                    await self._queue.put((msg, future))
                    ret = await future
                else:
                    ret = await asyncio.get_running_loop().run_in_executor(self._pool, self._route_message, msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _batch_loop(self) -> None:
        """Take every queued request, wait batch_wait for stragglers, answer them in one call."""
        loop = asyncio.get_running_loop()
        assert self._queue is not None
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self._batch_wait
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            t0 = loop.time()
            replies = await loop.run_in_executor(self._pool, self._route_batch, [m for m, _ in batch])
            self._stats.add(len(batch), loop.time() - t0)
            for (_, future), reply in zip(batch, replies):
                if not future.done():
                    future.set_result(reply)

    def _route_batch(self, msgs: list[dict]) -> list[dict]:
        """One predict_action call for several infer messages; per-message answers."""
        req_ids = [m.get("request_id", "default") for m in msgs]
        payloads = [m.get("payload", m) for m in msgs]
        bad = [i for i, p in enumerate(payloads) if not isinstance(p, dict) or "batch_images" not in p]
        if bad:
            return [self._error(rid, "Payload must be a dict with batch_images") for rid in req_ids]
        try:
            merged, sizes = merge_payloads(payloads)
            merged["batch_images"] = image_tools.to_pil_preserve(merged["batch_images"])
            output = self._policy.predict_action(**merged)
            parts = split_output(output, sizes)
        except Exception as e:
            logging.exception("Policy inference error (batch of %d)", len(msgs))
            return [self._error(rid, str(e)) for rid in req_ids]
        return [self._ok(rid, part) for rid, part in zip(req_ids, parts)]

    @staticmethod
    def _ok(req_id: str, data: dict) -> dict:
        return {"status": "ok", "ok": True, "type": "inference_result", "request_id": req_id,
                "data": data}

    @staticmethod
    def _error(req_id: str, message: str, mtype: str = "inference_result") -> dict:
        return {"status": "error", "ok": False, "type": mtype, "request_id": req_id,
                "error": {"message": message}}

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload

        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        elif mtype == "infer":
            if not isinstance(payload, dict):
                return self._error(req_id, "Payload must be a dict")
            try:
                payload["batch_images"] = image_tools.to_pil_preserve(payload["batch_images"])
                output = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return self._error(req_id, str(e))
            return self._ok(req_id, output)

        else:
            return self._error(req_id, f"Unsupported message type '{mtype}'", mtype="unknown")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise NotImplementedError("This module is not intended to be run directly.")
