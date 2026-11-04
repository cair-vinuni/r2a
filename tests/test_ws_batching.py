"""The websocket policy server folds concurrent clients' requests into one predict_action call."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals" / "libero_plus" / "harness"))

from tools.websocket_policy_server import (WebsocketPolicyServer, merge_payloads,  # noqa: E402
                                           split_output)


def _payload(n, tag):
    img = np.full((4, 4, 3), tag, np.uint8)
    return {"batch_images": [[img, img]] * n, "instructions": [f"task {tag}"] * n,
            "state": [np.zeros((1, 8), np.float32)] * n, "episode_step": [0] * n,
            "worker_id": [tag] * n, "unnorm_key": "libero"}


def test_merge_concatenates_per_item_fields_and_keeps_the_first_scalar():
    merged, sizes = merge_payloads([_payload(1, 1), _payload(2, 2)])
    assert sizes == [1, 2]
    assert len(merged["batch_images"]) == 3
    assert merged["instructions"] == ["task 1", "task 2", "task 2"]
    assert merged["worker_id"] == [1, 2, 2]
    assert merged["unnorm_key"] == "libero"


def test_split_cuts_batched_arrays_and_shares_the_rest():
    out = {"actions": np.arange(3 * 8 * 7, dtype=np.float32).reshape(3, 8, 7), "model": "x"}
    parts = split_output(out, [1, 2])
    assert parts[0]["actions"].shape == (1, 8, 7)
    assert parts[1]["actions"].shape == (2, 8, 7)
    assert np.array_equal(parts[1]["actions"][0], out["actions"][1])
    assert parts[0]["model"] == parts[1]["model"] == "x"


class _Recorder:
    """A policy that answers each item with its worker id and records the batch sizes it saw."""

    def __init__(self):
        self.batches = []

    def predict_action(self, batch_images, instructions, state=None, episode_step=None,
                       worker_id=None, **kw):
        self.batches.append(len(batch_images))
        acts = np.stack([np.full((8, 7), float(w), np.float32) for w in worker_id])
        return {"actions": acts}


def _drive(server, requests):
    """Enqueue requests to a running batch loop and collect their replies."""

    async def go():
        server._queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(server._batch_loop())
        futures = []
        for req in requests:
            fut = loop.create_future()
            await server._queue.put((req, fut))
            futures.append(fut)
        replies = await asyncio.gather(*futures)
        task.cancel()
        return replies

    return asyncio.run(go())


def test_queued_requests_are_answered_from_one_call_each_with_its_own_rows():
    policy = _Recorder()
    server = WebsocketPolicyServer(policy, max_batch=8, batch_wait_ms=20)
    replies = _drive(server, [_payload(1, tag) for tag in (3, 5, 7)])
    assert policy.batches == [3] or sum(policy.batches) == 3
    for reply, tag in zip(replies, (3, 5, 7)):
        assert reply["status"] == "ok"
        assert reply["data"]["actions"].shape == (1, 8, 7)
        assert float(reply["data"]["actions"][0, 0, 0]) == tag


def test_a_batch_never_exceeds_max_batch():
    policy = _Recorder()
    server = WebsocketPolicyServer(policy, max_batch=2, batch_wait_ms=20)
    replies = _drive(server, [_payload(1, tag) for tag in range(5)])
    assert max(policy.batches) <= 2
    assert [float(r["data"]["actions"][0, 0, 0]) for r in replies] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_failing_policy_answers_every_request_with_an_error():
    class Broken:
        def predict_action(self, **kw):
            raise RuntimeError("boom")

    server = WebsocketPolicyServer(Broken(), max_batch=4, batch_wait_ms=20)
    replies = _drive(server, [_payload(1, 1), _payload(1, 2)])
    assert all(r["status"] == "error" and "boom" in r["error"]["message"] for r in replies)


def test_max_batch_one_keeps_the_unbatched_path():
    server = WebsocketPolicyServer(_Recorder(), max_batch=1, workers=2)
    assert server._max_batch == 1
    reply = server._route_message(_payload(1, 9))
    assert reply["status"] == "ok" and float(reply["data"]["actions"][0, 0, 0]) == 9.0


@pytest.mark.parametrize("bad", [[], "x"])
def test_a_malformed_payload_is_an_error_not_a_crash(bad):
    server = WebsocketPolicyServer(_Recorder(), max_batch=4)
    replies = server._route_batch([{"payload": bad}, _payload(1, 1)])
    assert replies[0]["status"] == "error"


def _ws_server():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "evals" / "pi05" / "ws_server.py"
    spec = importlib.util.spec_from_file_location("pi05_ws_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_noise_is_a_function_of_seed_and_step_alone():
    ws = _ws_server()
    a = ws.chunk_noise(2033, 40, (50, 32))
    assert a.shape == (50, 32) and a.dtype == np.float32
    assert np.array_equal(a, ws.chunk_noise(2033, 40, (50, 32)))
    assert not np.array_equal(a, ws.chunk_noise(2034, 40, (50, 32)))
    assert not np.array_equal(a, ws.chunk_noise(2033, 41, (50, 32)))
    assert abs(float(a.mean())) < 0.1 and abs(float(a.std()) - 1.0) < 0.1


def test_padded_noise_repeats_the_last_item_to_the_batch_width():
    ws = _ws_server()
    items = [np.full((2, 3), float(i), np.float32) for i in range(3)]
    padded = ws._padded_noise(items, 8)
    assert padded.shape == (8, 2, 3)
    assert np.array_equal(padded[:3], np.stack(items))
    assert np.array_equal(padded[3:], np.stack([items[-1]] * 5))


def test_merge_carries_each_items_seed():
    merged, _ = merge_payloads([dict(_payload(1, 1), seed=[2033]), dict(_payload(2, 2), seed=[2040, 2041])])
    assert merged["seed"] == [2033, 2040, 2041]
