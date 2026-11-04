"""Per-episode results: episodes.jsonl records folded into summary.json and the per-task table."""

import json

from robot_ood.core.paths import repo_root

import pytest

from robot_ood.eval.episodes import (EPISODES_JSONL, SUMMARY_JSON, TRACE_PREFIX, EpisodeTrace, VideoWriter,
                                     deal_episodes,
                                     format_table, pending_episodes, per_task, read_episodes,
                                     shard_episodes, video_mode, write_summary)


def _shard(result_dir, shard, records):
    d = result_dir / f"shard{shard}_of_2"
    d.mkdir(parents=True)
    lines = ["not json"] + [json.dumps(r) for r in records]
    (d / EPISODES_JSONL).write_text("\n".join(lines) + "\n")


def test_read_episodes_gathers_every_shard_and_skips_bad_lines(tmp_path):
    _shard(tmp_path, 0, [{"task_id": 0, "desc": "a", "success": True}])
    _shard(tmp_path, 1, [{"task_id": 1, "desc": "b", "success": False},
                         {"task_id": 1, "desc": "b", "success": True}])
    assert [r["task_id"] for r in read_episodes(tmp_path)] == [0, 1, 1]


def test_per_task_folds_by_task_id_in_order():
    rows = per_task([{"task_id": 3, "desc": "c", "success": True},
                     {"task_id": 1, "desc": "a", "success": False},
                     {"task_id": 3, "desc": "c", "success": False}])
    assert [(r["task_id"], r["successes"], r["n"], r["rate"]) for r in rows] == [
        (1, 0, 1, 0.0), (3, 1, 2, 50.0)]


def test_write_summary_records_identity_and_totals(tmp_path):
    results, logs = tmp_path / "results", tmp_path / "logs"
    _shard(results, 0, [{"task_id": 0, "desc": "a", "success": True},
                        {"task_id": 0, "desc": "a", "success": False}])
    out = write_summary(results, logs, policy="pi05", protocol="openpi")
    doc = json.loads(out.read_text())
    assert out.name == SUMMARY_JSON
    assert (doc["policy"], doc["protocol"]) == ("pi05", "openpi")
    assert (doc["successes"], doc["n"], doc["rate"]) == (1, 2, 50.0)
    assert doc["per_task"][0]["desc"] == "a"
    assert doc["log_dir"] == str(logs)


def test_format_table_ends_with_the_total():
    table = format_table(per_task([{"task_id": 0, "desc": "put the bowl", "success": True},
                                   {"task_id": 0, "desc": "put the bowl", "success": False}]))
    lines = table.splitlines()
    assert lines[0].startswith("  task  0    1/2    50.0%  put the bowl")
    assert lines[-1].startswith("  total      1/2    50.0%")


def test_format_table_is_empty_without_rows():
    assert format_table([]) == ""


def test_shards_partition_every_episode_exactly_once_and_evenly():
    n_tasks, trials, shards = 10, 50, 24
    plans = [shard_episodes(n_tasks, trials, k, shards) for k in range(shards)]
    flat = sorted(pair for plan in plans for pair in plan)
    assert flat == [(t, e) for t in range(n_tasks) for e in range(trials)]
    sizes = {len(plan) for plan in plans}
    assert max(sizes) - min(sizes) <= 1


def test_a_shard_is_one_block_of_the_task_major_list():
    assert shard_episodes(3, 4, 1, 2) == [(1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]


def test_a_deal_of_an_arbitrary_episode_list_splits_it_in_order():
    pairs = [(0, 3), (2, 1), (2, 7), (9, 0), (9, 4)]
    assert deal_episodes(pairs, 0, 2) == [(0, 3), (2, 1), (2, 7)]
    assert deal_episodes(pairs, 1, 2) == [(9, 0), (9, 4)]


def test_pending_episodes_are_the_grid_no_record_covers(tmp_path):
    shard = tmp_path / "shard0_of_1"
    shard.mkdir()
    (shard / "episodes.jsonl").write_text(
        "".join(json.dumps({"task_id": t, "episode": e, "success": True}) + "\n"
                for t, e in [(0, 0), (0, 1), (1, 1)]))
    assert pending_episodes(tmp_path, 2, 2) == [(1, 0)]


def test_pending_episodes_of_an_untouched_cell_are_the_whole_grid(tmp_path):
    assert pending_episodes(tmp_path, 2, 2) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_a_shard_stays_inside_as_few_tasks_as_its_share_allows():
    shards = 28
    plans = [shard_episodes(10, 10, k, shards) for k in range(shards)]
    assert sum(len({t for t, _ in plan}) for plan in plans) <= 10 + shards


def test_one_shard_runs_everything():
    assert shard_episodes(2, 2, 0, 1) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_more_shards_than_episodes_leaves_some_idle():
    assert shard_episodes(1, 2, 5, 8) == []


def test_shard_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        shard_episodes(1, 1, 2, 2)


def test_video_writer_encodes_in_the_background_and_waits_on_close(tmp_path):
    import numpy as np
    frames = [np.zeros((16, 16, 3), np.uint8) for _ in range(4)]
    with VideoWriter("all", max_pending=1) as videos:
        assert videos.keep()
        assert videos.write(tmp_path / "a.mp4", frames, solved=True)
        assert videos.write(tmp_path / "b.mp4", frames, solved=False)
    assert (tmp_path / "a.mp4").stat().st_size > 0
    assert (tmp_path / "b.mp4").stat().st_size > 0


def test_video_writer_failures_mode_skips_solved_episodes(tmp_path):
    import numpy as np
    frames = [np.zeros((16, 16, 3), np.uint8)]
    with VideoWriter("failures") as videos:
        assert videos.keep()
        assert not videos.write(tmp_path / "ok.mp4", frames, solved=True)
        assert videos.write(tmp_path / "bad.mp4", frames, solved=False)
    assert not (tmp_path / "ok.mp4").exists() and (tmp_path / "bad.mp4").exists()


def test_video_writer_none_mode_collects_nothing(tmp_path):
    with VideoWriter("none") as videos:
        assert not videos.keep()
        assert not videos.write(tmp_path / "x.mp4", [], solved=False)
    assert list(tmp_path.iterdir()) == []


def test_video_mode_reads_the_environment(monkeypatch):
    monkeypatch.delenv("ROBOT_OOD_VIDEO", raising=False)
    assert video_mode() == "none"
    monkeypatch.setenv("ROBOT_OOD_VIDEO", "failures")
    assert video_mode() == "failures"
    monkeypatch.setenv("ROBOT_OOD_VIDEO", "sometimes")
    with pytest.raises(ValueError):
        video_mode()


def test_the_scorer_counts_records_before_videos(tmp_path):
    from robot_ood.eval.scoring import ScoreModule
    scorer = ScoreModule(repo_root() / "evals" / "libero_plus" / "score.py")
    _shard(tmp_path, 0, [{"task_id": 0, "desc": "a", "success": True},
                         {"task_id": 0, "desc": "a", "success": False}])
    (tmp_path / "shard0_of_2" / "rollout_t0_episode0_success.mp4").write_bytes(b"x")
    count = scorer.count_dir(tmp_path)
    assert (count.successes, count.n) == (1, 2)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "rollout_t0_episode0_success.mp4").write_bytes(b"x")
    (legacy / "rollout_t0_episode1_failure.mp4").write_bytes(b"x")
    count = scorer.count_dir(legacy)
    assert (count.successes, count.n) == (1, 2)


def test_an_episode_trace_keeps_every_step_and_check_per_arm(tmp_path):
    import numpy as np
    trace = EpisodeTrace(["left", "right"])
    for step in range(3):
        trace.step(step, {"left": np.full(7, step), "right": np.zeros(7)},
                   {"left": np.ones(7), "right": np.ones(7)}, {"left": np.full(7, 2.0), "right": np.ones(7)})
    trace.check(0, "left", 0.25, 17, 0.9, {"trust": 0.4, "shape": 0.7, "z_chk": np.arange(16.0)})
    trace.check(10, "left", 0.5, 18, 0.8, {"trust": 0.5, "shape": 0.6})
    path = trace.save(tmp_path / f"{TRACE_PREFIX}t0_episode0.npz")
    z = np.load(path)
    assert z["steps"].tolist() == [0, 1, 2] and list(z["arms"]) == ["left", "right"]
    assert z["q_left"].shape == (3, 7) and z["q_left"][2, 0] == 2.0
    assert (z["a_exec_left"] - z["a_policy_left"]).tolist() == [[1.0] * 7] * 3
    assert z["check_step_left"].tolist() == [0.0, 10.0] and z["cand_idx_left"].tolist() == [17.0, 18.0]
    assert z["phase_right"].shape == (0,)
    assert z["shape_left"] == pytest.approx([0.7, 0.6], rel=1e-6) and z["trust_left"] == pytest.approx([0.4, 0.5], rel=1e-6)
    assert z["z_chk_left"].shape == (2, 16) and z["z_chk_left"][0].tolist() == list(range(16))
    assert np.isnan(z["z_chk_left"][1]).all()          # a check without features: a nan row
