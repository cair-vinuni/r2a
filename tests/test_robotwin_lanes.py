"""RoboTwin's joint layout as two lanes: targets, rows and the open flag round-trip."""

from __future__ import annotations

import numpy as np

from robot_ood.data.robotwin import ARMS, JOINT_DIM, joint_rows, lane_step_row, lane_targets, open_flag


def test_lane_targets_split_the_joint_vector_and_rows_rejoin_it():
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-3, 3, (50, JOINT_DIM))
    chunk[:, [6, 13]] = rng.uniform(0, 1, (50, 2))
    targets = {arm: lane_targets(chunk, arm)[0] for arm in ARMS}
    grips = {arm: lane_targets(chunk, arm)[1] for arm in ARMS}
    assert targets["left"].shape == (50, 6) and np.array_equal(targets["right"], chunk[:, 7:13])
    assert np.array_equal(grips["left"], chunk[:, 6])
    assert np.array_equal(joint_rows(targets, grips), chunk)


def test_open_flag_counts_a_gripper_stopped_by_an_object_as_closed():
    assert open_flag(np.array([0.0, 0.5, 0.6, 0.79, 0.8, 1.0])).tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]



def test_a_lane_step_row_moves_its_own_arm_by_the_deltas_and_holds_the_other():
    states = {"left": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05], "right": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.95]}
    row = lane_step_row(states, "right", np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.1, -1.0]))
    assert np.allclose(row[:7], states["left"])
    assert np.allclose(row[7:13], [1.01, 1.08, 1.23, 1.3, 1.4, 1.6]) and row[13] == 1.0
    assert lane_step_row(states, "left", np.array([0.0] * 6 + [1.0]))[6] == 0.0


def test_a_lane_corpus_records_the_tasks_seen_phrasings(tmp_path):
    """The demo file carries every distinct seen phrasing of the download, and reads back as the task's bank."""
    import json

    import numpy as np

    from robot_ood.data.libero import task_phrasings, write_demo_file
    from robot_ood.data.robotwin import DEMO_SET, task_phrasings as pool_of

    task_dir = tmp_path / "place_empty_cup"
    instr = task_dir / DEMO_SET / "instructions"
    instr.mkdir(parents=True)
    (instr / "episode0.json").write_text(json.dumps({"seen": ["lift the cup", "grab the cup"], "unseen": ["x"]}))
    (instr / "episode1.json").write_text(json.dumps({"seen": ["grab the cup", "raise the cup"], "unseen": ["y"]}))
    pool = pool_of(task_dir)
    assert pool == ["lift the cup", "grab the cup", "raise the cup"]
    out = tmp_path / "place_empty_cup_demo.hdf5"
    demos = iter([{"actions": np.zeros((3, 7), dtype=np.float32), "joint_states": np.zeros((3, 7), dtype=np.float32)}])
    write_demo_file(out, "place empty cup", demos, phrasings=pool)
    assert task_phrasings(out) == pool
    bare = tmp_path / "bare_demo.hdf5"
    write_demo_file(bare, "bare task", iter([{"actions": np.zeros((2, 7), dtype=np.float32)}]))
    assert task_phrasings(bare) == ["bare task"]
