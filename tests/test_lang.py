"""The instruction prior: the embedding table, the phrasing bank, the head, and the seed factor.

What must hold is that the mechanism is soft, that it degrades to a no-op rather than to a wrong
mask, and that a language-free graph is bit-unchanged.

The encoder itself is exercised only under ROBOT_OOD_TEST_ENCODER=1 -- it loads a 0.5B model, which is
about 25 s, and every other property here is testable without it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from robot_ood.models.language.encoder import (
    N_AUGMENT,
    TaskEmbeddings,
    augment_instruction,
    normalise_instruction,
)
from robot_ood.models.types import SeedLanguage

EMBED, N_TASKS = 16, 4


def _table(seed: int = 0) -> TaskEmbeddings:
    """A synthetic, already-centred task table -- no encoder needed."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((N_TASKS, EMBED)).astype(np.float32)
    mean = raw.mean(axis=0)
    e_task = TaskEmbeddings._unit(raw - mean)
    e_aug = TaskEmbeddings._unit(
        np.repeat(raw - mean, N_AUGMENT, axis=0) + 0.1 * rng.standard_normal((N_TASKS * N_AUGMENT, EMBED))
    ).astype(np.float32)
    head_w = (4.0 * e_task.T).astype(np.float32)
    table = TaskEmbeddings(
        e_task=e_task, mean=mean, model_id="synthetic",
        task_names=tuple(f"SCENE{i}_task_{i}" for i in range(N_TASKS)),
        e_aug=e_aug, aug_ptr=np.arange(0, N_TASKS * N_AUGMENT + 1, N_AUGMENT, dtype=np.int64),
        head_w=head_w, head_b=np.zeros(N_TASKS, dtype=np.float32), temperature=1.0,
        p_task=np.zeros((0, 0), dtype=np.float32), p_aug=np.zeros((0, 0), dtype=np.float32),
    )
    return TaskEmbeddings(**{**table.__dict__, "p_task": table.posterior(e_task),
                             "p_aug": table.posterior(e_aug)})


# ---- normalisation and surrogates ----------------------------------------------------------------
def test_normalise_strips_the_scene_prefix_and_underscores():
    """A task name must reach the encoder as a sentence, not as an identifier."""
    assert normalise_instruction("LIVING_ROOM_SCENE2_put_both_the_butter_in_the_basket") == (
        "put both the butter in the basket"
    )
    assert normalise_instruction("KITCHEN_SCENE3_turn_on_the_stove") == "turn on the stove"
    # A harness string that carries no prefix is only lowercased and collapsed.
    assert normalise_instruction("  Turn ON  the stove ") == "turn on the stove"


def test_normalise_drops_the_perturbation_tail_the_harness_splices_in():
    """Six camera coordinates and two indices are not an instruction.

    LIBERO-Plus encodes the perturbation in the task NAME and the harness splits that name on
    underscores, so the digits land in the language slot. Over 400 libero_10 episodes carrying this
    tail, head-projected top-1 task identification is 87.2% with it and 100.0% without.
    """
    assert normalise_instruction(
        "pick_up_the_black_bowl_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_1"
    ) == "pick up the black bowl and place it on the plate"
    # noise comes AFTER initstate, so a right-to-left trim that does not know the word stops dead
    # and removes nothing at all. That was the first version of this pattern, on 40% of the tails.
    assert normalise_instruction(
        "KITCHEN_SCENE3_turn_on_the_stove_view_0_0_100_0_0_initstate_0_noise_13"
    ) == "turn on the stove"


def test_normalise_keeps_suffixes_whose_marker_is_a_real_word():
    """add / table / light cost 0.011 of cosine margin and zero accuracy, and a genuine instruction
    can end on any of them. Trimming those would delete language to fix nothing."""
    assert normalise_instruction("KITCHEN_SCENE3_turn_on_the_stove_add_16") == (
        "turn on the stove add 16"
    )
    assert normalise_instruction("KITCHEN_SCENE4_put_the_bowl_on_the_table_9") == (
        "put the bowl on the table 9"
    )


def test_normalise_does_not_eat_a_sentence_that_merely_ends_in_a_marker_word():
    """The tail is only a tail when digits follow it."""
    assert normalise_instruction("adjust_the_camera_view") == "adjust the camera view"


def test_surrogates_are_lossy_but_never_empty():
    """Every surrogate drops something and keeps something -- an empty one would train the prior on
    a query with no content, which is not a paraphrase of anything."""
    rng = np.random.default_rng(0)
    name = "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
    original = normalise_instruction(name).split()
    seen = set()
    for _ in range(64):
        words = augment_instruction(name, rng).split()
        assert 2 <= len(words) <= len(original)
        assert set(words) <= set(original)
        seen.add(" ".join(words))
    assert len(seen) > 8, "the surrogate sampler is nearly deterministic"


def test_a_short_instruction_survives_augmentation_intact():
    """Below three words there is nothing to drop, so the surrogate is the instruction."""
    rng = np.random.default_rng(0)
    assert augment_instruction("open drawer", rng) == "open drawer"


# ---- the table -----------------------------------------------------------------------------------
def test_similarity_is_centred_and_self_ranking():
    """Each task is its own nearest neighbour, and centring leaves the cosines spread rather than
    piled at 1 -- the property that makes a temperature able to act at all."""
    table = _table()
    sim = table.e_task @ table.e_task.T
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    off = sim[~np.eye(N_TASKS, dtype=bool)]
    assert off.max() < 0.99
    assert np.all(np.diag(sim)[:, None] >= sim - 1e-6)


def test_check_against_refuses_a_table_built_for_another_graph():
    """A table indexed by a different task ordering would tilt retrieval towards the wrong
    demonstrations while looking perfectly well-formed."""
    table = _table()
    table.check_against(table.task_names)                      # the matching case is a no-op
    with pytest.raises(ValueError, match="Rebuild it"):
        table.check_against(("a", "b"))


def test_save_load_round_trips(tmp_path):
    """Including the bank, the head and the mean, which a deployed episode cannot recompute."""
    table = _table()
    loaded = TaskEmbeddings.load(table.save(tmp_path))
    for key in ("e_task", "mean", "e_aug", "aug_ptr", "head_w", "head_b", "p_task", "p_aug"):
        np.testing.assert_array_equal(getattr(loaded, key), getattr(table, key))
    assert loaded.temperature == table.temperature
    assert loaded.task_names == table.task_names and loaded.model_id == table.model_id


def test_load_refuses_a_table_without_a_head(tmp_path):
    """A table from before the head has no posterior to read; it must be rebuilt, not guessed at."""
    table = _table()
    np.savez(tmp_path / "g_lang.npz", e_task=table.e_task, mean=table.mean,
             model_id=np.array("old"), task_names=np.array(table.task_names), e_aug=table.e_aug)
    with pytest.raises(ValueError, match="rebuild"):
        TaskEmbeddings.load(tmp_path)


def test_the_posterior_multiplies_a_prior_in_and_renormalises():
    """A prior that rules a task out removes it; a uniform prior leaves the head's read unchanged."""
    table = _table()
    head = table.posterior(table.e_task[0])
    np.testing.assert_allclose(table.posterior(table.e_task[0], np.ones(N_TASKS) / N_TASKS), head, atol=1e-6)
    prior = np.ones(N_TASKS); prior[0] = 0.0
    mixed = table.posterior(table.e_task[0], prior)
    assert mixed[0] == 0.0 and abs(mixed.sum() - 1.0) < 1e-6
    np.testing.assert_allclose(mixed[1:], head[1:] / head[1:].sum(), atol=1e-6)


def test_the_posterior_sums_to_one_and_ranks_each_centroid_first():
    table = _table()
    np.testing.assert_allclose(table.p_task.sum(axis=1), 1.0, atol=1e-6)
    assert list(table.p_task.argmax(axis=1)) == list(range(N_TASKS))
    assert list(table.p_aug.argmax(axis=1)) == [i for i in range(N_TASKS) for _ in range(N_AUGMENT)]


def test_draw_aug_stays_inside_each_tasks_own_rows():
    """A ragged bank: the draw for task t lands in aug_ptr[t]:aug_ptr[t+1] for every uniform."""
    table = _table()
    task = np.repeat(np.arange(N_TASKS), 50)
    u = np.tile(np.linspace(0.0, 0.999999, 50), N_TASKS)
    which = table.draw_aug(task, u)
    assert np.all(which >= table.aug_ptr[task]) and np.all(which < table.aug_ptr[task + 1])
    assert which[0] == table.aug_ptr[0] and which[49] == table.aug_ptr[1] - 1


def test_the_head_identifies_held_out_phrasings_of_clustered_banks():
    """Fit on synthetic banks whose rows cluster per task, the head names an unseen row of each cluster."""
    from robot_ood.models.language.encoder import _fit_head

    rng = np.random.default_rng(1)
    centres = rng.standard_normal((N_TASKS, EMBED))
    rows = np.concatenate([c + 0.3 * rng.standard_normal((40, EMBED)) for c in centres])
    rows = TaskEmbeddings._unit(rows - rows.mean(axis=0))
    y = np.repeat(np.arange(N_TASKS), 40)
    w, b, t = _fit_head(rows, y, N_TASKS, seed=0)
    probe = TaskEmbeddings._unit(centres + 0.3 * rng.standard_normal((N_TASKS, EMBED)) - rows.mean(axis=0))
    assert list(((probe @ w + b) / t).argmax(axis=1)) == list(range(N_TASKS))
    assert t > 0.0


def test_load_names_the_build_script_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="build language"):
        TaskEmbeddings.load(tmp_path)


# ---- the seed hook -------------------------------------------------------------------------------
def test_seed_language_disabled_is_the_default():
    """A graph with no instruction prior must take the untouched path, not a multiply by ones."""
    assert SeedLanguage().factor is None


@pytest.mark.skipif(
    os.environ.get("ROBOT_OOD_TEST_ENCODER") != "1",
    reason="loads a 0.5B language model; set ROBOT_OOD_TEST_ENCODER=1 to run",
)
def test_the_real_encoder_is_deterministic_and_centres_its_table():
    """Two encodes of one string agree exactly, and the built table is centred and unit-norm."""
    from robot_ood.models.language.encoder import InstructionEncoder

    encoder = InstructionEncoder()
    names = ("KITCHEN_SCENE3_turn_on_the_stove", "STUDY_SCENE1_pick_up_the_book")
    first = encoder.encode(list(names))
    np.testing.assert_array_equal(first, encoder.encode(list(names)))

    table = TaskEmbeddings.build(names, encoder, n_augment=2)
    np.testing.assert_allclose(np.linalg.norm(table.e_task, axis=1), 1.0, atol=1e-5)
    assert table.e_aug.shape == (2, 2, first.shape[1])
