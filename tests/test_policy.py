"""Policy descriptors: evals/policies/<id>.yaml and their consumers."""

import pytest

from robot_ood.eval.policy import DEFAULT_POLICY, Placement, load_policy, policy_ids


def test_the_shipped_policies_are_declared():
    assert set(policy_ids()) >= {"stablevla", "gr00t", "pi05"}


def test_stablevla_descriptor_matches_the_deployed_launch():
    p = load_policy("stablevla")
    assert p.conda_env == "stablevla"
    assert p.ckpt_key == "stablevla_ckpt"
    argv = p.server_argv(checkpoint="/x/ckpt", port=14190)
    assert argv[0] == "evals/stablevla/stablevla_policy_server.py"
    assert "--ckpt_path" in argv and "/x/ckpt" in argv and "14190" in argv


def test_gr00t_descriptor_matches_the_deployed_launch():
    p = load_policy("gr00t")
    assert p.ckpt_key == "gr00t_ckpt"
    assert p.python is not None and p.python.name == "python"
    argv = p.server_argv(checkpoint="/x/ckpt", port=14201)
    assert argv[0] == "evals/gr00t/ws_server.py"
    assert "--model-path" in argv and "/x/ckpt" in argv and "14201" in argv


def test_none_resolves_to_the_default(monkeypatch):
    monkeypatch.delenv("ROBOT_OOD_POLICY", raising=False)
    assert load_policy(None).id == DEFAULT_POLICY


def test_an_unknown_policy_names_the_known_ones():
    with pytest.raises(FileNotFoundError, match="stablevla"):
        load_policy("nonesuch")


def test_each_policy_names_its_authors_protocol():
    from robot_ood.core.protocol import get_protocol
    assert load_policy("stablevla").protocol == "openvla_oft"
    assert load_policy("gr00t").protocol == "nvidia"
    assert load_policy("pi05").protocol == "openpi"
    for pid in policy_ids():
        get_protocol(load_policy(pid).protocol)


def test_pi05_descriptor_runs_openpi_in_its_own_venv():
    p = load_policy("pi05")
    assert p.ckpt_key == "pi05_ckpt"
    assert p.python is not None and "openpi" in str(p.python)
    argv = p.server_argv(checkpoint="/x/ckpt", port=14300)
    assert argv[0] == "evals/pi05/ws_server.py"
    assert "--checkpoint" in argv and "/x/ckpt" in argv and "14300" in argv


def test_each_policy_declares_its_placement():
    for pid, servers_per_gpu, clients_per_server in (("gr00t", 2, 4), ("pi05", 1, 8), ("stablevla", 4, 2)):
        placement = load_policy(pid).placement
        assert (placement.servers_per_gpu, placement.clients_per_server) == (servers_per_gpu, clients_per_server), pid


def test_placement_scales_with_the_gpus_a_cell_gets():
    p = Placement(servers_per_gpu=2, clients_per_server=3)
    assert (p.servers(4), p.clients(4)) == (8, 24)
    assert (p.servers(1), p.clients(1)) == (2, 6)
    assert Placement.from_doc(None) == Placement()


def test_placement_rejects_zero_counts():
    with pytest.raises(ValueError):
        Placement.from_doc({"servers_per_gpu": 0})


def test_batching_servers_are_told_their_batch():
    assert "--max-batch" in load_policy("gr00t").server_argv(checkpoint="/x", port=1)
    assert "--max-batch" in load_policy("pi05").server_argv(checkpoint="/x", port=1)
