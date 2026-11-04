from __future__ import annotations

import pytest

from robot_ood.core.config import ABLATIONS, Ablation, SentinelConfig


def test_no_ablation_by_default(monkeypatch):
    monkeypatch.delenv("ROBOT_OOD_ABLATE", raising=False)
    assert Ablation.from_env() == Ablation()
    assert str(Ablation()) == "none"


def test_each_switch_parses_by_name(monkeypatch):
    for name in ABLATIONS:
        monkeypatch.setenv("ROBOT_OOD_ABLATE", name)
        assert Ablation.from_env() == Ablation(**{name: True})


def test_switches_combine_and_print_in_canonical_order(monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_ABLATE", " corrector, graph ")
    assert Ablation.from_env() == Ablation(graph=True, corrector=True)
    assert str(Ablation.from_env()) == "graph,corrector"


def test_an_unknown_switch_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_ABLATE", "graph,belief")
    with pytest.raises(ValueError, match="belief"):
        Ablation.from_env()


def test_the_sentinel_config_carries_the_ablation(monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_ABLATE", "entry")
    assert SentinelConfig.from_env().ablation == Ablation(entry=True)


def test_the_entry_corrector_is_off_under_its_ablation(tmp_path, monkeypatch):
    import robot_ood.runtime.harness as harness
    from robot_ood.runtime.harness import DeployHooks

    monkeypatch.setenv("MODE", "fcsr")
    monkeypatch.setenv("ROBOT_OOD_ABLATE", "entry")
    monkeypatch.setattr(harness.DeployHooks, "_load_entry", classmethod(lambda cls, suite: pytest.fail("entry loaded")))
    monkeypatch.setattr(harness.DeployHooks, "_load_sentinel", staticmethod(lambda cfg: "sentinel"))
    hooks = DeployHooks.from_env("object")
    assert hooks.entry is None and hooks.sentinel == "sentinel"

    monkeypatch.setattr(harness.DeployHooks, "_entry_for", staticmethod(lambda d: pytest.fail("entry loaded")))
    monkeypatch.setattr("robot_ood.data.segments.graph_plan_rows", lambda artifacts, chunk: 5)
    monkeypatch.setattr("robot_ood.runtime.sentinel.Sentinel", lambda cfg, fwm_dir: "sentinel")
    assert DeployHooks.for_graph("", tmp_path, 50).entry is None
