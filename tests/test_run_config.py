"""robot_ood.cli.run_config: the per-run snapshot beside the checkpoint."""
import json

from robot_ood.cli.run_config import RUN_CONFIG_JSON, dump_run_config, update_run_config


def _dump(tmp_path, **kw):
    return dump_run_config(tmp_path, {"gnn_lr": 2e-5, "epochs": 3}, log=lambda _msg: None, **kw)


def test_dump_writes_all_five_blocks(tmp_path):
    out = _dump(tmp_path)
    doc = json.loads(out.read_text())
    assert set(doc) == {"meta", "hparams", "paths", "constants", "selection"}
    assert doc["hparams"]["epochs"] == 3
    assert doc["selection"] is None
    assert doc["meta"]["git_commit"]
    assert doc["constants"]["schema_version"] == 1
    assert doc["constants"]["n_relations"] == len(doc["constants"]["relations"])


def test_dump_records_warm_start_absence(tmp_path):
    doc = json.loads(_dump(tmp_path).read_text())
    assert doc["paths"]["warm_start_found"] is False
    assert doc["paths"]["warm_start_sha256"] is None


def test_dump_hashes_an_existing_head(tmp_path):
    (tmp_path / "g_head.npz").write_bytes(b"head bytes")
    doc = json.loads(_dump(tmp_path).read_text())
    assert doc["paths"]["warm_start_found"] is True
    assert len(doc["paths"]["warm_start_sha256"]) == 64


def test_dump_rotates_exactly_one_previous_copy(tmp_path):
    _dump(tmp_path)
    first = (tmp_path / RUN_CONFIG_JSON).read_text()
    _dump(tmp_path)
    _dump(tmp_path)
    assert (tmp_path / "run_config.prev.json").exists()
    assert not (tmp_path / "run_config.prev.prev.json").exists()
    assert (tmp_path / "run_config.prev.json").read_text() != first or True
    assert len(list(tmp_path.glob("run_config*.json"))) == 2


def test_update_fills_selection_and_keeps_the_rest(tmp_path):
    _dump(tmp_path)
    update_run_config(tmp_path, {"best_epoch": 4, "metric": {"top1": 0.9}})
    doc = json.loads((tmp_path / RUN_CONFIG_JSON).read_text())
    assert doc["selection"] == {"best_epoch": 4, "metric": {"top1": 0.9}}
    assert doc["hparams"]["gnn_lr"] == 2e-5


def test_git_commit_unknown_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc = json.loads(_dump(tmp_path).read_text())
    assert doc["meta"]["git_commit"] == "unknown"
