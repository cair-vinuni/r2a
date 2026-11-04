# Artifacts

**Code:** `src/robot_ood/data/schema.py` (filenames), `src/robot_ood/data/geometry.py`
(`graph_hash`), `src/robot_ood/data/store.py` (manifest, gates), `src/robot_ood/core/paths.py`
**Build order:** `robot_ood.pipeline.dag`

One artifact directory — `outputs/<suite>/artifacts/` — is the contract between the offline side
and eval. Everything the method loads lives there, the C0 calibration included (embedded in
`g_fcsr.npz`). `GR_GRAPH_DIR` overrides the location; `base` mode loads none of it.

## Runtime artifacts

| artifact | step | consumer | what it is |
|---|---:|---|---|
| `g_nodes.npz` | 02 | entry and sentinel | graph nodes, raw trajectories, entry-state cloud, `ee_base` |
| `g_edges.npz` | 02 | sentinel | graph edges plus `log_idf` |
| `g_segments.npz` | 04 | sentinel | demonstrated segment memory |
| `action_scale.json` | 05 | sentinel | metres and radians per action unit; absence raises at launch |
| `g_lang.npz` | 06 | sentinel | task-instruction embeddings and the offline task posteriors |
| `g_head.npz` | 09 | sentinel | the locator GNN the belief comes out of; carries the embedded phase-skew report |
| `g_score.npz` | 13 | entry | entry score field and level head (`ScoreNet.load` rejects a checkpoint with no `level.*` keys) |
| `g_invdyn.npz` | 13 | entry | inverse-dynamics net |
| `g_fcsr.npz` | 14b | sentinel | one archive, three prefixes: `seg.` query encoder, `scorer.` C2 rescore, `corrector.` trust and shape heads; key `constants_json` embeds the C0 calibration |

## Build-only artifacts

| artifact | step | consumer | purpose |
|---|---:|---|---|
| `a_pi_raw.npz` | 07 | FCSR training | the frozen policy's action chunks over the corpus |
| `a_pi_raw.manifest.json` | 07 | `cache-policy` resume | which corpus windows are already cached |
| `run_config.json` | 09 | `eval` (`meta.policy`), humans | snapshot of the training run; never a config source |
| `ARTIFACTS.json` | 02 onward | `artifacts check`, `seed` | per-file provenance manifest |

The step column is the `robot_ood.pipeline.dag` id; `build --dry-run` and `train --dry-run` print
the steps in this order.

Keep a loose `constants.json` out of `artifacts/`: `graph_hash` keys on a file of exactly that
name inside the directory, so dropping one in changes every future stamp and orphans fitted
checkpoints. `ROBOT_OOD_CONSTANTS` points at a calibration file elsewhere;
`build constants` exports one and refuses to write into a directory holding `g_nodes.npz`.

## Compatibility: `graph_hash`

`graph_hash(nodes, edges)` (`robot_ood.data.geometry`) is a SHA-256 over the graph dimensions and,
in order: node `q`, `qdot`, `owner`, `task_id`, `stage`, `t_idx`, `phase`; edge `src`, `dst`,
`rel`, `indptr`. Derived arrays (`w`, `log_idf`) are excluded. Node ids are positional, so loaders
reject a missing or mismatched stamp before touching weights.

Stamped files group into families (`robot_ood.data.store.HASH_FAMILIES`); agreement is required
within a family only:

| family | files |
|---|---|
| `base` | `g_segments.npz` |
| `head` | `g_head.npz` |
| `fcsr` | `g_fcsr.npz` |

`action_scale.json`, `g_lang.npz` and `a_pi_raw.npz` carry no stamp.

## Manifest and stage gates

`ARTIFACTS.json` records each file's sha256, size, mtime, producing command, `graph_hash` and git
commit (`robot_ood.data.store.record_artifacts`). Validate a directory per stage:

```bash
python -m robot_ood.cli artifacts check <graph_dir> --stage <build|locator|entry|fcsr|deploy>
```

| stage | requires |
|---|---|
| `build` | `g_nodes.npz`, `g_edges.npz`, `g_segments.npz`, `action_scale.json`, `g_lang.npz`, `a_pi_raw.npz` |
| `locator` | `g_head.npz` |
| `entry` | `g_score.npz`, `g_invdyn.npz` |
| `fcsr` | `g_fcsr.npz` |
| `deploy` | the union of the four |

`deploy` is the gate the shipping arm must pass; `eval --mode fcsr` and `artifacts push` run it
before launching or uploading, and the pipeline's final step is exactly this check. The other four
check one stage's own output, so a failure names the stage to rerun — every problem line ends with
the regenerating command.

A hand-copied file has no manifest record and fails verification. `artifacts seed <src> <dst>`
clones a directory with its records (withholding what a new FCSR run exists to write);
`build graph` seeds the canonical directory itself.

`validate` refuses any path containing a `latest` component: `outputs/<suite>/latest` is a symlink
repointed by every new build, so gates and evals pin the explicit `artifacts/` path.

## Run outputs

| file | producer | purpose |
|---|---|---|
| `<run>/config.json`, `manifest.json`, `metrics.json`, `log.txt` | `StageLogger` | build config and metrics per run directory |
| `<result_dir>/run.json` | `scripts/run_sr.py` | eval provenance: policy, mode, `artifact_digest`, `run_config_sha`, git commit, child env |
| `outputs/sr/ledger.jsonl` | `scripts/run_sr.py` | append-only eval ledger |
| `tests/golden/parity.json` | `tests/capture_golden.py` | expected digests for the bit-exact parity test |

## Rebuilding safely

Do not rebuild a directory another run is reading; build into a new tree, gate it, then repoint.
`graph_hash` catches graph/checkpoint mismatches but not refitting `g_head.npz` against the same
graph; `tests/test_parity.py` catches that case.
