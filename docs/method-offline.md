# Offline: graph build and locator training

**Reads:** LIBERO demo HDF5, `configs/robot_ood.yaml` (+ `configs/trees/<suite>.yaml`)
**Writes:** `g_nodes.npz`, `g_edges.npz`, `g_head.npz`, `run_config.json`
**Code:** `src/robot_ood/build/graph.py`, `src/robot_ood/data/nodes.py`, `src/robot_ood/data/edges.py`,
`src/robot_ood/data/schema.py`, `src/robot_ood/models/retrieval/gnn.py`, `src/robot_ood/train/loop.py`,
`src/robot_ood/train/data.py`, `src/robot_ood/train/loss.py`, `src/robot_ood/train/tasks/locator.py`

```bash
python -m robot_ood.cli build --suite long                        # all stale build steps, in order
python -m robot_ood.cli build graph --suite long                  # just the graph: g_nodes + g_edges
python -m robot_ood.cli train locator --suite long --device cuda:0  # g_head.npz
```

Suites: `object | spatial | goal | long`. Bare `build --suite <s>` runs every stale build step in
order — graph, segments, calibrate, language, cache-policy — re-running a step when an input's hash
changed, not merely when an output is missing (`--dry-run` prints why). `build graph` writes into a
run directory under `outputs/` and then seeds the canonical artifact directory
`outputs/<suite>/artifacts` (`GR_GRAPH_DIR` overrides it); every later stage reads that one path.
Config resolves through `robot_ood.core.paths.Paths` from `configs/robot_ood.yaml` (roots, defaults,
axes; `$ROBOT_OOD_CONFIG` overrides the file, `$ROBOT_OOD_OUTPUTS` the output root) plus one
`configs/trees/<suite>.yaml` per artifact tree. See [configuration](configuration.md).

Terms: a **node** is a coarsened group of consecutive raw frames from one demo. A **strand** is one
demonstration (the node's `owner`). A **lane** is one task (`task_id`). **Phase** is normalized
progress through a demo in raw-frame space, in [0, 1].

Every downstream artifact indexes the node table positionally, so a rebuilt graph invalidates all of
them; each artifact carries a `graph_hash` stamp that loaders check ([artifacts](artifacts.md)).

---

## 1. Graph build

### 1.1 Input: LIBERO HDF5

`SUITE_HDF5_DIRS` in `src/robot_ood/data/libero.py` maps suite to HDF5 directory:
`object → libero_object`, `spatial → libero_spatial`, `goal → libero_goal`, `long → libero_10`
(NOT `libero_long`). Suite `long`: 10 task files × 50 demos = 500 demos.

Extracted per demo of `T` raw frames:

| field | shape | dtype | source |
|---|---|---|---|
| `q` | `[T, 7]` | f32 | joint positions (Panda, 7-DoF) |
| `qdot` | `[T, 7]` | f32 | finite difference of `q`, last row repeated |
| `grip` | `[T]` | u1 | `mean(\|obs/gripper_states\|) < 0.035` → `1` |
| `stage` | `[T]` | i2 | cumulative count of gripper releases |
| `task_id` | scalar | i2 | sort order of the HDF5 file |

**Gripper polarity: `grip = 1` means CLOSED** — the opposite of the raw `gripper_states` magnitude.
`measured_grip()` prefers the measured channel and falls back to the command channel
(`actions[:, 6] > 0.5`) only when it is absent. Every query-side gripper flag goes through
`grip_flag()` (`GRIP_OPEN_THR = 0.035`, defined once in `robot_ood.data.schema`), so polarity always
agrees with `NodeTable.grip`.

### 1.2 Coarsening: raw frames to nodes

`coarsen_segments` in `src/robot_ood/data/nodes.py` merges `COARSEN = 5` consecutive raw frames into
one node, cutting a node short at every gripper-state change — each node holds exactly one gripper
state. Vectorised, no Python loop over frames:

```python
changed[0] = True
changed[i] = grip[i] != grip[i - 1]
last_reset = np.maximum.accumulate(np.where(changed, idx, -1))
pos_in_run = idx - last_reset
is_new_node = (pos_in_run % coarsen) == 0
node_id = np.cumsum(is_new_node) - 1
```

Measured on the shipped `long` artifact: 28,476 nodes / 138,090 raw frames / 500 demos / 10 tasks —
4.849 raw frames per node, below `COARSEN = 5` because of the gripper-induced cuts. The belief
filter's transition kernel divides by this figure to convert a raw-frame advance into a node advance
([runtime](method-runtime.md)).

### 1.3 `g_nodes.npz` (`NodeTable`)

Per-node arrays, length `V`:

| name | shape | dtype | value | note |
|---|---|---|---|---|
| `q` | `[V, 7]` | f32 | `q_raw[start]` | the **first** frame of the node, not the mean |
| `qdot` | `[V, 7]` | f32 | `q_raw[end] − q_raw[start]` | net displacement, not a per-step delta |
| `grip` | `[V]` | u1 | `1 = closed` | §1.1 |
| `stage` | `[V]` | i2 | cumulative releases | diagnostic only; nothing reads it (§1.5) |
| `t_idx` | `[V]` | i4 | node index within its demo | resets to 0 per demo |
| `t_frac` | `[V]` | f4 | `t_idx / (n_nodes_in_demo − 1)` | **node**-space coordinate ∈ [0, 1] |
| `t_raw` | `[V]` | i4 | global raw offset of the node's first frame | |
| `n_members` | `[V]` | i2 | raw frames merged, 1..5 | |
| `owner` | `[V]` | i4 | demo id | the strand |
| `task_id` | `[V]` | i2 | task id | the lane |
| `phase` | `[V]` | f4 | `arange(T)/(T−1)` at the first frame | **raw**-space coordinate ∈ [0, 1] |
| `is_virtual` | `[V]` | u1 | synthesised-node mask | all zero when `virtual_per_entry = 0` (default); optional key, absent loads as all-real |

CSR pointers, length `n_demos + 1`: `demo_ptr` (node ranges per demo) and `raw_ptr` (the same in
raw-frame indices).

Full raw trajectories are retained:

| name | shape | dtype | why kept |
|---|---|---|---|
| `q_raw`, `qdot_raw` | `[Nraw, 7]` | f32 | so `SEG_K` (segment length) stays a runtime choice |
| `ee_raw` | `[Nraw, 6]` | f32 | `concat(obs/ee_pos, obs/ee_ori)` as recorded, **world frame**; orientation is robosuite's unnormalized axis-angle. Fusion instead reads the derived `NodeTable.ee_base` property, which FKs `q_raw` through `robot_ood.core.kinematics.PandaKinematics` into the **robot base** frame (the world position of the base changes per scene). This column is the ground truth the FK is checked against |
| `a_raw` | `[Nraw, 7]` | f32 | the demo's own action commands, verbatim; source of the `a_hist` query channel (§2.5) |

`NodeTable.segment(node, k)` and `NodeTable.ee_segment(node, k)` slice `k` raw frames forward from a
node's `t_raw`, edge-padded at the demo end. They are the only path from a node id to a trajectory;
both the belief readout and the segment memory go through them.

`t_frac` and `phase` are different coordinates and not interchangeable: `t_frac` is node-index-space,
the network's input via `psi(t_frac)`; `phase` is progress-space, the share of the episode's moving
frames already covered, used for binning and for comparing demos of unequal length. They diverge
because coarsening is not uniform, and because time spent holding still advances neither.

Invariants, raised on load: `phase_conv` (`"travelled(q)/travelled(T-1) full"`) is stamped into the
file and re-checked by `NodeTable.check_phase_convention()`; `NodeTable.check_coarsening()` checks
`qdot == q_raw[last] − q_raw[first]`. `tests/test_nodes.py` and `tests/test_virtual_nodes.py`
enforce these.

### 1.4 `g_edges.npz` (`EdgeSet`)

`R = 13` relations (`schema.N_RELATIONS`). `rel` is an `i1` index into this table; **the order is
load-bearing and APPEND-ONLY** — it is the row of the relation embedding table, so inserting a name
silently re-labels every stored edge after it. The belief transition kernel names a subset by string
(`schema.TRACK_KERNEL_RELATIONS`).

| rel index | name | meaning | fan-out | edges on `long` |
|---|---|---|---|---|
| 0–4 | `next1, next2, next4, next8, next16` | forward along one demo, dilated | 1 each | 27,976 / 27,476 / 26,476 / 24,476 / 20,476 |
| 5–9 | `prev1 … prev16` | backward, dilated | 1 each | mirror of the above |
| 10 | `sibling` | kNN: different demo, same task, same gripper state | `K_SIBLING = 8` | 314,436 |
| 11 | `align` | kNN: different task, same phase bin (`NBINS_ALIGN = 20`), same gripper state | `K_ALIGN = 4` | 199,368 |
| 12 | `approach` | virtual → real at the entry stratum | `VIRTUAL_PER_ENTRY = 0` | 0 on the default all-real graph |

767,564 edges total on `long`. `sibling` and `align` are symmetrised and deduplicated after the kNN,
so `k` is the fan-out before symmetrisation (realised mean degrees 11.04 and 7.00).

Dilation sets the receptive field: `reach = layers × max(dilations) × coarsen = 3 × 16 × 5 = 240` raw
frames (`GraphContract.reach_raw_frames`), just under `long`'s p50 demo length of 263. `next1` alone
would reach ~15 frames — short of a pre-grasp approach (~80 frames).

Storage is CSR sorted by `dst` with `indptr`, so `index_add_` coalesces collisions during message
passing. Also stored: `log_idf [V] f4`, an additive output-logit prior computed by `node_idf`
(`src/robot_ood/data/edges.py`) from sibling reachability. Edge weights `w [E] f4` come from the
shared learned pair metric `robot_ood.data.metrics.C2Metric` — a head carrying different C2 tensors
is a different graph even when the geometry is byte-identical, which is why C2's tensors are hashed
into `graph_hash`.

### 1.5 `stage` and `task_id` are not network inputs

The node feature vector is `[q(7), qdot(7), grip(1), psi(t_frac)(8)]`. `stage` stays out: the
query-dependent relation gate learns when to trust an `align` edge, and one misread teleop sample
shifts every later frame's cumulative count. `task_id` stays out: it would tie the input width to
the suite. Task membership reaches the network through the relations (`sibling` within-task,
`align` cross-task) and, downstream, through the belief's own task marginal, which tilts the
segment shortlist cut and fills its `task_affinity` slot. `tests/test_gnn.py` holds
what this buys: no parameter tensor has a dimension equal to `V`, parameter count does not move
with graph size, one state dict runs on two differently-sized graphs.

---

## 2. Locator training

`train locator` fits one network, `GraphRetrieverNet`, over check sequences
(`robot_ood.train.tasks.locator.LocatorTask` via `robot_ood.train.loop.train_graph`). It writes
`g_head.npz` beside the graph, dumps `run_config.json` there before epoch one
(`robot_ood.cli.run_config`), and after training replays the locator over held-out demos and
embeds the phase-skew report into `g_head.npz` (`robot_ood.eval.phase_skew.embed_report`) —
`train fcsr` samples its query phase from it. Bare `train --suite <s>` runs locator, entry, fcsr
(stale ones only), then the deploy gate.

Inputs beyond the node and edge tables, both required: `g_lang.npz` (`build language`) for the
instruction channel and `a_pi_raw.npz` (`cache-policy`) for the policy-chunk channel.

### 2.1 Runtime interface

```python
from robot_ood.runtime.locator import GraphLocator
locator = GraphLocator.load(graph_dir, cfg=cfg, device=device)
step = locator.step(q_hist, g_hist=g_hist, task=task)      # phase_hat, task_prob, belief views
locator.record_chunk(a_policy)                              # the chunk issued after this check
```

`load` reads `g_nodes.npz`, `g_edges.npz`, `g_head.npz` and `g_lang.npz`, and raises on a
`graph_hash` mismatch, since node ids are positional. The locator carries the belief between
checks as the network's memory ([runtime](method-runtime.md) §2).

### 2.2 Head artifact

`g_head.npz`: a flat npz stamped with `graph_hash`, holding normalization statistics (`mean`,
`std`), the parameter blocks for query encoder, learned seed (including C2), node encoder,
relation gate, update network, node readout, instruction projection (`instr_proj`, `instr_q`),
policy projection (`policy_proj`) and memory feature (`mem`), plus `layers` and `agg` as 0-d
scalars. No tensor is dimensioned by `V`. Written by `robot_ood.models.retrieval.gnn.save_checkpoint`.

### 2.3 Architecture

`GraphRetrieverNet` in `src/robot_ood/models/retrieval/gnn.py`: `hidden = 64`, `layers = 3` with
shared weights, `n_rel = 13`, `n_psi = 4`, `D = 7`. `prefix_dim = 2·7 + 1 = 15` (`[q, qdot, grip]`),
`node_in_dim = 15 + 2·4 = 23` (+ `psi(t_frac)`).

| module | shape | purpose |
|---|---|---|
| `enc` (QueryEncoder) | `Linear(15→64) ReLU Linear(64→64)`, out `Linear(64→64)` | window → `e_Q [64]` |
| `instr_proj` | `Linear(896→32)` | instruction → `u [32]`, shared by the query and every node's task |
| `instr_q` | `Linear(32→64)`, zero-init | `u` → added to `e_Q` |
| `policy_proj` | `Linear(56→64)`, zero-init | the policy chunk → added to `e_Q` |
| `seed_head` (LearnedSeed) | `tier1 Linear(64→64)`, `log_sigma`, `log_tau`, C2 MLP `Linear(8→32) ReLU Linear(32→1)` | two-tier seed, top `SEED_TOPK = 128` |
| `mem` | `Linear(1→64)` | the memory's per-node log mass → h⁰ on the memory-seeded nodes |
| `node` | `Linear(23→64) ReLU Linear(64→64)` | node features → `n_v [V, 64]`, cached once after load |
| `rel` (RelationGate) | `emb [13, 64]` + MLP `Linear(128→64) ReLU Linear(64→64)` | query-conditioned relation embeddings, recomputed every hop |
| `upd` | `Linear(192→64) ReLU Linear(64→64)` | node-state update |
| `out` | `Linear(133→64) ReLU Linear(64→1)` | per-node logit over `[h, n_v, move, cos_instr, has_instr, −‖chunk − action‖, has_policy]` |

### 2.4 Forward pass

1. **Encode the query window** to `e_Q`; add the projected instruction and policy chunk, each
   times its presence flag.
2. **Encode graph nodes** once and cache `n_v`.
3. **Seed**: the two-tier learned seed gives amplitudes `a` on its top-128 nodes; `h⁰ = a·e_Q·n_v`.
   With a memory, its top-128 nodes and their successors along every `next` relation join the seed
   with `h⁰ += mem(log memory)·n_v`. The seed mask is the initial `reached`.
4. **Propagate**, 3 hops, shared weights, `reached` tracked explicitly.
5. **Read out**: `logits = out([h_L, n_v, move, extra]) + log_idf`, unreached nodes `−inf`;
   `belief = softmax(logits)`, `phase_hat = Σ belief · phase`, `task_prob = Σ belief by task`.

### 2.5 Training rows

`robot_ood.train.sequences.SequenceSampler` draws rows of `n_checks ~ U{4..8}` checks on one
demo, `CHECK_EVERY = 8` raw frames apart. Each check is a window at its frame under one of four
modes (`traversal .48`, `entry .28`, `entry_static .16`, `drift .08`, the same constructions as
before), a policy chunk (the cached policy's frames before the check, corrupted by
`PolicyCorruptor`, absent with probability `.2`), and its own true node. The row's instruction is
`exact .3 / paraphrase .4 / none .3`. Memory modes and the curriculum: [runtime](method-runtime.md)
§2.3.

### 2.6 Label and loss

One label rule, `robot_ood.train.target.TargetBuilder`:

```python
phys_band     = { v : |phase[v] - phase[true]| <= PHASE_TOL and ||q[v] - q[true]|| <= JOINT_TOL }
target_set    = { v in phys_band : task[v] == task[true] }   if has_instr else phys_band
target_weight ∝ exp(-||q[v] - q[true]|| / WEIGHT_SCALE), true node × TRUE_NODE_BOOST,
                true task × TRUE_TASK_BOOST when has_instr is off, sum 1
```

`PHASE_TOL = 0.05`, `JOINT_TOL = 0.07` on `long` (`GR_JOINT_TOL`; `0.10` is the default),
`WEIGHT_SCALE = 0.10`, both boosts `2` (`robot_ood.core.constants`). Negatives are every other
reached node.

`check_loss` in `src/robot_ood/train/loss.py`:

```text
point_loss = -Σ target_weight · logsigmoid(logit[target_set]) - mean logsigmoid(-logit[negatives])
rank_loss  = -Σ target_weight · log_softmax(logit)[target_set]
loss       = 0.3 · point_loss + 0.7 · rank_loss
```

The row's loss is the mean over its checks; a check none of whose targets was reached contributes
zero.

### 2.7 Training loop and model selection

`GraphTrainer` in `src/robot_ood/train/loop.py`: `N_QUERY = 4096` rows per epoch (2048 for the
sequence run on `long`), batch `GR_QBATCH = 8` rows, AdamW at `lr = 1e-3`, `weight_decay = 1e-5`,
30 epochs (`configs/robot_ood.yaml`: `epochs`, `stage_a`). The split is by demo, never by frame:
`HELDOUT_STRIDE = 5` holds out every fifth strand; held-out demos keep their nodes in the graph.
`robot_ood.train.runner.BestEpoch` restores the epoch with the lowest held-out score
([runtime](method-runtime.md) §2.3); `run_config.json` records `best_epoch` and the metrics.

Before deployment, validate the tree with
`python -m robot_ood.cli artifacts check <graph_dir> --stage <stage>` ([operations](operations.md)).
