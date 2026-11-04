# Runtime: locator, FCSR, entry corrector

**Reads:** `g_nodes.npz`, `g_edges.npz`, `g_head.npz`, `g_segments.npz`,
`g_fcsr.npz`, `g_score.npz`, `g_invdyn.npz`, `action_scale.json` — all under `GR_GRAPH_DIR`
**Writes:** results only (evaluation result directories; the train commands below write the
artifacts they name)
**Code:** `src/robot_ood/runtime/`, `src/robot_ood/models/{retrieval,fusion,entry}/`,
`src/robot_ood/train/`, `evals/libero_plus/harness/`

```bash
python -m robot_ood.cli train locator --suite long   # -> g_head.npz
python -m robot_ood.cli train entry   --suite long   # -> g_score.npz, g_invdyn.npz
python -m robot_ood.cli train fcsr    --suite long   # -> g_fcsr.npz
python -m robot_ood.cli train         --suite long   # stale ones of the three, then the deploy gate
```

All three train against a graph `build` already wrote ([method-offline.md](method-offline.md)).
Artifact producers/consumers: [artifacts.md](artifacts.md); every knob: [configuration.md](configuration.md).

Terms used throughout. A **demo** is one recorded demonstration; a **node** is a short run of its
raw frames, one point on one demo ([method-offline.md](method-offline.md)). A **strand** (or
**lane**) is one demo's chain of nodes; **phase** is normalised progress along a demo in `[0, 1]`.
A **belief** is a probability distribution over all nodes — "where in which demonstration the arm
is". A **check** is one locator step plus one plan, every `plan_rows` policy steps. A
**segment** is a 16-raw-frame slice of one demo with its recorded actions. The **trust** is the
scalar share of the reference correction the corrector executes.

## 1. The deployed chain

`MODE` is one word: `base` runs the frozen policy alone; `fcsr` turns on the entry corrector and
the sentinel. `robot_ood.runtime.harness.DeployHooks.from_env` builds the chain and raises
`ValueError` on any other value. Nothing else configures which mechanism runs.

```bash
export MODE=fcsr GR_GRAPH_DIR=outputs/long/artifacts
```

The benchmark client (`evals/libero_plus/harness/eval_libero.py`) holds one `DeployHooks` per
process and calls it at four points of the rollout:

| call | when | does |
|---|---|---|
| `hooks.reset(shard)` | per episode, before `env.reset()` | clears the sentinel's per-episode state |
| `hooks.correct_entry(env, obs, t, frame_cb)` | once, at `t == num_steps_wait` | the t = 0 walk of §4; identity under `base` |
| `hooks.plan(env, state, step, ...)` | every policy step | feeds the query window; on a chunk boundary returns a `BlendPlan` |
| `hooks.record_action(action, step, shard)` | every env step | feeds the commanded-action window |

The `BlendPlan` is applied in `M1Inference.step`
(`evals/libero_plus/harness/model2libero_interface.py`) immediately after `self.raw_actions` is
refilled from the policy server: `self.raw_actions = blend.apply(self.raw_actions)`. The policy
stays frozen; every check only rewrites the pending action buffer it already produced.

The entry corrector and the sentinel cannot collide: the corrector advances only the env counter
`t`, the sentinel runs off the policy counter `step`, still 0 when the policy first acts. The
sentinel never writes the OSC null-space reference — the corrector owns it (§4).

Nothing in either mechanism writes a mujoco `data.ctrl` buffer or reads privileged simulator
state. C0 calibration constants resolve through `robot_ood.core.config.calibrated_constants`: an
explicit `set_constants` install, else the `$ROBOT_OOD_CONSTANTS` file, else the calibration
`train fcsr` embedded inside `g_fcsr.npz` (why a deploy needs only the artifact directory), else
the code default with one `warn_uncalibrated` warning per key per process.

## 2. The locator — where in the demonstration the arm is

**Code:** `src/robot_ood/runtime/{sentinel,locator,retrieve,readout}.py`,
`src/robot_ood/models/retrieval/gnn.py`, `src/robot_ood/train/{sequences,tasks/locator}.py`

One network, one belief over nodes, one label rule. Phase, task and confidence are readouts of the
belief, not modules. The retrieval head ([method-offline.md](method-offline.md)) reads four
channels at every check:

| channel | shape | source at deploy |
|---|---|---|
| `obs_window` | `[8, 15]` | the query window: `q`, `qdot`, `grip` of the last 8 env steps |
| `instr_emb` | `[896]` | the episode's instruction, read as a task posterior and mixed into the task table's embeddings (`GraphRetriever.language_embedding`, §2.4); absent when the harness passes no task |
| `policy_chunk` | `[8, 7]` | the chunk the policy issued after the previous check (`BlendPlan.apply` hands it to `GraphLocator.record_chunk`); absent on the first check |
| `memory` | `[V]` | the previous check's belief, kept on its top `MEMORY_TOPK = 2048` nodes; absent on the first check |

### 2.1 One check

```
belief    = softmax(node_logit)                       over the reached nodes
phase_hat = Σ belief · phase
task_prob = Σ belief by task
memory    = truncate_topk(belief, 2048)               the next check's input
```

`GraphLocator.step` (`src/robot_ood/runtime/locator.py`) runs `GraphRetriever.observe` with the
three optional channels and keeps the truncated belief. `reset` clears it. There is no transition
kernel and no separate task fusion: the memory enters the network as a per-node feature on the
seed (`log memory`, floored at `MEMORY_FLOOR = -12`), and its top `MEMORY_SEED = 128` nodes, grown
one hop along every `next` relation, join the seed set — the propagation is learned, not fixed. The instruction is projected once (896 → 32) by a layer
shared with the node side, tilts `e_Q`, and reaches the readout as the cosine to each node's task
instruction. The policy chunk tilts `e_Q` and reaches the readout as the mean distance to each
node's demonstrated action chunk. Presence flags ride beside both features, so an absent channel
is a zero with a flag, never a zero mistaken for an opinion.

### 2.2 Cadence

`Sentinel` (`src/robot_ood/runtime/sentinel.py`) is keyed on `wid`; each worker owns one
`GraphLocator`, built lazily on the first check. `SentinelConfig.plan_rows` must divide the
policy's action chunk and may not exceed the frames one segment spans, so a plan never asks the
arm to cover a segment's distance in more steps than the demonstration took; the graph itself
sets it (`robot_ood.data.segments.graph_plan_rows`). `graph_stride` must match the retriever's
training spacing.
`Sentinel.check` raises if `observe` was not called on every env step. `grip` is broadcast
constant across the window, matching the training sampler's `np.repeat`. Every check plans;
`Sentinel.plan` returns `None` only off a check boundary.

### 2.3 Training

`train locator` (`robot_ood.train.tasks.locator`) rolls the network over sequences of
`n_checks ~ U{4..8}` checks on one demo, `CHECK_EVERY = 8` raw frames apart
(`robot_ood.train.sequences.SequenceSampler`). Each check has its own window mode
(`traversal .48 / entry .28 / entry_static .16 / drift .08`; a perturbation radius is drawn
log-uniformly over the band the entry field trains on, from a quarter of the corpus's median
leave-one-demo-out nearest-frame distance — measured over every raw frame no other demo repeats
verbatim — up to the benchmark's recorded robot-init magnitude) and its own policy chunk — the cached
policy's own frames, corrupted by `PolicyCorruptor` (lag / wrong task / gain), absent with
probability `.2`. The row's instruction regime is fixed: `exact .3 / paraphrase .4 / none .3`;
without an instruction the label band spans every task with the true task counting double
([method-offline.md](method-offline.md) §2.6). The memory between checks follows the curriculum:

| epochs | memory of check k ≥ 1 | check 0 |
|---|---|---|
| `0 .. STAGE_A-1` (10) | teacher `.7` (the previous check's target weights) / none `.3` | none |
| `STAGE_A .. epochs-1` (20) | self `0 → .7` linearly (the previous belief, detached) / none `.15` / wrong `.15` (the target weights of a node on another strand at the same phase) / teacher the rest | wrong `.15` / none |

`none` and `wrong` are not optional: without them the network copies the memory and drifts at
deploy. The loss is the mean `check_loss` over rows and checks; every check's graph is freed by
its own backward, the memory is detached between checks. Selection scores a held-out rollout
under self memory from the demo start: `phase_mae + 0.1·(1 − top1_target) + 0.1·(1 − task_acc)
+ 0.01·recovery`, `recovery` the mean number of checks until the top-1 node lies in the target
set after a wrong memory at check 0. After saving, the run replays the locator over the held-out
demos at the deployed cadence and embeds the phase-skew report in `g_head.npz`
(`robot_ood.eval.phase_skew.embed_report`), which `train fcsr` reads to sample deploy-realistic
query phases.

### 2.4 Reading the instruction

**Code:** `src/robot_ood/models/language/encoder.py`, `g_lang.npz`

Each task's *bank* is every phrasing its demonstrations record (the benchmark's own instruction
pool where it has one, the instruction alone where it does not) plus lossy word-drop surrogates of
them. `build language` embeds the bank with the frozen snapshot, centres the table on the per-task
centroids, and fits a linear softmax head on the centred unit rows:

```
x        = unit(embed(instruction) - mean)
p_task   = softmax((x @ W + b) / T)                      over the graph's tasks
```

`W, b` are refit on the whole bank under the weight decay whose held-out likelihood on one fifth
of the bank is best, and `T` is the temperature that maximises that held-out likelihood. The same
snapshot is also read generatively: each task name is scored as the continuation of
`"Instruction: <instruction>\nThe robot task is:"`, softmaxed over the tasks. The posterior is the
product of the two reads, renormalised. The head knows the corpus' own wording (the benchmark's
seen pools on RoboTwin, where the task names say nothing), the generative read knows the words of
the task name (paraphrases that keep them), and each covers what the other misses. The snapshot is
Qwen2.5-0.5B — StableVLA's own LLM backbone, so no second model enters the deployment.
`embed_query_posterior` then hands the locator `p_task @ e_task`: the centroids mixed by that
posterior, in the table's own centred space. A wrong read tilts the belief instead of deleting the
right task, because the posterior is soft.

`g_lang.npz` carries `e_task` (the centred centroids), `e_aug`/`aug_ptr` (the bank rows training
draws its paraphrases from, ragged per task), the head, `p_aug` (the product posterior of every
bank row, the instruction prior training hands the graph) and `p_task` (its mean per task).

## 3. FCSR — which continuation, and how much of it

**Code:** `src/robot_ood/models/{retrieval,fusion}/`,
`src/robot_ood/runtime/{step,fuse,ee_track,retrieve,bundle}.py`,
`src/robot_ood/core/{rotation,kinematics,action}.py`, `src/robot_ood/train/fcsr.py`,
`src/robot_ood/train/tasks/fcsr.py`, `src/robot_ood/train/augment.py`

Three artifacts: `g_segments.npz` (`build segments`), `g_fcsr.npz` (`train fcsr`),
`action_scale.json` (`build calibrate`). One job trains the segment retriever, the scorer and
the corrector against one objective, so they are one checkpoint: `FcsrNet`
(`src/robot_ood/models/fcsr_net.py`) writes them into `g_fcsr.npz` under the prefixes `seg.`,
`scorer.` and `corrector.`; `RuntimeBundle` (`src/robot_ood/runtime/bundle.py`) reads the file
once and builds each piece, plus `action_scale.json`. Missing any of them raises — no silent
fallback; `g_fcsr.npz` is in `REQUIRED_ARTIFACTS["deploy"]` (`robot_ood.data.store`), so the gate
fails before a run starts rather than at the first check.

### 3.1 The runtime path

Every env step, `hooks.plan` / `hooks.record_action` push `q_now` and the commanded action into
the windows. At a chunk boundary:

```
Sentinel.check       the locator reads the window, instruction, chunk and memory -> phase_hat (§2)
Sentinel.plan        ee_now = FK(q_now); robot_ood.runtime.step plan
  step.plan          fusion on -> fuse.plan_fusion
    plan_fusion      QueryContext from the window; SegmentRetriever.retrieve -> K = 8 candidates
                     returns BlendPlan(authority=CorrectionAuthority(...), cands=..., ee=ee_now)
harness refill       self.raw_actions = blend_plan.apply(self.raw_actions)
  BlendPlan.apply    CorrectionAuthority.apply(a_policy, cands, ee_now, query_phase=...).a_exec
```

`ee_now` and `phase_hat` are each computed once per check and threaded into both the query context
and the fusion call — never a second estimate. `BlendPlan.a_track` (the `EeTrack` chunk) is for
tracing only; once `authority` is set, `apply` never reads it.

**Frame convention:** everything is in the robot base frame. `NodeTable.ee_base`
forward-kinematicks every raw frame's `q_raw`; the live pose is `FK(q_now)` the same way — LIBERO
parks the base differently per scene, so a world-frame retrieval crossing a scene boundary would
command tens of action units toward another room. `ee_raw` (world) survives only as FK ground
truth (`tests/test_train.py::test_forward_kinematics_round_trips_the_recorded_end_effector_pose`).

### 3.2 Segment memory and retrieval

`g_segments.npz`: segments of `SEGMENT_LEN = 16` raw frames, per row `start_node`, `end_node`,
`owner`, `task_id`, `phase_start`, `phase_end`, `action_chunk [16, 7]` (the demo's recorded
actions), `ee_start`/`ee_end` (base frame). Virtual nodes carry no segment (`node_to_segment`
holds `-1`). The deploy loop runs no image encoder.

`QueryEncoder` (`src/robot_ood/models/retrieval/encoders.py`) turns a `QueryContext` — joint
history, velocity, grip, cleanliness `w`, action history, live `ee`, `phase` — into a query
embedding and a predicted future action chunk (`predict_future`). `ee` and `phase` are observed,
never estimated: `ee` is the caller's live base-frame pose, `phase` is §2's `phase_hat`; the
retriever raises rather than substitute an estimate. An absent action history routes through a
learned `action_absent` vector, not zeros. `lang_dim` is 0 in the shipped checkpoint — FCSR
retrieves purely on the observed window and the locator's phase.

Retrieval runs in two stages (`SegmentRetriever`, `src/robot_ood/models/retrieval/retriever.py`).
The **shortlist cut** is one matmul against a cached `[S, hidden]` key table, tilted by the
locator's belief, keeping `M = 24` (`SEG_TIER1_M` default in `robot_ood.core.schemas`;
`GR_SEG_TIER1_M` overrides). The **final score** then ranks those M down to `K = 8`.

The tilt is the belief's own task marginal, in log space, and it applies to the cut alone:

```
shortlist_score = e_q @ key_table.T + SHORTLIST_PRIOR_W * log task_p[segment.task]
SHORTLIST_PRIOR_W = 1.0
```

so a segment of a task the belief has ruled out drops out before rescoring. The prior is
deliberately **not** carried into the final score — it decides who is ranked, not how they rank.
The final score is the embedding match plus the scorer's fine check:

```
score = match + fine_check(features),    match = (e_q * seg_key).sum(-1)
```

`match` is added to the MLP's output rather than fed in as a feature, which is what keeps gradient
flowing to the key encoder: the features themselves never read it. `SegmentScorer`
(`src/robot_ood/models/retrieval/scorer.py`) is an MLP over the four `SEG_FEATURES`, in order:

| feature | meaning | at deploy |
|---|---|---|
| `d_state` | `‖ee_query − segment.ee_start‖`, normalised units | live |
| `d_future` | `1 − cos(ActionEncoder(query future), ActionEncoder(segment chunk))` | live |
| `phase_diff` | `phase_query − segment.phase_start`, signed | live |
| `task_affinity` | `task_p[candidate.task]`, the belief's mass on that candidate's task | live |

`task_affinity` is continuous, not an identity test: the belief spreads mass over the tasks it
cannot separate, so a corpus thin in this task still offers its neighbours something to say.
Training attaches the same marginal to every batch before any consumer reads it
(`FcsrTask._with_belief`), so the column means the same thing on both sides. Features are
standardised and clamped at `STAT_CLAMP = 4` sigma, stats stamped under `scorer.`. No parameter
tensor is dimensioned by the segment count — the segment memory is swappable at deploy without
retraining (`SegmentScorer.forward` asserts the feature stack against `SEG_FEATURES`). No hard
phase mask on the shortlist: `phase_query` is an estimate, so `phase_diff` stays a soft feature.

### 3.3 The blend

The blend target is a **pose**, not a recorded action chunk — replaying another trajectory's
deltas open-loop from a perturbed pose cannot close a state error:

```
a_track = track_chunk(ee_ref, ee_now, scale)     # robot_ood.models.fusion.target
```

`ee_ref` reads raw frames from each candidate's first node on (`PoseReference`,
`src/robot_ood/models/fusion/target.py`), edge-padded at its demo's end, so the reference is the
demo's own recorded pose rather than a resampling of the node grid. Row `k` is measured against
`predecessors()[k]`: the live pose for row 0, `ee_ref[k−1]` after — only row 0 closes the loop.
`track_chunk` (torch) is the differentiable restatement of `EeTrack.chunk` (numpy,
`src/robot_ood/runtime/ee_track.py`); a test pins the two against each other. The candidate's
recorded chunk survives as a gate feature only.

At `K = 8` top-1 is right a minority of the time, so the corrector
(`src/robot_ood/models/fusion/corrector.py`) aggregates posterior-weighted over the shortlist:
`p = softmax(score)`, `value = Σ_k p_k · value_k`. Averaging poses is safe (a point inside the
convex hull, servoed with a closed-loop row 0) where averaging disagreeing action deltas cancels
toward zero. The aggregate pose trajectory tracked from the live pose is `a_ref`.

The correction is one gain over one step:

```
G      = d · A                            d ∈ (0, 1) per check, A ∈ (0, 1)^{C×6} per row and dim
a_exec = a_policy + G ⊙ (a_ref − a_policy)
```

`A` (the **shape** head) reads per-row features `z_row` (24 wide: policy pose, aggregated
candidate pose, `ee_rel` — candidate `ee_start`/`ee_end` relative to the live pose, never
absolute): where the correction acts. `d` (the **trust** head) reads per-check features `z_chk`
(16 wide): whether the retrieval is worth following.

| block | columns | what it says |
|---|---|---|
| shortlist confidence | 3 | top-1 mass, entropy, mean \|candidate − policy\| |
| phase | 2 | query phase, phase gap to the aggregate |
| score | 1 | top-1 tier-2 score |
| step size | 2 | mean ‖a_ref − a_policy‖ over position and over rotation |
| start offset | 2 | the top candidate's start-pose offset, position and rotation |
| `consensus_outlier` | 1 | does the policy chunk lie outside the spread of the retrieved demos |
| candidate spread | 1 | weighted spread of the candidates' end poses: do they agree on a goal |
| gate | 2 | the locator's own view of this check: off-manifold distance, belief entropy |
| task | 2 | shortlist mass on the believed task, and mass on its own most-agreed task |

The split is a
feature split, not a loss split; `corrector_features` is the one builder both training and
`CorrectionAuthority` call, and both feature rows are standardised by mean/std stamped into the
checkpoint, a feature the sample never varied keeping unit scale. Convex by construction — both heads are sigmoids, so `a_exec` lies on the closed
segment between `a_policy` and `a_ref` elementwise. The shape head's bias initialises at −1.0
with the last layer zeroed (an untrained gate emits ≈ 0.27, "trust the policy"); the trust head
starts at 0.5.

**The gripper is never blended.** Fusion writes only dims `0:6`; dim 6
(`robot_ood.core.schemas.GRIPPER_COL`) passes through verbatim — an interpolated value is a
command the binary gripper cannot obey.

**Units.** The policy emits unnormalised OSC deltas `[dx, dy, dz, drx, dry, drz, gripper]`; the
reference segment is in metres and radians. `build calibrate` fits two scalars into
`action_scale.json`: for each block, the measured pose change of a transition is regressed by least
squares through the origin on the commands of the `RESPONSE_TAPS` control steps ending in it, and
the scale is the summed gain, so a controller that realises a command one step late reads the
same as one that realises it at once;
`ActionScale.check(0.8)` (`robot_ood.core.action`) halts on a bad fit, and a missing file raises
naming `build calibrate` — a wrong scale mis-drives the arm silently. Rotation deltas are composed,
never subtracted: `logmap(R_to @ R_fromᵀ)`, with Taylor branches below `_SMALL_ANGLE` keeping the
gradient defined at zero. Row 0 of `a_track` is an unbounded pose error, so the fusion target
applies `row0 <- bound · tanh(row0 / bound)` with `bound = ACTION_LIMIT = 1.0` (LIBERO clips at
`[−1, 1]`); `tanh` rather than clip because the training loss reaches the retriever through this
target.

### 3.4 Training: one objective

`train fcsr` fits the retriever and the corrector as one objective — one module (`FcsrNet`),
one optimiser, one forward, one backward, one checkpoint (`FcsrTask`,
`src/robot_ood/train/tasks/fcsr.py`; run options in `robot_ood.train.fcsr`). Demonstration-only:
no rollout, episode outcome or policy weight is read. It first runs
`robot_ood.build.calibration.ensure_constants`, which resolves the C0 calibration (§1) or measures
it in-process; the result is embedded in the checkpoint.

The two loss terms reach two disjoint parameter sets, because the corrector reads the shortlist as
constants — `topk` is not differentiable, so a downstream gradient could not repick segments
anyway. A weight ramp (`lambdas` in `robot_ood.train.fcsr`) admits the corrector term at epoch 5,
once tier 1 has settled (the segment weight dropping to 0.5); the corrector's feature statistics
are stamped off the held-out shortlist that same epoch.

The segment term is `0.3·phase_ce + 1.0·infonce_global` (on the same tier-1 dot product deploy
ranks by) `+ 0.5·future_loss + 0.5·shortlist_ce` — the last a CE over the tier-2 rescore of the
real tier-1 shortlist at deploy's `M`, the true continuation at a shuffled column, doubling as the
calibration objective. Hard negatives are mined for same state / different future, same phase /
wrong task, and same-state different-task inside the entry band. `future_temp` is fitted after the
last epoch by bisection over held-out logits and stamped as `seg.future_temp`.

The corrector term (`robot_ood.train.loss.CorrectorLoss`) has one target, `a_corr`: the query's
own demonstration continued from the live pose, on every row. Rows are corrupted on purpose
(`robot_ood.train.augment`) on both operands. The retrieval side swaps a column (`WRONG_TASK`,
`WRONG_PHASE`), displaces one candidate's start, end and reference rows (`OFFSET`) or displaces
every candidate's later rows and end while the starts stay (`TARGET`: the scene moved, the demos
still agree); each is realised on the real retrieved shortlist before the corrector sees the row,
and the shortlist never holds the query's own demonstration. The policy side lags the chunk
(`LAG`), swaps its task (`P_TASK`), scales it (`GAIN`) or displaces the live pose the chunk
starts from (`P_SHIFT`: the arm is not where the demonstrations were, the policy acts as if it
were, and `a_corr` is the recovery). The label the trust head answers is measured on the row:

```
helps = [ mse(a_ref, a_corr) < mse(a_policy, a_corr) ]
L     = mse(a_exec, a_corr) + bce(d, helps)
```

so a corrupted retrieval puts the reference far from `a_corr` and pushes the dose down, a
corrupted policy puts it on `a_corr` and pushes the dose up, and a clean row settles at what the
neighbouring demonstrations are worth. The BCE is a plain mean, not class balanced: `d` is the
probability that the reference helps, prior included, because most deployed checks are ones where
it does not, and the executed dose is `d` times the shape.
Each epoch reports held-out `mse` at the policy, at the full reference, at the
best constant gain, at the learned gain and at the per-row oracle dose
(`robot_ood.models.fusion.corrector.optimal_dose`, the closed-form best scalar had the target been
known — an evaluation oracle only), plus the trust head's AUC and accuracy — a corrector that does
not beat the best constant has learned a level, not a decision.

## 4. The entry corrector — the t = 0 mechanism

**Code:** `src/robot_ood/runtime/entry.py`, `src/robot_ood/models/entry/{score,invdyn,target}.py`,
`src/robot_ood/core/kinematics.py`, `src/robot_ood/train/tasks/entry.py`

Runs once per episode, before the policy's first action, then never again. `ScoreEntryCorrector`
carries no parameters: it reads two trained nets (`g_score.npz`, `g_invdyn.npz`, both from
`train entry`, both fit on demonstration HDF5 alone — a different policy needs no retrain) and a
diagnostic target. It reads joint encoders (`robot_ood.core.kinematics.q_arm`), writes the OSC
controller's null-space posture reference (`robot_ood.core.kinematics.set_nullspace_home` — the
only call in the codebase that reaches it), and emits actions through `env.step` like the policy.
`DeployHooks._load_entry` constructs it once per process from `default_paths().graph(suite)`,
which honours `GR_GRAPH_DIR` — the same directory the pre-launch artifact gate
(`scripts/run_sr.py`) checked. Its control steps are charged to the episode's `max_steps` and
rendered into the replay video.

### 4.1 The walk

Config is `EntryConfig` (`src/robot_ood/core/config.py`): four fields, each an `EN_*` env override
on a calibrated default — `score_sigma_hi` (`EN_SIGMA_HI`), `score_sigma_lo` (`EN_SIGMA_LO`),
`score_refine_steps` (`EN_REFINE_STEPS`), `score_step_cap` (`EN_STEP_CAP`). Absent both env and
calibration, the dataclass literal is used and `warn_uncalibrated` fires.

```python
# WHERE -- pure inference, costs no env steps.
q_hat = q0
for _ in range(cfg.score_refine_steps):                      # 3
    sigma = clamp(level_estimate(q_hat) / sqrt(7), cfg.score_sigma_lo, cfg.score_sigma_hi)
    q_hat = tweedie_denoise(score_net(q_hat, sigma), q_hat, sigma)

# WALK -- one env step per iteration while the gap is outside the radius, then a resync.
level_hat = level_estimate(q_hat)
while remaining > level_hat and n_steps < _MAX_WALK:         # _MAX_WALK = 64
    dq = clip_norm(q_hat - q_now, cfg.score_step_cap)        # rate limit only
    action = invdyn(q_now, dq); action[6] = OPEN
    set_nullspace_home(env, q_hat)                           # every step
    obs = env.step(action); q_now = q_arm(env)
    remaining = norm(q_hat - q_now)                          # measured, never predicted
if n_steps > 0:
    obs = env.step(RESYNC)                                   # zero EE delta, gripper open
```

The stopping rule: the walk runs only while the gap to `q_hat` is outside the level head's own
reading at `q_hat`, so a start already inside that radius costs no env step at all. `remaining` is
measured off the encoders after every step (under OSC no step lands its full commanded
displacement) and compared against the head's reading at the target — one head at two
configurations, no fixed magnitude, so the rule re-normalises itself on retrain. A scene
perturbation leaves the arm where the demonstrations start (delta ~0.0005 rad against a radius
~0.009 on the RoboTwin lanes), and an env step emitted there moves nothing but changes the rollout
the policy then produces; a robot-init perturbation starts 0.2-0.3 rad out and walks 3-5 steps.
`_MAX_WALK` bounds a walk that never closes, logging a warning and `last["walk_capped"]`. Every
emitted step holds the gripper open.

**Why joint space.** `Robot_Initial_States` perturbs `init_qpos` at reset; the OSC controller
captures its null-space posture reference from that perturbed configuration and never refreshes
it. The error lives in the null space of the end-effector Jacobian — nothing commanding EE deltas
can act on it, and left alone the controller drags the arm back to the perturbed posture. The walk
re-points the reference at `q_hat` on every step; the write persists for the episode.

**Components.** `ScoreNet` learns `s(q, σ) ≈ ∇_q log p_σ(q)` by denoising score matching over
every demo's own t = 0 frame, plus a separate-trunk level head `δ̂(q) = exp(MLP(z(q)))` — radians
off-manifold, input only the standardised `q` because σ is exactly what deploy does not know.
Tweedie's formula `q̂ = q + σ²·s(q, σ)` is the posterior mean under the Gaussian corruption model;
3 refinement passes trade posterior quality against drifting to the mode. The level head regresses
a radius, the score field wants a σ; they differ by `√D`, so `_RADIUS_TO_SIGMA = 1/√7`, applied
identically in `robot_ood.runtime.entry`, `EntryConfig` and `robot_ood.train.tasks.entry`. The σ
clamps are the benchmark's measured Robot-Init ceiling and the corpus's leave-one-demo-out noise
floor (`robot_ood.build.measure.RobotInitMagnitude` / `EntryPerturbation`), converted the same
way. `InverseDynamicsNet` maps `(q_t, Δq) → a_t` from consecutive demo triples, output clipped to
the corpus's action range; `score_step_cap` (the corpus's p99 per-step joint displacement,
`robot_ood.build.measure.EntryStepCap`) keeps every query inside that range — it plays no role in
stopping. `EntryTarget` is the demo-start cloud under a Ledoit–Wolf shrunk covariance; only the
`err0`/`err1` diagnostics read it, no control branch.

### 4.2 Training and telemetry

`train entry` fits both nets back to back off `g_nodes.npz` alone. The score task draws σ
log-uniformly over the deployed query range, uses the σ-weighted DSM loss (finite at every scale),
and fits the level head in log space on the exactly-known corruption displacement — the label is
free. The two loss terms are summed unweighted; the heads share only the standardisation buffers,
so each gradient is its own. `delta_on_clean` — held-out median `δ̂` on unperturbed starts — is
the logged health number. `ScoreNet.load` refuses an artifact with no `level.*` keys, naming the
retrain command; `InverseDynamicsNet.load` refuses a width mismatch. Both raise, never warn.

`ScoreEntryCorrector.last` after every run: `fired`, `err0`/`err1`, `steps`, `delta` (the
displacement the field asked for), `residual` (the gap handed off), `realised`
(`moved_total / commanded`; `nan`, not 0, when the commanded total is zero), `walk_capped`,
`level0`/`level_hat`/`level_now`, `moved_total`.

---

Success-rate results for all three mechanisms live in [results.md](results.md); running the
evaluation ladder, in [operations.md](operations.md).
