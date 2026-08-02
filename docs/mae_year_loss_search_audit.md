# MAE-year loss-search audit

## Scope and upstream

- Canonical source: `src/mae-year` from `origin/experiment/new-mae`.
- Audited upstream SHA: `b2f6ef6b5bd6697b4beeb6838d875082c226fa8f`.
- `git fetch origin` on 2026-08-02 returned the same SHA as the supplied
  reference, so there was no newer upstream delta.
- Work is isolated on `experiment/mae-year-loss-search` in the sibling
  `KTxTS_project-mae-year-loss-search` worktree. Neither
  `experiment/new-mae` nor the earlier `experiment/new-mae-loss-search`
  worktree was modified, rebased, or reset.
- No actual-data training run, CUDA run, push, fixed-test load, or fixed-test evaluation is
  part of this preparation. The bounded CPU smoke performs forward/backward
  only and does not take an optimizer step.

## Canonical mae-year contract

### Year and data selection

The runner requires exactly one `--year`, either `2019` or `2023`. `ODDataset`
then resolves that year's dong-code workbook, distance matrix, static feature
table, OD matrix, validation city codes, test holdout city codes, and merge
cache. It never constructs a joint-year dataset and never adds a year
indicator.

- 2019 graph: 1,101 nodes; train nodes exclude the 2019 validation and test
  city indices; validation uses only `fixed_val_meta_2019.pt`.
- 2023 graph: 1,137 nodes; train nodes exclude the 2023 validation and test
  city indices; validation uses only `fixed_val_meta_2023.pt`.

The configured `od_data_2019.csv` and `od_data_2023.csv` files are absent from
the upstream repository. Without a correction, both canonical dataset
constructors fail before training. The minimal fallback loads only
`X_OD_raw` from the matching `base_data_<year>.pt` artifact and requires its
shape to equal `(num_nodes, num_nodes)`. The artifact is common raw graph data,
not fixed-test metadata. Its exact file hash is included in the corresponding
year's fingerprint.

### Dataset preprocessing

The OD target is the sum of the five purpose columns (home, commute, school,
business, and other) when a CSV is available, or the shape-validated raw OD
matrix described above. OD model inputs and targets use `log1p`; distance uses
`log1p`. The output dictionary provides:

- `X_static`: `(N,F)` static features. Eighteen sorted numeric features are
  standardized with a `StandardScaler` fit only on same-year training nodes,
  followed by `is_masked` and `is_merged` indicators.
- `X_OD_masked`: `(N,N)` log1p OD input with every masked/holdout/inactive row
  and column zeroed.
- `X_dist`: `(N,N)` log1p distance input. Inactive rows/columns receive the
  established 5.5 fill.
- `A_spatial`: `(N,N)` geographical adjacency. Inactive rows/columns are zero.
- `y_OD`: `(N,N)` log1p OD target after any merge augmentation.
- `mask`/`loss_mask`: `(N,)` nodes whose row or column is predicted.
- `active_node_mask`: `(N,)`; merged-away secondary nodes are false.

The dataset exposes 1,000 stochastic samples per epoch. Its mask-size ceiling
progresses linearly from 3 to 150. Seeds are oversampled with weight
`max(max outgoing raw OD, max incoming raw OD) / 1000`, floored at 1. A sample
starts from one or multiple seeds and expands over the adjacency graph, with a
random fallback when a frontier is exhausted.

### Merge augmentation and active nodes

The upstream merge strategy is unchanged:

- known + known background merges can occur with probability 0.3, up to 30;
- masked-cluster merges can occur with probability 0.5;
- masked + masked and masked + known are selected from cached adjacent pairs.

The surviving primary node receives merged raw OD rows/columns, the diagonal
adds both original self-loops and both cross-flows, static/distance values are
reconstructed from the same-year merge cache, and the secondary node becomes
inactive. The masking strategy, merge probabilities, cache semantics, and
augmentation outputs were not redesigned.

All loss candidates now reduce only over:

`(masked origin OR masked destination) AND active origin AND active destination`.

This prevents gradients from cells incident to merged-away nodes. Validation
uses the same active-pair intersection.

### Model input/output and self-loops

`ODMAE.forward` consumes batched `X_static`, `X_OD_masked`, `X_dist`,
`A_spatial`, `mask`, and `active_node_mask`. It combines static embeddings,
outgoing/incoming OD attention pooling, directional OD GCN messages, a learned
mask token, distance-bucket attention bias, a Transformer encoder, bilinear OD
decoding, learned distance friction, and a decode-distance bias.

The single output is an unconstrained `(B,N,N)` prediction on the same log1p
scale as `y_OD`. It is the only output to which a registry loss is applied;
there is no Meta-Gravity prior output. The dedicated self-loop predictor is
added to the model's diagonal prediction. Spatial GCN adjacency explicitly
zeros self-loops before message passing. The diagonal indexing tensors now use
the prediction device, so CUDA forward no longer mixes CPU indices with a CUDA
prediction.

The canonical `train.py` disables PyTorch's MHA fast path. The new runner keeps
that runtime setting. In the real-data smoke, enabling the eval fast path with
the additive distance mask produced all-NaN output, while the canonical
disabled-fast-path contract was finite.

## Fixed validation and Task 0-4

Each year's checked metadata contains the same three validation city labels
(`동탄`, `위례`, `검단`), five tasks per city, and 50 samples per city/task:
15 groups and 750 samples per year. City dong codes remain year-specific.

- Task 0: no background merge and no target merge.
- Task 1: target nodes are unmerged; optional known+known background merges.
- Task 2: masked+known target merges, plus optional known background merges.
- Task 3: masked+masked target merges, plus optional known background merges.
- Task 4: mixed masked+known and masked+masked target merges, plus optional
  known background merges.

For validation, same-year test holdout features/OD are hidden from model input.
The evaluated cells are rows or columns incident to the task's masked city,
intersected with active-node pairs. Prediction log values are capped at 20
before `expm1`, negative raw predictions are floored at zero, and RMSE
squaring is performed in float64. A non-finite model output or target raises a
named sample error.

Every record stores raw-scale CPC, RMSE, and `%RMSE = RMSE / mean(target)` for
all cells, off-diagonal cells, and diagonal cells. It also stores each region's
cell count. CSV/JSON group records aggregate by year/city/task and include
sample counts. Metadata completeness is checked before dispatch and the
evaluator requires all scheduled samples to return; no exception can silently
shrink the scoring set.

Because every run has one required year, checkpoint selection never averages
2019 and 2023. `best_cpc.pt` maximizes the same-year mean sample CPC;
`best_rmse.pt` minimizes the same-year mean sample RMSE. `latest.pt` is saved
every epoch. 2023 remains the primary selection track for future third-new-town
prediction; 2019 is a separate robustness/reporting track.

## Audit findings and minimal corrections

1. **Huber call mismatch — fixed.** `train.py` calls every criterion as
   `(prediction, target, current_alpha, mask)`, but upstream `HuberLoss.forward`
   accepted only `(prediction, target, mask)`, causing a positional-argument
   `TypeError`. It now accepts and intentionally ignores `current_alpha`,
   supports an optional mask, and returns differentiable zero for an empty
   selection.
2. **Inactive-node training cells — fixed.** The upstream training loss used
   only the row/column mask. Cells touching merged-away nodes could contribute
   loss and gradient. The canonical loop now intersects the cell mask with both
   endpoint active flags; the registry enforces the same rule independently.
3. **CUDA diagonal indices — fixed.** Both `torch.arange` tensors in
   `models.py` were created on CPU. They now use `pred_od.device`.
4. **Validation overflow — fixed.** Unbounded `expm1(pred)` and float32
   residual squaring could produce Inf. Prediction log values are capped at 20,
   targets/predictions are checked for finiteness, and metric arithmetic is
   float64.
5. **Silent sample omission — fixed.** A sample exception formerly printed a
   warning, returned `None`, and allowed scoring on a subset. It now raises a
   contextual error; Task 0-4 metadata and final result counts are mandatory.
6. **Default-loss ambiguity — made explicit.** Upstream `train.py` defaults to
   `weighted_mse`, while a comment in `loss.py` says the hybrid was more
   accurate. There is no encoded benchmark or selection policy proving the
   hybrid is the canonical winner. The upstream default is preserved for
   compatibility, while the new runner requires an explicit `--loss`; neither
   baseline is silently promoted.
7. **Missing OD CSVs — fixed with a bounded fallback.** The year-specific
   config paths do not exist in the upstream tree. Only the matching base
   artifact's raw OD matrix is used after an exact shape check, and the source
   hash is fingerprinted.
8. **MHA eval fast path — preserved.** This was already disabled in canonical
   `train.py` but would have been lost in a standalone runner. The runner now
   applies the same setting before model use.

No model layer, masking policy, merge policy, or validation task definition was
otherwise changed.

## Baseline loss registry

Let `p` and `t` be final OD prediction and target on the log1p scale, and let
the mean range only over the common valid-cell mask above. Alpha progresses
linearly from 1 to 10 over epochs and is stored in metadata/checkpoints.

1. `weighted_mse`

   `mean((p - t)^2 * (1 + alpha * t))`

   This is value- and gradient-identical to upstream `WeightedMSELoss`.

2. `hybrid_weighted_mse`

   `weighted = mean((p - t)^2 * (1 + alpha * t))`

   `raw_huber = mean(Huber_delta=100(expm1(min(p,12)), expm1(t)))`

   `total = weighted + real_penalty_weight * raw_huber`

   The default `real_penalty_weight` is 0.005, exactly matching upstream. Epoch
   metadata records `weighted_mse`, unscaled `raw_huber`, and
   `scaled_raw_huber` separately. Value and gradient match upstream total.

3. `huber`

   `mean(Huber_delta(p,t))`, default `delta=1`.

   For residual `r=p-t`, Huber is `0.5*r^2` when `|r|<=delta`, otherwise
   `delta*(|r|-0.5*delta)`. Alpha is ignored.

The registry contains only these three candidates. All receive the same
prediction/target scale, node/cell mask, and active-node contract.

## Follow-up hypotheses, not defaults or winners

- `huber_od` with target-volume exponent `tau=0.35` may improve robustness
  while retaining volume emphasis; it must be re-derived for mae-year.
- a weak raw-count CPC hybrid may align optimization with checkpoint CPC, but
  couples all selected cells and needs overflow/denominator guards;
- a bin-balanced hybrid may address OD imbalance, but weights must come only
  from the matching year's training targets and masking exposure;
- Poisson deviance may model count scale naturally, but needs nonnegative mean,
  zero-target, and exponential overflow handling.

None is registered in this first baseline stage. The older Meta-Gravity prior
loss contract is intentionally excluded.

## Why the previous mae_new search is not portable

The earlier loss-search branch uses a different dataset decomposition and a
different model that returns both final OD and a Meta-Gravity prior. Its runner
applies the selected loss to both outputs. Canonical mae-year instead consumes
a single normalized static tensor plus spatial adjacency and active-node mask,
uses stratified graph masking and cached merge augmentation, and returns one OD
matrix. Its fixed validation includes Task 0 and uses mae-year merge semantics.

Copying the prior model, dataset, or validation adapter would therefore change
the model and scoring contract rather than test only the final loss. Reused
ideas are limited to atomic persistent checkpoints, full RNG restoration,
environment/code/data manifests, fingerprints, CSV/JSON reporting, and
gradient/memory diagnostics.

## Runner artifacts and resume policy

Runs are stored under:

`<output-root>/<year>/<loss>/<fingerprint>/`

The fingerprint includes Git SHA, code file hashes and aggregate code-manifest
hash, year, seed, full model config, preprocessing, mask/merge config, loss name
and every parameter, alpha curriculum, validation settings, deterministic and
optimizer settings, and hashes for only the selected year's data artifacts.

Each epoch records total/component losses, pre-clip gradient norm, clipping
count/fraction, CUDA peak allocated/reserved memory (or null on CPU), epoch
runtime, and validation metrics when scheduled. Validation has sample/group
CSV and JSON. Final output has summary CSV/JSON and `completed.json` with
Python, PyTorch, CUDA, NumPy, scikit-learn, seed, backend determinism, Git,
model/loss config, hashes, diagnostics, and elapsed runtime.

`latest.pt` restores model, optimizer, OneCycle scheduler, epoch, Python RNG,
NumPy RNG, PyTorch CPU RNG, every CUDA RNG, and the DataLoader generator. RNG
tensors are normalized back to contiguous CPU byte tensors before PyTorch RNG
APIs, avoiding the CUDA `map_location` failure mode. An incomplete matching
run resumes automatically from `latest.pt`; `--resume` additionally requires
that it exist. Only a matching `completed.json` is skipped.

## Verification performed

- `compileall` with bytecode cache redirected to `/tmp` (the sibling worktree
  is intentionally outside the original sandbox write root): passed.
- full `unittest` discovery: 20/20 passed.
- exact upstream parity: weighted MSE and hybrid weighted MSE values and
  gradients passed at zero tolerance.
- finite forward/backward for all registry losses; empty mask; zero target;
  inactive-node gradient blocking: passed.
- Huber four-argument canonical train-loop call: passed.
- small canonical ODMAE forward and device-local diagonal indexing: passed.
- validation overflow guard, off-diagonal/diagonal cell split, contextual
  sample failure, official Task 0-4 completeness: passed.
- checkpoint round trip; CPU-normalized RNG restore; uninterrupted versus
  resumed NumPy/Python/DataLoader order: passed.
- fingerprint separation by year, loss, and loss parameter: passed.
- real-data bounded CPU smoke with canonical dataset/model implementation:
  one forward/backward for 2019 and one for 2023 passed; fixed validation
  evaluated one sample for every one of 3 cities × 5 tasks (15/year), with
  task counts `{0:3,1:3,2:3,3:3,4:3}`. Overall evaluated cells by task were
  2019 `{0:28567,1:28467,2:13154,3:19731,4:15309}` and 2023
  `{0:40810,1:40394,2:27178,3:27110,4:26926}`.
- the smoke guarded every `torch.load`; observed paths were only
  `base_data_{2019,2023}.pt` and `fixed_val_meta_{2019,2023}.pt`.
  `fixed_test` access was false.

No actual-data CPU optimizer step, CUDA execution, long training, or push was
performed.

## Colab next steps

Mount Drive and install the repository requirements first. These are bounded
CUDA preflight commands (one optimizer step and one fixed-validation sample per
city/task), not full baselines:

```bash
# Primary 2023 weighted-MSE preflight
python scripts/run_mae_year_loss_search.py \
  --year 2023 --loss weighted_mse --seed 42 \
  --epochs 1 --batch-size 1 --max-train-batches 1 \
  --validation-every 1 --fixed-max-samples-per-group 1 \
  --device cuda \
  --output-root /content/drive/MyDrive/KTxTS/mae-year-loss-search
```

```bash
# Separate 2019 robustness weighted-MSE preflight
python scripts/run_mae_year_loss_search.py \
  --year 2019 --loss weighted_mse --seed 42 \
  --epochs 1 --batch-size 1 --max-train-batches 1 \
  --validation-every 1 --fixed-max-samples-per-group 1 \
  --device cuda \
  --output-root /content/drive/MyDrive/KTxTS/mae-year-loss-search
```

After both preflights complete, remove `--max-train-batches` and
`--fixed-max-samples-per-group`, set the intended epoch/batch values, and run
`weighted_mse` and `hybrid_weighted_mse` as separate fingerprints. Do not
combine years or average their checkpoints. Re-running an unfinished command
uses its `latest.pt`; `--resume` can be supplied when an existing checkpoint is
required.
