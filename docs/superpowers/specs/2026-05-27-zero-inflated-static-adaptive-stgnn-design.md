# Zero-Inflated Static Adaptive STGNN Design

Date: 2026-05-27

## Context

The project currently has a static-adaptive STGNN variant under `GNN/`. The first version adds a building-attribute static graph to the STGNN baseline, but the initial result is weaker than the original learned-graph STGNN on the electricity task.

The target training organization is:

- Electricity keeps one standalone model.
- Chilled water, steam, and hot water each train a separate standalone model.

This design improves only the three non-electricity meter models. Electricity should keep the existing single regression head so its continuous-load behavior is not affected.

## Data Observations

The notebook `ashrae-start-here-a-gentle-introduction.ipynb` highlights that missing readings, zero readings, and non-zero readings are distinct states for each building/meter pair over time.

Statistics from the current preprocessed split files:

| meter | train zero rate | valid zero rate | test zero rate |
| --- | ---: | ---: | ---: |
| electricity | 5.19% | 0.87% | 0.82% |
| chilled_water | 14.17% | 17.00% | 24.63% |
| steam | 14.28% | 9.08% | 5.00% |
| hot_water | 28.25% | 26.30% | 16.45% |

The non-electricity meters show seasonal zero-rate shifts:

- Chilled water has higher zero rates in winter months.
- Hot water has higher zero rates in summer months.
- Steam has higher zero rates in warmer months.

Strict consecutive `meter_reading == 0` runs are short in the current preprocessed files, usually no more than two hours per building. Therefore, the improvement should be framed as learning a seasonal and contextual zero state, not as assuming long uninterrupted zero blocks in the preprocessed training rows.

## Model Design

Keep the existing static-adaptive graph constructor and STGNN encoder.

For electricity:

- Use the existing `StaticAdaptiveSTGNNGRURegressor`.
- Output a single continuous prediction.

For chilled water, steam, and hot water:

- Add `StaticAdaptiveZeroInflatedSTGNNGRURegressor`.
- Share the same graph constructor, GCN stack, layer attention, and GRU encoder.
- Replace the single regression head with two heads:
  - `zero_head`: predicts whether the observed target is exactly zero.
  - `value_head`: predicts latent non-negative usage intensity.

The forward pass returns enough information for training and diagnostics:

```text
zero_logit = zero_head(last_hidden)
p_zero = sigmoid(zero_logit)
value = softplus(value_raw)
prediction = (1 - p_zero) * value
```

The main prediction remains a soft-gated value. Hard thresholding is used only for diagnostic metrics in the first implementation.

## Loss Design

For observed target positions only:

```text
zero_target = meter_reading == 0

zero_classification_loss = BCEWithLogits(zero_logit, zero_target)
positive_regression_loss = MSE(prediction, target), only where target > 0
zero_suppression_loss = MSE(prediction, 0), only where target == 0

total_loss = zero_classification_loss
           + positive_regression_loss
           + zero_suppression_loss
```

Initial weights are all `1.0`. This keeps the first version interpretable.

Use a training-set `pos_weight` for the zero classification loss:

```text
pos_weight = non_zero_count / zero_count
```

This balances the zero class in `BCEWithLogitsLoss`, where the positive class is `zero_target == 1`.

Approximate training-set values from current data:

| meter | pos_weight |
| --- | ---: |
| chilled_water | 6.05 |
| steam | 6.00 |
| hot_water | 2.54 |

If a batch has no positive targets or no zero targets, skip that component for the batch instead of forcing artificial samples.

## Evaluation

Keep existing regression metrics:

- MSE
- MAE
- RMSE
- R2
- SMAPE

Add zero-state diagnostics for the three non-electricity meters:

- `zero_accuracy`
- `zero_precision`
- `zero_recall`
- `zero_f1`
- `zero_false_positive_rate`
- `zero_false_negative_rate`
- `mean_prediction_on_zero`

Use `p_zero >= 0.5` only for these diagnostics.

Also report thresholded regression metrics:

```text
thresholded_prediction = 0 if p_zero >= 0.5 else prediction
```

The primary regression metrics continue to use the soft prediction. Thresholded metrics are comparison artifacts for deciding whether hard zeroing should become the main output later.

## Rolling Evaluation

Use soft prediction for main metrics.

For history writeback during auto-rolling evaluation:

```text
writeback_prediction = clip(soft_prediction, 0, 1)
```

Do not hard-zero predictions before writing them back in the first implementation. The current data does not show long strict zero runs, and hard writeback could suppress legitimate recovery from zero to small positive usage.

## File Scope

Planned implementation scope:

- `GNN/static_adaptive_graph.py`
  - Keep the current regressor for electricity.
  - Add the zero-inflated regressor for non-electricity meters.
  - Reuse shared graph and encoder code where practical.

- `GNN/static_adaptive_stgnn_gcn_gru_auto-rolling.py`
  - Keep as the electricity static-adaptive script.
  - Fix imports to reference `GNN.static_adaptive_graph`.
  - Keep output artifacts under `GNN/`.

- `GNN/static_adaptive_stgnn_other_meters_auto-rolling.py`
  - Add a new script for chilled water, steam, and hot water.
  - Read `preprocessed_data/meter_1`, `preprocessed_data/meter_2`, and `preprocessed_data/meter_3`.
  - Train one zero-inflated model per meter.
  - Save outputs under `GNN/static_adaptive_stgnn_other_meters_auto-rolling_outputs/meter_{id}`.

Do not change preprocessing data in this iteration.

Do not modify the currently running cloud output artifacts.

## Verification

The implementation is ready when:

1. The electricity smoke test still passes with the standard regression head.
2. A non-electricity smoke test covers:
   - zero/value head output shapes.
   - zero-inflated loss masking for `target_mask == False`.
   - zero samples contributing to zero classification and zero suppression.
   - non-zero samples contributing to regression.
   - rolling evaluation returning soft regression metrics, thresholded metrics, and zero-state diagnostics.
3. The scripts can run with the configured `graduation_project_env` environment.
4. No unrelated project files or generated model artifacts are changed.
