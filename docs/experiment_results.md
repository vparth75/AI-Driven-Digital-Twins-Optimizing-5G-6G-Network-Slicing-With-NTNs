# Experiment results

## Reproducibility

Run the full comparison and then create its chart from the project root:

```bash
python scripts/run_all_experiments.py
python scripts/plot_comparison.py
```

The raw aggregate measurements are stored in
`results/comparison/summary.csv`. The chart is stored in
`results/comparison/method_comparison.png`.

## First completed comparison

The first completed comparison used the simulation configuration in
`configs/paper_config.yaml`, including 50 UEs, 20 MHz bandwidth, and ten
evaluation episodes per method. Values are mean plus/minus standard deviation.

| Method | Latency (ms) | Throughput (Mbps) | Jitter (ms) | Resource utilization (%) |
|---|---:|---:|---:|---:|
| DT-DDPG | 340.51 +/- 10.01 | 160.38 +/- 3.72 | 146.83 +/- 6.98 | 100.00 +/- 0.00 |
| Standalone DDPG | 248.47 +/- 6.00 | 195.23 +/- 3.36 | 102.40 +/- 3.62 | 100.00 +/- 0.00 |
| Proportional Fairness | 228.49 +/- 3.31 | 199.36 +/- 3.50 | 149.07 +/- 2.61 | 99.08 +/- 0.11 |
| Q-learning | 209.80 +/- 3.93 | 206.10 +/- 3.89 | 89.03 +/- 3.11 | 100.00 +/- 0.00 |
| DQN | 205.97 +/- 4.38 | 205.44 +/- 3.90 | 85.20 +/- 4.35 | 100.00 +/- 0.00 |

## Interpretation

In this run, DQN has the lowest latency and jitter. Q-learning has the highest
throughput. DT-DDPG performs worst on latency and throughput. Therefore, this
experiment does **not** support a claim that DT-DDPG outperforms every
baseline. The correct conclusion is that the current DT-DDPG configuration
requires further tuning and validation.

## Limitations and next research iteration

- The reported standard deviations represent ten evaluation episodes, not
  independent full training runs. Repeat the complete training procedure with
  several random seeds before making a strong statistical claim.
- Q-learning and DQN use the repository's centralized ten-action allocation
  catalog, not an intractable 10^50 per-UE joint action space.
- DT-DDPG should be tuned using a documented experiment plan: learning rates,
  reward weights, synchronization interval, traffic predictor, and exploration
  schedule should be varied one at a time and evaluated using the same seeds.

## Digital-Twin synchronization sensitivity study

`scripts/tune_dt_ddpg.py` begins that tuning process by varying only the
Digital Twin synchronization interval. It keeps the model architecture,
reward, training duration, and evaluation seeds fixed. This isolates the
effect of synchronization frequency and avoids selecting a configuration from
an uncontrolled collection of changes. A three-seed, 300-episode screening
study can be run with:

```bash
python scripts/tune_dt_ddpg.py --episodes 300 --seeds 3
```

It writes both raw seed-level metrics and an aggregate summary below
`results/tuning/`; the filename includes the episode count, seed count, and
tested synchronization intervals so separate studies are never overwritten. A
candidate should only be promoted to a paper-length
1,000-episode run if it improves latency consistently across training seeds.

### Screening result

The completed 300-episode, three-seed screening study produced the following
result. Values are mean plus/minus standard deviation over independent
training seeds.

| Twin synchronization interval | Latency (ms) | Throughput (Mbps) | Jitter (ms) |
|---|---:|---:|---:|
| Every 1 step | **247.88 +/- 10.16** | **201.91 +/- 4.40** | **108.60 +/- 5.50** |
| Every 2 steps | 263.52 +/- 13.23 | 194.52 +/- 2.68 | 116.28 +/- 6.68 |
| Every 5 steps (original configuration) | 255.19 +/- 3.03 | 198.17 +/- 2.55 | 115.93 +/- 1.02 |

Synchronizing the Twin every simulator step was best on all three measured
metrics in this screening study. It should be the candidate taken forward to
a 1,000-episode confirmation run; this screening result alone should not be
compared directly with the earlier 1,000-episode five-method comparison.

The confirmation command runs only the winning setting, without rerunning the
two weaker candidates:

```bash
python scripts/tune_dt_ddpg.py --episodes 1000 --seeds 3 --intervals 1
```

Create the report chart with:

```bash
python scripts/plot_tuning.py
```

### 1,000-episode confirmation

The selected setting (synchronization every one step) was retrained for 1,000
episodes with the same three independent training seeds. Its result was
**272.46 +/- 28.27 ms latency**, **191.76 +/- 13.67 Mbps throughput**, and
**119.42 +/- 8.59 ms jitter**.

This is a valid, more rigorous result for the tuned DT-DDPG configuration. It
does not outperform DQN, Q-learning, or Proportional Fairness in the earlier
five-method comparison, so the report must not claim a universal DT-DDPG
improvement. The tuning result instead shows that synchronization frequency is
an important factor and that further work is needed on the reward design and
traffic/channel prediction before making a performance claim.

Create its separate chart with:

```bash
python scripts/plot_tuning.py \
  --input results/tuning/dt_ddpg_sync_sensitivity_1000ep_3seeds_sync1_summary.csv
```
