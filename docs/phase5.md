# Phase 5: baseline algorithms

All baselines use the same `NetworkSimulator`, configuration, RB projection,
power limits, traffic process, and seeds as DT-DDPG. The only difference is
their allocation policy.

Every method also receives the simulator's documented one-RB-per-UE floor;
this prevents a discrete or early-learning action from assigning zero rate and
creating unbounded latency. It is applied uniformly, not as an advantage for a
particular method.

- **Standalone DDPG:** runs the same continuous DDPG agent directly against
  the physical-only environment, with no Digital Twin prediction layer.
- **Proportional Fairness:** allocates bandwidth according to instantaneous
  demand divided by exponentially averaged achieved rate.
- **Q-learning:** uses the paper's learning rate 0.01, epsilon 0.1, and ten
  discrete allocation levels.
- **DQN:** learns over exactly the same ten discrete choices as Q-learning.

The paper says Q-learning has ten bandwidth levels *per UE*. At 50 UEs this is
an intractable joint space of 10^50 actions. The implementation therefore uses
a documented, centralized ten-policy catalog: action 0 is equal bandwidth and
higher actions increasingly prioritize joint traffic demand and channel gain.
This keeps comparison conditions equivalent while making tabular Q-learning
computationally possible. It is an implementation assumption, not a paper
claim.

PyTorch is required for standalone DDPG and DQN. The PF and Q-learning scripts
run without it. A five-method comparison must wait until all five methods can
run; it must not substitute or fabricate missing results.

When PyTorch is available, `scripts/run_all_experiments.py` trains/evaluates
all five methods over the same evaluation seeds and writes unplotted summary
CSV/JSON under `results/comparison/`. It fails before writing results if any
PyTorch-dependent method cannot run.
