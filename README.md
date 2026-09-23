# AI-Driven Digital Twin 5G/6G NTN Slicing

## Current status: Phase 5 implemented

This repository now has a reproducible, modular **physical-network simulator**
for the first two phases of an implementation based on Ali and Arslan's 2026
paper. It now includes a predictive Digital Twin, standalone PyTorch DDPG,
the DT-mediated environment, and all four requested baseline implementations.

Run the Phase 1 simulation:

```bash
python3 scripts/run_phase1.py
python3 scripts/demo_digital_twin.py
python3 scripts/train_toy_ddpg.py
python3 scripts/train_dt_ddpg.py
python3 scripts/run_proportional_fairness.py
python3 scripts/run_q_learning.py
pytest -q
```

The run writes `results/phase1_topology.png`, showing the UAV/FBS, the 50 UEs,
their links, and Rician channel power gain. Configuration lives in
`configs/paper_config.yaml`; every unsupported parameter is identified there as
an implementation assumption. The Phase 1 model implements paper Eqs. 1, 2,
and 8-11: 3D distance, Rician fading, Shannon rate, latency, and resource
constraints.

See [the Phase 1 technical note](docs/phase1.md) for equation mapping, all
implementation assumptions, and verification details.
See [the Phase 2 technical note](docs/phase2.md) for synchronization,
prediction, state construction, and Digital-Twin action simulation.
See [the Phase 3 technical note](docs/phase3.md) for the PyTorch DDPG design
and the toy continuous-control validation.
See [the Phase 4 technical note](docs/phase4.md) for the integrated DT-DDPG
decision loop and logged metrics.
See [the Phase 5 technical note](docs/phase5.md) for equivalent baseline
conditions and the scalable discrete-action assumption.
