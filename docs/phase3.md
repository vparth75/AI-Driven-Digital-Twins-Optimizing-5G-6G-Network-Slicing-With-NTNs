# Phase 3: DDPG

This phase implements DDPG directly with PyTorch modules and autograd. It does
not use a reinforcement-learning library and is deliberately independent of
the Digital Twin until Phase 4.

`Actor` uses the paper's 256/128 ReLU hidden layers by default and produces a
continuous, tanh-bounded action. `Critic` consumes state and action together,
then uses 512/256/128 ReLU layers to estimate Q-values. `DDPGAgent` owns live
and target networks, Adam optimizers, replay buffer, OU noise, Bellman MSE
updates, deterministic-policy updates, and tau=0.001 soft target updates.

`scripts/train_toy_ddpg.py` validates the agent first on a one-dimensional
continuous action task. It uses smaller 64/32 layers only for that toy test;
this does not change the configured network-slicing architecture.

## Dependency note

PyTorch is declared in `requirements.txt`. Run this phase with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_toy_ddpg.py
```

The paper provides actor/critic widths, batch size, target tau, and OU sigma
schedule. Discount factor, optimizer learning rates, replay capacity, and OU
theta are explicit assumptions in `paper_config.yaml`.
