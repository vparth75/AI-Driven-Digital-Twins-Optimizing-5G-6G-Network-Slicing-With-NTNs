# Phase 1: physical-network simulator

Phase 1 implements the physical network only. It is intentionally independent
of the future Digital Twin and reinforcement-learning layers so its numerical
behaviour can be tested in isolation.

## Paper correspondence

| Paper element | Phase 1 implementation |
|---|---|
| Eq. 1, FBS-to-UE distance | `src/network/geometry.py` computes 3D Euclidean distance. |
| Eq. 2, Rician channel | `src/channel/rician.py` combines deterministic LoS and complex Gaussian NLoS amplitudes, then returns a power gain. |
| Eqs. 5-7, FBS mobility | `src/mobility/models.py` bounds FBS motion to 10 m/s and moves it toward the current UE centroid. |
| Eq. 8, achievable rate | `src/metrics/core.py` applies Shannon capacity with bandwidth, power, gain, noise, and interference. |
| Eqs. 9-10, latency | Packet transmission time plus processing and synchronization-delay terms are averaged across UEs. Synchronization error is zero in Phase 1. |
| Eq. 11, constraints | Bandwidth is rounded to resource blocks and projected to 20 MHz; power is clipped to its configured range. |

## Assumptions required by the paper's omissions

The paper specifies 50 UEs, 20 MHz total bandwidth, 10 m/s FBS speed, and
the model equations. It does not specify the values below, so they are not
paper claims. They live in `configs/paper_config.yaml` and can be changed
without modifying code:

- 1 km by 1 km operating area, 100 m FBS altitude, and 1.5 m UE height.
- 100 resource blocks, so each block is 200 kHz in this simplified model.
- A one-RB-per-UE allocation floor. The paper permits zero bandwidth, but the
  floor avoids unbounded packet latency and gives every eMBB UE a service
  opportunity under an untrained or discrete policy.
- 3.5 GHz carrier, free-space reference gain, path-loss exponent 2.2, and
  Rician K factor 6.
- Orthogonal-RB interference of 0 W, 7 dB receiver noise figure, and 5-23 dBm
  FBS transmit-power bounds.
- Bounded random-walk UEs, log-normal packet traffic, and 1 ms processing
  delay.

The `e_i/c` term in the paper's latency equation has no stated unit for
`e_i`. The simulator treats it as metres of synchronization error, which gives
seconds after division by the speed of light. This term stays zero until the
Digital Twin is introduced in Phase 2.

## How to run and verify

```bash
python3 scripts/run_phase1.py
pytest -q
```

The runner writes `results/phase1_topology.png`. Its default equal-RB policy
is a reference allocation rather than a proportional-fairness baseline; Phase
5 will add fair, comparable baseline policies.
