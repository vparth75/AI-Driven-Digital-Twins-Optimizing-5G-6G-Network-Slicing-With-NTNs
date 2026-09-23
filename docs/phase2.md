# Phase 2: Digital Twin

`DigitalTwin` is a virtual layer over the physical `NetworkSimulator`; it does
not change the physical state or apply actions to it. Its `advance()` method
receives a physical snapshot on every simulator step and behaves in two modes:

1. Every configured synchronization interval, it receives UE positions,
   FBS position/velocity, Rician channel gains (CSI), and traffic information.
   The FBS estimate is corrected by a 3D position-and-velocity Kalman filter,
   as described by paper Eqs. 3-4.
2. Between updates, it extrapolates FBS/UE motion, predicts distance-driven
   channel change from the last CSI, and carries its EMA traffic estimate.

The state has ten normalized features per UE: CSI/channel gain, predicted
traffic, 3D UE position, 3D FBS position, synchronization error, and CSI. The
duplicate channel/CSI representation is intentional: Algorithm 1 lists both,
whereas Eq. 14 lists channel gain but omits CSI.

The virtual state also retains the most recently observed packet sizes. They
are deliberately distinct from the EMA traffic forecast used in the policy
state and action simulation.

`simulate()` projects an allocation to valid RB and power constraints, then
uses the virtual channel and predicted demand to compute rates, latency,
throughput, utilization, and paper-style reward. This permits safe offline
action evaluation, without applying that action to the physical simulator.

## Assumptions

The paper does not give a numeric synchronization interval, Kalman covariance,
power-reward coefficient beta, or units compatible with its `e_i / c` latency
term after defining `e_i` as traffic-prediction error. These choices are
explicit in `configs/paper_config.yaml`. In particular, sync error is kept in
bits for the state and reward; a configurable conversion to metres is applied
only for the propagation-delay term.

Run `python3 scripts/demo_digital_twin.py` to see the physical-to-twin flow.
