# Phase 4: DT-DDPG integration

`DTDDPGEnvironment` is the integration boundary between the three completed
layers. On every decision step it follows this sequence:

```text
Digital Twin state -> continuous actor action -> RB/power projection
-> Digital Twin simulation and reward -> physical-network application
-> next physical observation -> DT synchronization or prediction
```

The first M actor outputs are normalized into bandwidth preferences; the next
M become powers between configured limits. The shared physical projection then
rounds bandwidth to valid resource blocks and enforces the total-bandwidth
constraint. The Twin predicts reward before the same feasible allocation is
applied to the physical simulator. Per-step logs retain both predicted and
actual latency/throughput, allocation, channel, traffic, synchronization error,
and utilization. This preserves the Digital Twin's distinct role rather than
turning it into a simple environment wrapper.

`scripts/train_dt_ddpg.py` contains the paper-style replay/update training
loop and saves checkpoints plus the final episode's CSV logs. It requires
PyTorch. DDPG training has not yet been run locally because PyTorch is not
available in the active environment.
