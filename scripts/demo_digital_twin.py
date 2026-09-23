"""Show physical state -> DT synchronization/prediction -> allocation simulation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.digital_twin import DigitalTwin
from src.network import NetworkSimulator


def main() -> None:
    config = load_config("configs/paper_config.yaml")
    physical = NetworkSimulator(config)
    twin = DigitalTwin(physical)
    twin.reset(physical.reset())
    print("step | DT mode     | mean sync error (kb) | simulated latency (ms) | reward")
    print("-----+-------------+----------------------+------------------------+--------")
    for _ in range(12):
        actual = physical.step()
        state = twin.advance(actual)
        estimate = twin.simulate(physical.default_allocation())
        mode = "synchronized" if state.is_synchronized else "predicted"
        print(
            f"{state.step:4d} | {mode:11s} | {abs(state.synchronization_error_bits).mean() / 1e3:20.2f} | "
            f"{estimate.average_latency_s * 1e3:22.2f} | {estimate.reward:6.2f}"
        )


if __name__ == "__main__":
    main()
