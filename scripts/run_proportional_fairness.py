"""Run the PF baseline once under the shared physical simulation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.runner import run_proportional_fair
from src.config import load_config


if __name__ == "__main__":
    metrics, _ = run_proportional_fair(load_config("configs/paper_config.yaml"), seed=42)
    print(metrics)
