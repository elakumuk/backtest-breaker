"""Print a small offline LLM response and Ken French dataset summary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bb.data import DataGate
from bb.llm import chat


def main() -> None:
    """Run the D1 smoke demonstration."""
    response = chat([{"role": "user", "content": "Say hello in 5 words"}])
    print(response.content)
    returns, rf = DataGate().in_sample()
    print(f"Returns shape: {returns.shape}; RF shape: {rf.shape}")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")


if __name__ == "__main__":
    main()
