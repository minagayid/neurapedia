"""Run the dependency-light Neurapedia research demonstration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurapedia.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["demo", "--shape", "48", "48", "24", "--output", "neurapedia-report.html"]))
