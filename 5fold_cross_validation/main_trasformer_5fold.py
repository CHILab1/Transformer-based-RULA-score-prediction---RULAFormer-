"""Compatibility entrypoint for the corrected, shared implementation."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spectre.cli import main
    main(default_protocol="group5")
