#!/usr/bin/env python3
"""Evaluate L1/L2 Global PairAcc using the bundled CLaM-H implementation."""

from pathlib import Path
import runpy
import sys

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party/clam_h"
sys.path.insert(0, str(VENDOR_ROOT))

if __name__ == "__main__":
    runpy.run_module("motionspeech_eval.evaluate_generated_pairacc", run_name="__main__")
