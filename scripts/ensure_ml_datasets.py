#!/usr/bin/env python3
"""Extract routing model CSVs from data/ml_datasets/ml_datasets.zip when missing."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml_datasets"
ZIP_PATH = DATA_DIR / "ml_datasets.zip"
REQUIRED = (
    "routing_nodes.csv",
    "routing_edges.csv",
    "congestion_ml.csv",
)


def main() -> int:
    missing = [name for name in REQUIRED if not (DATA_DIR / name).is_file()]
    if not missing:
        print("ML datasets already present.")
        return 0

    if not ZIP_PATH.is_file():
        print(f"Missing archive: {ZIP_PATH}")
        print("Required CSV files:", ", ".join(missing))
        return 1

    print(f"Extracting {ZIP_PATH.name} into {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    still_missing = [name for name in REQUIRED if not (DATA_DIR / name).is_file()]
    if still_missing:
        print("After extraction, still missing:", ", ".join(still_missing))
        return 1

    print("ML datasets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
