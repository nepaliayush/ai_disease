"""Download the five public, anonymized datasets used in the project.

Clinical datasets:
  - Diabetes  (Pima Indians Diabetes Database, UCI)            -> diabetes.csv
  - Heart     (Cleveland Heart Disease, UCI)                   -> heart.csv
  - Liver     (Indian Liver Patient Dataset, UCI)              -> liver.csv
  - CKD       (Chronic Kidney Disease, UCI)                    -> ckd.csv

Symptom dataset:
  - Disease-Symptom dataset (132 symptoms x 41 diseases, public
    Kaggle "Disease Prediction Using Machine Learning")        -> symptoms.csv

Only public, anonymized datasets are used (UCI / Kaggle mirrors). Run:
    python -m scripts.download_data
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

from app.config import RAW_DATA_DIR

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "diabetes.csv": [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    ],
    "heart.csv": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    ],
    "liver.csv": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
    ],
    "ckd.csv": [
        "https://uob-ds.github.io/cfd2021/_downloads/907917c3cf8500a73493d9cc1bce6076/ckd_full.csv",
        "https://lisds.github.io/textbook/_downloads/907917c3cf8500a73493d9cc1bce6076/ckd_full.csv",
    ],
    "symptoms.csv": [
        "https://raw.githubusercontent.com/parthsompura/Disease-prediction-using-Machine-Learning/master/Training.csv",
    ],
}


def download(name: str, urls: list[str]) -> Path:
    out = RAW_DATA_DIR / name
    if out.exists() and out.stat().st_size > 0:
        print(f"  [skip] {name} already present")
        return out
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = r.text
            if not text.strip():
                continue
            out.write_text(text, encoding="utf-8")
            print(f"  [ok]   {name} <- {url}")
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {url} -> {exc}")
    raise RuntimeError(f"Could not download {name} from any source")


def main() -> None:
    print("Downloading datasets to", RAW_DATA_DIR)
    for name, urls in SOURCES.items():
        download(name, urls)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())