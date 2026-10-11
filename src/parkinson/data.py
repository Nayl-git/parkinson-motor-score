"""Competition data: downloaded from the organisers' public repository, checked by SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SOURCE = "https://raw.githubusercontent.com/probabl-ai/bobathon-esilv/main/data/{name}"
FILES = {
    "X_train.csv": "29ee38e9b6909e9cf96741853a543a8f4b41fdc90ca175e37155d086960b1668",
    "y_train.csv": "8d8583da6899fa8cbe58b1dd522693f24211da93f6b376ae42ba471394743775",
    "X_test.csv": "d23fb1be60e2bea1ba867e5b25f12ccaef1bac06b76045d9f9b4eee4a3343d88",
}


def _sha256(path: Path) -> str:
    # Line endings normalised, so a copy downloaded on Windows matches too.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = data_dir / name
        if not path.is_file():
            print(f"downloading {name}")
            urllib.request.urlretrieve(SOURCE.format(name=name), path)
        if _sha256(path) != digest:
            raise ValueError(f"{name}: unexpected content (SHA-256 mismatch), delete it and retry")


def load(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training visits (with `target`) and test visits."""
    download(data_dir)
    x_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    x_test = pd.read_csv(data_dir / "X_test.csv")
    return x_train.merge(y_train, on="Index"), x_test


if __name__ == "__main__":
    download()
    print("data ready in", DATA_DIR)
