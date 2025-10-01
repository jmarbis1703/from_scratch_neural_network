from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TitanicData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def _synthetic_titanic(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pclass = rng.integers(1, 4, size=n)
    female = rng.integers(0, 2, size=n)
    age = np.clip(rng.normal(30 + 10*(pclass==1) - 5*female, 14, size=n), 1, 80)
    sibsp = rng.integers(0, 3, size=n)
    logit = 1.5*(female==1) - 0.7*(pclass-1) - 0.02*(age-30) - 0.3*sibsp
    prob = 1/(1+np.exp(-logit))
    survived = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"pclass": pclass, "female": female, "age": age, "sibsp": sibsp, "survived": survived})
    return df


def load_titanic(csv_path: Path | str = "Titanic Dataset.csv", seed: int = 42) -> TitanicData:
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        expected = {"pclass", "sex", "age", "sibsp", "survived"}
        if not expected.issubset(set(df.columns)):
            raise ValueError(f"CSV must have columns {expected}, got {set(df.columns)}")
        df = df[list(expected)].dropna()
        df["female"] = (df["sex"].astype(str).str.lower() == "female").astype(int)
    else:
        df = _synthetic_titanic(n=1200, seed=seed)

    X = df[["pclass", "female", "age", "sibsp"]].to_numpy(dtype=np.float32)
    y = df["survived"].to_numpy(dtype=np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    return TitanicData(X_tr, X_va, X_te, y_tr, y_va, y_te, scaler)
