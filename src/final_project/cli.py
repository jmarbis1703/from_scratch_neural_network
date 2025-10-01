from __future__ import annotations
from .data.titanic import load_titanic
from .models.nn_scratch import NNClassifier
from .eval.metrics import binary_report

def fast_titanic(seed: int = 42) -> None:
    data = load_titanic(seed=seed)
    Xtr, Xva, Xte = data.X_train, data.X_val, data.X_test
    ytr, yva, yte = data.y_train, data.y_val, data.y_test

    model = NNClassifier(input_dim=Xtr.shape[1], hidden=16, num_classes=2, seed=seed, lr=5e-3)
    model.fit(Xtr, ytr, Xva, yva, epochs=50, batch_size=64, patience=5, seed=seed)

    proba = model.predict_proba(Xte)[:, 1]
    print(binary_report(yte, proba))

if __name__ == "__main__":
    fast_titanic()
