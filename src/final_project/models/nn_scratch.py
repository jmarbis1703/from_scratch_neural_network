from __future__ import annotations
import numpy as np

Array = np.ndarray

def to_one_hot(y: Array, num_classes: int) -> Array:
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def softmax(z: Array) -> Array:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float32)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs: Array, y_true_oh: Array) -> Array:
    p = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return -np.sum(y_true_oh * np.log(p), axis=1)

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int, rng: np.random.Generator):
        limit = np.sqrt(6.0 / (n_inputs + n_neurons))
        self.weights = rng.uniform(-limit, limit, size=(n_inputs, n_neurons)).astype(np.float32)
        self.biases = np.zeros((1, n_neurons), dtype=np.float32)
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.inputs = None
        self.output = None
        self.w_m = np.zeros_like(self.weights)
        self.w_v = np.zeros_like(self.weights)
        self.b_m = np.zeros_like(self.biases)
        self.b_v = np.zeros_like(self.biases)

    def forward(self, inputs: Array) -> None:
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

    def backward(self, dvalues: Array) -> None:
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues @ self.weights.T

class ReLU:
    def forward(self, inputs: Array) -> None:
        self.inputs = inputs
        self.output = np.maximum(0.0, inputs)

    def backward(self, dvalues: Array) -> None:
        self.dinputs = dvalues * (self.inputs > 0.0)

class OptimizerAdam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0

    def update_params(self, layer: LayerDense) -> None:
        self.t += 1
        for w, dw, m, v in ((layer.weights, layer.dweights, layer.w_m, layer.w_v),
                            (layer.biases,  layer.dbiases, layer.b_m, layer.b_v)):
            m[...] = self.b1 * m + (1 - self.b1) * dw
            v[...] = self.b2 * v + (1 - self.b2) * (dw * dw)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            w[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class NNClassifier:
    def __init__(self, input_dim: int, hidden: int, num_classes: int, seed: int = 42, lr: float = 1e-3):
        self.rng = np.random.default_rng(seed)
        self.l1 = LayerDense(input_dim, hidden, self.rng)
        self.a1 = ReLU()
        self.l2 = LayerDense(hidden, num_classes, self.rng)
        self.opt = OptimizerAdam(lr=lr)

    def _forward_logits(self, X: Array) -> Array:
        self.l1.forward(X); self.a1.forward(self.l1.output)
        self.l2.forward(self.a1.output)
        return self.l2.output

    def predict_proba(self, X: Array) -> Array:
        logits = self._forward_logits(X)
        return softmax(logits)

    @property
    def num_classes(self) -> int:
        return self.l2.weights.shape[1]

    def fit(self, X_tr: Array, y_tr: Array, X_va: Array, y_va: Array,
            epochs: int = 300, batch_size: int = 128, patience: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_tr_oh = to_one_hot(y_tr, self.num_classes)
        y_va_oh = to_one_hot(y_va, self.num_classes)
        best_val = np.inf
        best = None
        wait = 0
        for epoch in range(epochs):
            idx = rng.permutation(len(X_tr))
            Xb, yb = X_tr[idx], y_tr_oh[idx]
            for i in range(0, len(Xb), batch_size):
                xb = Xb[i : i + batch_size]
                yb_ = yb[i : i + batch_size]
                logits = self._forward_logits(xb)
                probs = softmax(logits)
                dlogits = (probs - yb_) / yb_.shape[0]
                self.l2.backward(dlogits); self.a1.backward(self.l2.dinputs); self.l1.backward(self.a1.dinputs)
                self.opt.update_params(self.l1); self.opt.update_params(self.l2)
            # validation
            val_probs = self.predict_proba(X_va)
            val_loss = cross_entropy(val_probs, y_va_oh).mean()
            if val_loss < best_val - 1e-4:
                best_val, wait = val_loss, 0
                best = (self.l1.weights.copy(), self.l1.biases.copy(), self.l2.weights.copy(), self.l2.biases.copy())
            else:
                wait += 1
                if wait >= patience:
                    break
        if best is not None:
            (self.l1.weights, self.l1.biases, self.l2.weights, self.l2.biases) = best

    def save(self, path: str) -> None:
        np.savez_compressed(path, l1w=self.l1.weights, l1b=self.l1.biases, l2w=self.l2.weights, l2b=self.l2.biases)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.l1.weights = data["l1w"]; self.l1.biases = data["l1b"]
        self.l2.weights = data["l2w"]; self.l2.biases = data["l2b"]
