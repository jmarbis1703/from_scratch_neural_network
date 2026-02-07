from __future__ import annotations
import numpy as np

Array = np.ndarray

def to_one_hot(y: Array, num_classes: int) -> Array:
    """Converts integer class labels to one-hot encoded vectors."""
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def softmax(z: Array) -> Array:
    """Computes stable softmax values for each set of scores in z."""
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)  
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs: Array, y_true_oh: Array) -> float:
    """Calculates Mean Cross Entropy Loss."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -np.sum(y_true_oh * np.log(p)) / y_true_oh.shape[0]

class LayerDense:
    """ Fully connected layer implementing He Initialization and Adam optimizer state."""
    def __init__(self, n_inputs: int, n_neurons: int, rng: np.random.Generator):
        limit = np.sqrt(2.0 / n_inputs)
        self.weights = rng.normal(0.0, limit, size=(n_inputs, n_neurons)).astype(np.float64)
        self.biases = np.zeros((1, n_neurons), dtype=np.float64)
        
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        
        self.inputs: Array | None = None
        self.output: Array | None = None

        self.w_m = np.zeros_like(self.weights)
        self.w_v = np.zeros_like(self.weights)
        self.b_m = np.zeros_like(self.biases)
        self.b_v = np.zeros_like(self.biases)

    def forward(self, inputs: Array) -> None:
        """ Forward pass. """
        self.inputs = np.asarray(inputs, dtype=np.float64)
        self.output = self.inputs @ self.weights + self.biases

    def backward(self, dvalues: Array) -> Array:
        """ Backward pass. Calculates gradients and returns error for previous layer. """
        dvalues = np.asarray(dvalues, dtype=np.float64)
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        return dvalues @ self.weights.T

class ReLU:
    """Rectified Linear Unit activation."""
    def forward(self, inputs: Array) -> None:
        self.inputs = np.asarray(inputs, dtype=np.float64)
        self.output = np.maximum(0.0, self.inputs)

    def backward(self, dvalues: Array) -> Array:
        dvalues = np.asarray(dvalues, dtype=np.float64)
        return dvalues * (self.inputs > 0.0)

class OptimizerAdam:
    """ Adam Optimizer implementation. """
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0

    def step(self):
        """Increments the time step once per batch."""
        self.t += 1

    def update_params(self, layer: LayerDense) -> None:
        for w, dw, m, v in (
            (layer.weights, layer.dweights, layer.w_m, layer.w_v),
            (layer.biases, layer.dbiases, layer.b_m, layer.b_v),
        ):
            m[...] = self.b1 * m + (1.0 - self.b1) * dw
            v[...] = self.b2 * v + (1.0 - self.b2) * (dw * dw)
            m_hat = m / (1.0 - self.b1 ** self.t)
            v_hat = v / (1.0 - self.b2 ** self.t)
            w[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class NNClassifier:
    def __init__(self, input_dim: int, hidden: int, num_classes: int, seed: int = 42, lr: float = 1e-3):
        self.rng = np.random.default_rng(seed)
        self.l1 = LayerDense(input_dim, hidden, self.rng)
        self.a1 = ReLU()
        self.l2 = LayerDense(hidden, num_classes, self.rng)
        self.opt = OptimizerAdam(lr=lr)
        self.history: list[dict] = []

    def _forward_logits(self, X: Array) -> Array:
        self.l1.forward(X)
        self.a1.forward(self.l1.output)
        self.l2.forward(self.a1.output)
        return self.l2.output

    def predict_proba(self, X: Array) -> Array:
        return softmax(self._forward_logits(X))

    def predict(self, X: Array) -> Array:
        return np.argmax(self._forward_logits(X), axis=1)

    def fit(self, X_tr, y_tr, X_va, y_va, epochs=20, batch_size=128, patience=3):
        y_tr_oh = to_one_hot(y_tr, self.l2.weights.shape[1])
        y_va_oh = to_one_hot(y_va, self.l2.weights.shape[1])
        best_loss = np.inf
        wait = 0

        print(f"Training on {len(X_tr)} samples, Validating on {len(X_va)} samples...")
        for epoch in range(epochs):
            idx = self.rng.permutation(len(X_tr))
            Xb_shuffled, yb_shuffled = X_tr[idx], y_tr_oh[idx]

            for i in range(0, len(Xb_shuffled), batch_size):
                xb = Xb_shuffled[i : i + batch_size]
                yb = yb_shuffled[i : i + batch_size]      
              
                logits = self._forward_logits(xb)
                
                dlogits = (softmax(logits) - yb) / yb.shape[0]
                d_l2 = self.l2.backward(dlogits)
                d_a1 = self.a1.backward(d_l2)
                _    = self.l1.backward(d_a1)

                self.opt.step()
                self.opt.update_params(self.l1)
                self.opt.update_params(self.l2)

            # Evaluation
            tr_loss = cross_entropy(self.predict_proba(X_tr), y_tr_oh)
            va_loss = cross_entropy(self.predict_proba(X_va), y_va_oh)
            acc = np.mean(self.predict(X_va) == y_va)
            
            print(f"Epoch {epoch+1:02d}: Train Loss {tr_loss:.4f}, Val Loss {va_loss:.4f}, Val Acc {acc:.4f}")
            self.history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

            if va_loss < best_loss:
                best_loss = va_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
