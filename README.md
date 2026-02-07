# NumPy Neural Engine: Deep Learning from First Principles

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/dependencies-lightweight-orange)

**A mathematically rigorous deep learning framework built entirely in NumPy.**

A deep learning framework built entirely in NumPy, implementing automatic differentiation, optimization, and weight initialization without relying on autograd frameworks.

Gradients have been verified against PyTorch's Autograd to within `1e-9` precision.

##  Core Features

* **Vectorized Backpropagation:** Manual chain rule implementation with efficient batched gradient computation.
* **Adam Optimizer:** — From-scratch implementation following the Kingma & Ba paper, with explicit momentum (`beta1`) and RMSProp (`beta2`) updates.
* **He Initialization:** Proper weight scaling for ReLU networks to avoid vanishing/exploding gradients.
* **Tested & Typed:** Unit tests, gradient verification suite, type hints, and CI/CD.

## Results

Benchmarked on MNIST with a (784 → 64 → 10 architecture).

| Metric | Result |
| :--- | :--- |
| **Accuracy** | >96% (Validation) |
| **Convergence** | < 10 Epochs |
| **Gradient Check** | **Verified vs PyTorch** |

<p align="center">
  <img src="mnist_training_curve.png" alt="Training Curve" width="600">
  <br>
  <i>Figure 1: Cross-Entropy Loss convergence using custom Adam optimizer.</i>
</p>

### Gradient Verification (PyTorch)
To prove the math is correct, I wrote a test suite that initializes this engine and a PyTorch model with identical weights, feeds the same batch, and compares the gradients.

```python
# from tests/compare_torch.py
diff = np.abs(grad_custom - grad_pt).max()
print(f"Max Gradient Difference: {diff:.9f}")
# Output: Max Gradient Difference: 0.000000000
```

## Getting Started 
### 1. Installation
The core enging requires only NumPy.
``` bash
git clone [https://github.com/jmarbis1703/from_scratch_neural_network.git](https://github.com/jmarbis1703/from_scratch_neural_network.git)
cd from_scratch_neural_network

# Install core engine
pip install -e .

# (Optional) Install PyTorch only if you want to run the verification tests
pip install -e ".[dev]"
```

### 2. Training
Train the model on MNIST:
``` bash
python -m final_project.cli
```
### 3. Run Gradient Check
Verify the backpropagation calculus:
``` bash
pytest tests/compare_torch.py
```

## Implementation Detail: The Adam Step
The optimizer logic was built to mirror the original Kingma & Ba (2014):
``` python
# from src/final_project/models/nn_scratch.py
m[...] = self.b1 * m + (1.0 - self.b1) * dw
v[...] = self.b2 * v + (1.0 - self.b2) * (dw * dw)
m_hat = m / (1.0 - self.b1 ** self.t)
v_hat = v / (1.0 - self.b2 ** self.t)
w[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```
## Project Structure
```
├── src/final_project/
│   ├── models/nn_scratch.py   # The Engine (Layers, Optimizer, Backprop)
│   ├── data/mnist.py          # Data Pipeline
│   └── eval/metrics.py        # Performance Metrics
├── tests/
│   ├── compare_torch.py       # PyTorch Parity Tests (The standard)
│   └── test_grad_check.py     # Finite Difference Checks
```




