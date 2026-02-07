# From-Scratch Deep Learning Engine (NumPy)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

**A pure NumPy implementation of a Deep Neural Network, featuring custom backpropagation, Adam optimization, and PyTorch verification.**

A deep learning training engine built from scratch to demonstrate the mathematical foundations of neural networks—no TensorFlow or PyTorch for the core logic.

## Key Features
- **Custom Training Engine**: Vectorized forward and backward propagation using NumPy
- **Adam Optimizer**: Built from scratch without external libraries
- **Verified Accuracy**: Gradients mathematically validated against PyTorch
- **Real Performance**: Achieves >95% accuracy on MNIST handwritten digits

## Quickstart

### 1. Installation 
Clone the repo and set up a virtual environment:
```bash
# Clone the repo
git clone [https://github.com/jmarbis1703/from_scratch_neural_network.git](https://github.com/jmarbis1703/from_scratch_neural_network.git)
cd from_scratch_neural_network

# Create and activate virtual environment (Linux/Mac)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (NumPy, PyTorch for verification, etc.)
pip install -e .
```
Windows Users: The activation command differs on Windows. Run .\venv\Scripts\activate instead of source venv/bin/activate.

### 2. Run the Demo
Train the network on the MNIST dataset (784 → 64 → 10):
```bash
python -m final_project.cli
```
Outputs `mnist_training_curve.png` showing loss convergence.

### 3. Verify Math / Gradient Check
Run the comparison script to prove the NumPy gradients match PyTorch's autodiff engine:
```bash
pytest tests/compare_torch.py
```
## Project Structure
from_scratch_neural_network/
├── src/final_project/
│   ├── models/nn_scratch.py   # Core: LayerDense, ReLU, Adam, Softmax
│   ├── data/mnist.py          # Data loading & preprocessing
│   ├── eval/metrics.py        # Classification reports
│   └── cli.py                 # Training entry point
├── tests/
│   ├── compare_torch.py       # PyTorch vs NumPy verification
│   ├── test_grad_check.py     # Finite difference checks
│   └── test_softmax_loss.py   # Unit tests for loss functions
├── pyproject.toml             # Dependencies
└── README.md

## Implementation Details
- **LayerDense**: Implements He initialization and caches inputs for the backward pass
- **Backpropagation**: Calculates partial derivatives via the chain rule, passing error terms (`dinputs`) backward layer by layer
- **Optimizer** Custom implementation of Adam with momentum and RMSProp correction.
- **Unit Tests**: Includes finite difference gradient checks and PyTorch equivalence tests
