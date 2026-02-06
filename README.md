# From-Scratch Deep Learning Engine (NumPy)

**A pure NumPy implementation of a Deep Neural Network, featuring custom backpropagation, Adam optimization, and PyTorch verification.**

A deep learning training engine built from scratch to demonstrate the mathematical foundations of neural networks—no TensorFlow or PyTorch for the core logic.

## Key Features
- **Custom Training Engine**: Vectorized forward and backward propagation using NumPy
- **Adam Optimizer**: Built from scratch without external libraries
- **Verified Accuracy**: Gradients mathematically validated against PyTorch
- **Real Performance**: Achieves >95% accuracy on MNIST handwritten digits

## Quickstart

### 1. Install
```bash
pip install -e .
```

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

## Implementation Details
- **LayerDense**: Implements He initialization and caches inputs for the backward pass
- **Backpropagation**: Calculates partial derivatives via the chain rule, passing error terms (`dinputs`) backward layer by layer
- **Unit Tests**: Includes finite difference gradient checks and PyTorch equivalence tests
