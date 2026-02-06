import numpy as np
import torch
import torch.nn as nn
from final_project.models.nn_scratch import LayerDense, ReLU, softmax, cross_entropy


#   Verifies that the custom NumPy implementation produces IDENTICAL gradients to PyTorch when initialized with the same weights.
def test_compare_vs_pytorch(): ## 
    print("\n PyTorch Comparison Test ")
    
    N, D_in, H, D_out = 64, 50, 32, 3
    rng = np.random.default_rng(42)
    X_np = rng.normal(size=(N, D_in)).astype(np.float32)
    y_np = rng.integers(0, D_out, size=N)
    
    # Custom Model
    l1 = LayerDense(D_in, H, rng)
    a1 = ReLU()
    l2 = LayerDense(H, D_out, rng)
    
    # PyTorch Model
    model_pt = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out)
    )
    
    # Weights
    with torch.no_grad():
        model_pt[0].weight.copy_(torch.from_numpy(l1.weights.T))
        model_pt[0].bias.copy_(torch.from_numpy(l1.biases.reshape(-1)))
        model_pt[2].weight.copy_(torch.from_numpy(l2.weights.T))
        model_pt[2].bias.copy_(torch.from_numpy(l2.biases.reshape(-1)))

    l1.forward(X_np)
    a1.forward(l1.output)
    l2.forward(a1.output)
    out_custom = softmax(l2.output)
    
    out_pt_logits = model_pt(torch.from_numpy(X_np))
    loss_pt = nn.CrossEntropyLoss()(out_pt_logits, torch.from_numpy(y_np).long())

    y_oh = np.zeros((N, D_out))
    y_oh[np.arange(N), y_np] = 1.0
    dlogits = (out_custom - y_oh) / N
    l1.backward(a1.backward(l2.backward(dlogits)))
    
    loss_pt.backward()

    grad_custom = l1.dweights
    grad_pt = model_pt[0].weight.grad.detach().numpy().T
    
    diff = np.abs(grad_custom - grad_pt).max()
    print(f"Max Gradient Difference: {diff:.9f}")
    
    assert diff < 1e-5, "Gradients do not match!"
    print(" NumPy engine matches PyTorch engine.")

if __name__ == "__main__":
    test_compare_vs_pytorch()
