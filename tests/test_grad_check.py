import numpy as np
from final_project.models.nn_scratch import LayerDense, softmax, cross_entropy, to_one_hot

def test_gradient_check_dense():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 3)).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    yoh = to_one_hot(y, 2)

    layer = LayerDense(3, 2, rng)
    eps = 1e-5

    layer.forward(X)
    logits = layer.output
    probs = softmax(logits)
    dlogits = (probs - yoh) / yoh.shape[0]
    layer.backward(dlogits)
    grad_analytic = layer.dweights.copy()

    grad_fd = np.zeros_like(layer.weights)
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            w = layer.weights[i, j]
            layer.weights[i, j] = w + eps
            layer.forward(X); l1 = cross_entropy(softmax(layer.output), yoh).mean()
            layer.weights[i, j] = w - eps
            layer.forward(X); l2 = cross_entropy(softmax(layer.output), yoh).mean()
            layer.weights[i, j] = w
            grad_fd[i, j] = (l1 - l2) / (2 * eps)

    rel_err = np.linalg.norm(grad_analytic - grad_fd) / (np.linalg.norm(grad_analytic) + np.linalg.norm(grad_fd) + 1e-12)
    assert rel_err < 1e-4
