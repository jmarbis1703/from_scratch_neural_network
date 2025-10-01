import numpy as np
from final_project.models.nn_scratch import softmax, cross_entropy, to_one_hot

def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(10, 5)).astype(np.float32)
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)

def test_cross_entropy_non_negative():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(16, 3)).astype(np.float32)
    y = rng.integers(0, 3, size=16)
    p = softmax(z)
    yoh = to_one_hot(y, 3)
    loss = cross_entropy(p, yoh)
    assert np.all(loss >= -1e-6)
