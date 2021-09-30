import numpy as np

def l2norm(X, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = np.array([sum([i**2 for i in X]) + eps for ii in X])
    X = np.divide(X, norm)
    return X

X = np.random.random_sample((512))

a = l2norm(X)
print(a)

b = X/np.linalg.norm(X)
print(b)