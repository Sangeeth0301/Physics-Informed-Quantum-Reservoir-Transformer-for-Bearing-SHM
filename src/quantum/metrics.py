import numpy as np

def frobenius_divergence(K1, K2):
    """
    Computes the Frobenius distance between two kernel matrices.
    Provides fundamental evaluation of separation between the two regimes.
    """
    # Normalize matrices by Frobenius norm to handle magnitude shifts
    norm1 = np.linalg.norm(K1, 'fro')
    norm2 = np.linalg.norm(K2, 'fro')
    
    K1_norm = K1 / norm1 if norm1 > 1e-12 else K1
    K2_norm = K2 / norm2 if norm2 > 1e-12 else K2
    
    return np.linalg.norm(K1_norm - K2_norm, 'fro')

def compute_mmd(K_XX, K_YY, K_XY):
    """
    Unbiased Maximum Mean Discrepancy (MMD) estimator between two populations.
    Assumes K_XX and K_YY are symmetric matrices mapping intra-class similarity,
    and K_XY mapping cross-class similarity.
    """
    n = K_XX.shape[0]
    m = K_YY.shape[0]
    
    if n > 1:
        sum_XX = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
    else:
        sum_XX = 0
        
    if m > 1:
        sum_YY = (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
    else:
        sum_YY = 0
        
    sum_XY = np.sum(K_XY) / (n * m)
    
    mmd2 = sum_XX + sum_YY - 2 * sum_XY
    return np.sqrt(max(0, mmd2))

def kernel_eigenvalue_spectrum(K):
    """
    Compute real, sorted, eigenvalues of the kernel matrix.
    Safe bounded handling for PSD matrices.
    """
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(np.real(eigvals))[::-1]
    eigvals[eigvals < 0] = 0
    return eigvals
