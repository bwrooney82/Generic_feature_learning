import os

import numpy as np
import torch
from scipy.stats import norm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Algorithm 1
def nuclear_norm_subgradient(B):
    """
    Calculate the subgradient of the nuclear norm for matrix B.
    Args:
        B: Input matrix (m x n)
    Returns:
        Subgradient matrix (m x n)
    """
    # Perform SVD
    U, S, V = torch.svd(B, some=False)
    U1, V1 = U[:, :len(S)], V[:, :len(S)]  # Non-zero singular value components

    # Randomized component for zero singular values
    m, n = B.shape
    s = len(S)
    U2, V2 = U[:, s:], V[:, s:]  # Zero singular value components
    D = torch.randn(m - s, n - s).cuda()  # Random matrix for zero component
    D = D / torch.norm(D, p="fro")  # Normalize D

    # Subgradient calculation
    subgradient = U1 @ V1.T + U2 @ D @ V2.T
    return subgradient


def normalize_matrix_to_unit_norm(A):
    norm_2 = torch.linalg.norm(A, ord=2)
    if norm_2 > 0:
        A = A / norm_2
    return A


def optimize_transformation_matrix(Y_nor, Y_cur, lambda_, beta=0.01, max_iter=100, tol=1e-3, gamma=1e-2, alpha=0.05):
    """
    Optimize the transformation matrix A using the given parameters.

    Args:
        Y_nor (torch.Tensor): Normal feature matrix, shape (n1, dim)
        Y_cur (torch.Tensor): Current (possibly abnormal) feature matrix, shape (1, dim)
        lambda_ (float): Regularization parameter for log(det(A^2))
        beta (float): Step size for SGM optimization
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance

    Returns:
        Optimal matrix A
    """
    n1, dim = Y_nor.shape
    A = torch.eye(dim).cuda()  # Initialize A as identity matrix

    Y = torch.cat((Y_nor, Y_cur), dim=0)

    mu = Y_nor.mean(dim=0)  # sample mean
    V = ((Y_nor - mu).T @ (Y_nor - mu))
    S = V / (n1 - 1)  # covariance matrix S

    epsilon = (1e-3) * torch.eye(dim).cuda()

    loss_history = []
    grad_norm_history = []
    for k in range(1, max_iter + 1):

        # 1. Calculate the loss components
        AY_nor = torch.norm(A @ Y_nor.T, p='nuc')  # ||A * Y_nor||_*
        AY_cur = torch.norm(A @ Y_cur.T, p='nuc')  # ||A * Y_cur||_*
        AY = torch.norm(A @ Y.T, p='nuc')  # ||A * Y||_*

        log_det_term = lambda_ * n1 * 2 * torch.logdet(A)  # ln(det(A)^2)

        inv_ASAT = torch.inverse(A @ S @ A.T + epsilon)
        trace_term = lambda_ * torch.trace(inv_ASAT @ (A @ V @ A.T))  # tr((ASA^T)-1 AVA^T)

        # 2. Compute the loss
        loss = AY_nor + AY_cur - AY + trace_term + log_det_term

        loss_history.append(loss.item())
        # print(f"Iteration {k}: Loss = {loss.item()}")

        # 3. Compute gradient ∇L(A)
        G = inv_ASAT
        Omega = -G @ (A @ V @ A.T) @ G
        # grad_Lg = -2 * n1 * (torch.inverse(A.T)) - Omega.T @ A @ S.T - Omega @ A @ S - G.T @ A @ V.T - G @ A @ V
        grad_Lg = lambda_ * (- Omega.T @ A @ S.T - Omega @ A @ S - G.T @ A @ V.T - G @ A @ V)

        # Subgradient of nuclear norm terms
        grad_AY_nor = nuclear_norm_subgradient(A @ Y_nor.T) @ Y_nor
        grad_AY_cur = nuclear_norm_subgradient(A @ Y_cur.T) @ Y_cur
        grad_AY = nuclear_norm_subgradient(A @ Y.T) @ Y

        grad_det = lambda_ * 2 * n1 * torch.inverse(A)
        # Combine gradient terms
        # grad_A = grad_AY_nor + grad_AY_cur - grad_AY - grad_Lg - grad_det
        # grad_A = grad_AY_nor + grad_AY_cur - grad_AY
        grad_A = - grad_Lg - grad_det

        grad_norm = torch.norm(grad_A)
        grad_norm_history.append(grad_norm.item())
        # print(f"Iteration {k}: Gradient Norm = {grad_norm.item()}")

        # Update rule with diminishing step size
        step_size = beta / (k ** 0.5)
        A_next = A - step_size * grad_A

        # Normalize A to satisfy ||A||_2 = 1
        A_next = normalize_matrix_to_unit_norm(A_next)

        # Check for convergence
        delta_A = torch.norm(A_next - A)
        if delta_A < tol:
            # print(f"Convergence achieved at iteration {k}.")
            break
        A = A_next
    
    Z_cur = A @ Y_cur.T
    Z_nor = A @ Y_nor.T
   
    mu_z = Z_nor.mean(dim=1, keepdim=True)
    S_z = ((Z_nor - mu_z) @ (Z_nor - mu_z).T) / (n1 - 1)
    S_regularized = S_z + gamma * torch.eye(dim).cuda()

    # Calculate the RHT statistic
    delta_z = Z_cur - mu_z
    R = n1 * delta_z.T @ torch.inverse(S_regularized) @ delta_z
    RHT_statistic = torch.trace(R) / dim

    # 计算 F1 和 F2
    F1 = (1 / dim) * torch.trace(torch.linalg.inv(S_regularized)).item()
    F2 = (1 / dim) * torch.trace(torch.linalg.inv(S_regularized) @ torch.linalg.inv(S_regularized)).item()

    # 计算 θ1 和 θ2
    theta1 = (1 - gamma * F1) / (1 - dim * (1 - gamma * F1) / n1)
    common_term = 1 - dim / n1 + dim * gamma * F1 / n1
    theta2 = ((1 - gamma * F1) / (common_term ** 3)) - (gamma * (F1 - gamma * F2) / (common_term ** 4))

    z_alpha = norm.ppf(1 - alpha)

    control_limit = z_alpha * np.sqrt(np.maximum(0, 2 * dim * theta2)) + dim * theta1

    return A, Z_cur, RHT_statistic.item(), control_limit,k

def RHT_statistic(Z_nor,Z_cur, gamma=1e-2, alpha=0.05):
    
    dim, n1 = Z_nor.shape

    mu_z = Z_nor.mean(dim=1, keepdim=True)
    S_z = ((Z_nor - mu_z) @ (Z_nor - mu_z).T) / (n1 - 1)
    S_regularized = S_z + gamma * torch.eye(dim).cuda()
    
    delta_z = Z_cur - mu_z
    R = n1 * delta_z.T @ torch.inverse(S_regularized) @ delta_z
    RHT_statistic = torch.trace(R) / dim
     # 计算 F1 和 F2
    F1 = (1 / dim) * torch.trace(torch.linalg.inv(S_regularized)).item()
    F2 = (1 / dim) * torch.trace(torch.linalg.inv(S_regularized) @ torch.linalg.inv(S_regularized)).item()

    # 计算 θ1 和 θ2
    theta1 = (1 - gamma * F1) / (1 - dim * (1 - gamma * F1) / n1)
    common_term = 1 - dim / n1 + dim * gamma * F1 / n1
    theta2 = ((1 - gamma * F1) / (common_term ** 3)) - (gamma * (F1 - gamma * F2) / (common_term ** 4))

    z_alpha = norm.ppf(1 - alpha)

    control_limit = z_alpha * np.sqrt(np.maximum(0, 2 * dim * theta2)) + dim * theta1

    return RHT_statistic.item(), control_limit

