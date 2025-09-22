import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# 常量定义
T = 0.00001  
k_B = 1.0  
N_k =100  
tol = 1e-6  
mixing_ratio = 0.4  

# k-mesh
kx = np.linspace(-np.pi, np.pi, N_k)
ky = np.linspace(-np.pi, np.pi, N_k)
kx_grid, ky_grid = np.meshgrid(kx, ky)
momentum_points = np.c_[kx_grid.ravel(), ky_grid.ravel()]


delta_m = 0.1  
mu = 1.5   
n_particles = 1.0  

t = 1
delta = 0.5
tm = 0.5*(1 - delta) * t
tp = 0.5*(1 + delta) * t
U = 5

def construct_hamiltonian(kx, ky, delta_m):
    h_AA = -2 * (tm * np.cos(kx + ky) + tp * np.cos(kx - ky)) 
    h_BB = -2 * (tp * np.cos(kx + ky) + tm * np.cos(kx - ky)) 
    h_AB = -2 * (np.cos(kx) + np.cos(ky))
    H_up = np.array([[h_AA - U * delta_m, h_AB],
                     [h_AB, h_BB + U * delta_m]])
    H_down = np.array([[h_AA + U * delta_m, h_AB],
                       [h_AB, h_BB - U * delta_m]])
    H_total = np.block([[H_up, np.zeros_like(H_up)], [np.zeros_like(H_down), H_down]])  
    return H_total

def fermi_distribution(energy, mu, T):
    return 1.0 / (np.exp((energy - mu) / (k_B * T)) + 1)


for iteration in range(1000):
    delta_m_out = 0.0
    n_total = 0.0

    for kx, ky in momentum_points:
        H_k = construct_hamiltonian(kx, ky, delta_m)
        eigenvalues, eigenvectors = eigh(H_k)

        for idx, eps_k in enumerate(eigenvalues):
            f_k = fermi_distribution(eps_k, mu, T)
            n_total += f_k
            delta_m_out += f_k * eigenvectors[:, idx].conj().dot(
                np.diag([1, -1, -1, 1]).dot(eigenvectors[:, idx])
            )

    delta_m_out /= (4 * N_k**2)
    n_total /= (2 * N_k**2)

    # update mu
    mu += (n_particles - n_total) * 0.01

    if abs(delta_m_out - delta_m) < tol:
        print(f"自洽计算收敛: δm = {delta_m_out}, µ = {mu}, 迭代次数 = {iteration}")
        break

    # update δm
    delta_m = mixing_ratio * delta_m_out + (1 - mixing_ratio) * delta_m

else:
    print("迭代未能收敛，请检查参数设置或改进初始猜测。")

print(f"最终秩序参数 δm = {delta_m}")
print(f"最终化学势 µ = {mu}")
