import numpy as np
from scipy.linalg import eigh

def SCF(
    t=0.5,
    delta=0.5,
    U=5,
    h=0,         # 磁场强度
    mgopen=1,    # 磁场开关 1=开, 0=关
    T=1e-4,      # 温度
    k_B=1.0,     # Boltzmann常数
    N_k=100,     # k点数
    tol=1e-6,    # 收敛阈值
    mixing_ratio=0.4,
    max_iter=3000,
    max_outer_iter=50,
    n_particles=1.0,
    verbose=True
):
    tm = (1 - delta) * t
    tp = (1 + delta) * t

    # 构建k点网格
    kx = np.linspace(-np.pi, np.pi, N_k)
    ky = np.linspace(-np.pi, np.pi, N_k)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    momentum_points = np.c_[kx_grid.ravel(), ky_grid.ravel()] 

    sigma_z_block = np.diag([1, -1, -1, 1])

    def construct_hamiltonian(kx, ky, delta_m):
        h_AA = -2 * (tm * np.cos(kx + ky) + tp * np.cos(kx - ky)) 
        h_BB = -2 * (tp * np.cos(kx + ky) + tm * np.cos(kx - ky)) 
        h_AB = -2 * (np.cos(kx) + np.cos(ky))

        # Up block
        H_up = np.array([
            [h_AA - U * delta_m - h , h_AB],
            [h_AB, h_BB + U * delta_m - h ]
        ])
        # Down block
        H_down = np.array([
            [h_AA + U * delta_m + h , h_AB],
            [h_AB, h_BB - U * delta_m + h ]
        ])
        # Block diagonal Hamiltonian
        H_total = np.block([
            [H_up, np.zeros_like(H_up)],
            [np.zeros_like(H_down), H_down]
        ])
        return H_total

    def fermi_distribution(energy, mu):
        arg = (energy - mu) / (k_B * T)
        return 1.0 / (np.exp(arg) + 1)

    def iteration(mu_init):
        delta_m = 0.1  # 初始猜测
        mu = mu_init

        for iter_inner in range(max_iter):
            delta_m_out = 0.0
            n_total = 0.0

            for kx_val, ky_val in momentum_points:
                H_k = construct_hamiltonian(kx_val, ky_val, delta_m)
                eigenvalues, eigenvectors = eigh(H_k)

                for idx, eps_k in enumerate(eigenvalues):
                    f_k = fermi_distribution(eps_k, mu)
                    n_total += f_k
                    psi = eigenvectors[:, idx]
                    delta_m_out += f_k * np.real(np.vdot(psi, sigma_z_block @ psi))

            delta_m_out /= (4 * N_k**2)  
            n_total /= (2 * N_k**2)

            # 更新化学势mu，使粒子数匹配
            mu += (n_particles - n_total) * 0.1

            # 混合更新delta_m
            delta_diff = abs(delta_m_out - delta_m)
            delta_m_new = mixing_ratio * delta_m_out + (1 - mixing_ratio) * delta_m
            delta_m = delta_m_new

            if delta_diff < tol:
                return mu, n_total, delta_m

        if verbose:
            print("Warning: Inner iteration did not converge.")
        return mu, n_total, delta_m

    mu = 1.0  # 初值
    for outer_it in range(max_outer_iter):
        mu_val, n_val, delta_m_val = iteration(mu)

        diff_n = abs(n_val - n_particles)
        
        if verbose:
            print(f"iter {outer_it}: delta_m = {delta_m_val:.6f}, mu = {mu_val:.6f}, n = {n_val:.6f}")

        if diff_n < 1e-2:
            if verbose:
                print("SCF calculation converged.")
                print(f"Final order parameter δm = {delta_m_val:.6f}")
                print(f"Final chemical potential µ = {mu_val:.6f}")
                print(f"Final particle density n = {n_val:.6f}")
            break
        else:
           if diff_n < 0.0:
             mu = mu*0.7 - mu_val*0.3 - diff_n
           else:
             mu = mu*0.3 + mu_val*0.7 - diff_n
                 
    else:
        if verbose:
            print("Warning: SCF calculation did not converge in outer loop.")

    return mu_val, n_val, delta_m_val


# if __name__ == "__main__":
#         mu_, n_, delta_m_ = SCF(verbose=True, U=5, h=0.0)
if __name__ == "__main__":
    h_values = np.linspace(0.0, 0.4, 20)
    
    # 准备结果列表，存储每组计算结果
    results = []

    for h_ in h_values: 
        mu_, n_, delta_m_ = SCF(verbose=True, U=5, h=h_)
        results.append([h_, mu_, delta_m_, n_])
    
    # 将结果保存到txt文件，列名注释
    header = "h\tmu\tdelta_m\tn"
    results = np.array(results)
    np.savetxt("scf_results4.txt", results, header=header, comments="", fmt="%.8f", delimiter="\t")
    
    print("数据已保存到 scf_results.txt")
    
    

