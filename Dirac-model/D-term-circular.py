# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 23:25:35 2026

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

class HallConductivitySim:
    def __init__(self, t1=1.0, t2=0.5, delta=0.5, U=5.0, deltam=0.37, N=400, a=2, mu=0.1):
        # 物理参数初始化
        self.t1 = t1
        self.t2 = t2
        self.delta = delta
        self.J = U*deltam
        self.N = N
        self.a = a
        self.mu = mu
        self.U = U
        
        
        self.omega = np.linspace(0.1, 6, 500)
        self.kx = np.linspace(-np.pi/a, np.pi/a, N)
        self.ky = np.linspace(-np.pi/a, np.pi/a, N)
        self.dk = self.kx[1] - self.kx[0]
        self.factor = 2.32 #*10**(-11)
        
        
        self.results = {}

    @staticmethod
    @njit(parallel=True)
    def _compute_core(omega, kx, ky, dk, s, c, d, t1, t2, delta_param, J, a, mu):
        num_omega = len(omega)
        N_grid = len(kx)
        sigma = np.zeros(num_omega)
        dr = dk / 100.0  
        
        sig_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sig_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        eye = np.eye(2, dtype=np.complex128)
        prefactor = (dk**2) / (4 * np.pi**2)

        
        cos_c, sin_c = np.cos(c), np.sin(c)
        cos_d, sin_d = np.cos(d), np.sin(d)

        for i in prange(N_grid):
            for j in range(N_grid):
                k_x, k_y = kx[i], ky[j]

                # Hamiltonian
                def get_h(kx_v, ky_v):
                    h0 = -2 * t2 * (np.cos(kx_v*a + ky_v*a) + np.cos(kx_v*a - ky_v*a))
                    hx = -2 * t1 * (np.cos(kx_v*a) + np.cos(ky_v*a))
                    hz = 2 * t2 * delta_param * (np.cos(kx_v*a + ky_v*a) - np.cos(kx_v*a - ky_v*a)) - J * s
                    return h0 * eye + hx * sig_x + hz * sig_z

                # center point
                H_center = get_h(k_x, k_y)
                eigval, eigvec = np.linalg.eigh(H_center)
                e_diff = eigval[1] - eigval[0]
                if e_diff < 1e-6: continue

                # velocity matrix
                vx_mat = (get_h(k_x + dr, k_y) - get_h(k_x - dr, k_y)) / (2 * dr)
                vy_mat = (get_h(k_x, k_y + dr) - get_h(k_x, k_y - dr)) / (2 * dr)
                
                # ra
                v01_a = eigvec[:, 0].conj() @ (vx_mat @ eigvec[:, 1])
                r01_a = v01_a / (-1j * e_diff)
                
                # rb
                v10_b = eigvec[:, 1].conj() @ (vy_mat @ eigvec[:, 0])
                r10_b = v10_b / (1j * e_diff)

                # dr/dkc dkd 
                offsets = [
                    (-dr*cos_c - dr*cos_d, -dr*sin_c - dr*sin_d), # (-c, -d)
                    (-dr*cos_c + dr*cos_d, -dr*sin_c + dr*sin_d), # (-c, +d)
                    ( dr*cos_c - dr*cos_d,  dr*sin_c - dr*sin_d), # (+c, -d)
                    ( dr*cos_c + dr*cos_d,  dr*sin_c + dr*sin_d)  # (+c, +d)
                ]
                
                r01_a_neighbors = np.zeros(4, dtype=np.complex128)
                r10_b_neighbors = np.zeros(4, dtype=np.complex128)

                for idx in range(4):
                    knx = k_x + offsets[idx][0]
                    kny = k_y + offsets[idx][1]
                    
                    H_n = get_h(knx, kny)
                    eval_n, evec_n = np.linalg.eigh(H_n)
                    ediff_n = eval_n[1] - eval_n[0]
                    
                    
                    vx_n = (get_h(knx + dr, kny) - get_h(knx - dr, kny)) / (2 * dr)
                    vy_n = (get_h(knx, kny + dr) - get_h(knx, kny - dr)) / (2 * dr)
                    
                    v01_a_n = evec_n[:, 0].conj() @ (vx_n @ evec_n[:, 1])
                    v10_b_n = evec_n[:, 1].conj() @ (vy_n @ evec_n[:, 0])
                    
                    r01_a_neighbors[idx] = v01_a_n / (-1j * ediff_n)
                    r10_b_neighbors[idx] = v10_b_n / (1j * ediff_n)

                d2_r01_a = (r01_a_neighbors[3] - r01_a_neighbors[2] - r01_a_neighbors[1] + r01_a_neighbors[0]) / (4.0 * dr**2)
                d2_r10_b = (r10_b_neighbors[3] - r10_b_neighbors[2] - r10_b_neighbors[1] + r10_b_neighbors[0]) / (4.0 * dr**2)

                # conductivity
                integrand = r01_a * d2_r10_b  - d2_r01_a*r10_b  
                
                f0 = 1.0 / (1.0 + np.exp((eigval[0] - mu) / 0.0001))
                f1 = 1.0 / (1.0 + np.exp((eigval[1] - mu) / 0.0001))
                f_diff = f0 - f1

                for k in range(num_omega):
                    d=0.01
                    delta_fun = (1.0 / (d * np.sqrt(np.pi))) * np.exp(-((e_diff - omega[k])**2) / d**2)
                    sigma[k] += f_diff * (delta_fun * integrand).real * prefactor
                    
        return sigma

    def run(self, s_values=[1, -1], c=0, d=0):
        
        for s in s_values:
            print(f"Calculating: s={s}, t1={self.t1:.4f}, mu={self.mu:.4f}...")
            self.results[s] = self._compute_core(
                self.omega, self.kx, self.ky, self.dk, s, c, d,
                self.t1, self.t2, self.delta, self.J, self.a, self.mu
            )

    def _get_combined_data(self, mode):
        
        if 1 not in self.results or -1 not in self.results:
            return None
        
        if mode.lower() == "spin":
            return (self.results[1] - self.results[-1]) * self.factor
        elif mode.lower() == "charge":
            return (self.results[1] + self.results[-1]) * self.factor
        return None

    def plot_results(self, mode="spin"):
        
        data = self._get_combined_data(mode)
        if data is None:
            print("No data available. Run simulation first.")
            return

        plt.figure(figsize=(8, 6), dpi=100)
        color = 'g' if mode == "spin" else 'b'
        plt.plot(self.omega, data.real, color=color, lw=2, label=f'{mode.capitalize()}')
        plt.axhline(0, color='black', lw=0.5, linestyle='--')
        plt.xlabel(r'Photon energy $\hbar\omega$ (eV)')
        plt.ylabel(r'Signal ($10^{-11} Am/V^3$)')
        plt.title(f'{mode.capitalize()} Current (t1={self.t1}, t2={self.t2}, mu={self.mu})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
    def save_data(self, mode="spin"):
        
        data = self._get_combined_data(mode)
        if data is None:
            print("No data available to save.")
            return

        output_array = np.column_stack((self.omega, data.real))
        filename = f'TIC-2-{mode.lower()}-t2={self.t2}.txt'
        header = f'omega(eV)  {mode}_result_real'
        np.savetxt(filename, output_array, header=header, fmt='%.8e')
        print(f"Data successfully saved to: {filename}")

if __name__ == "__main__":

    sim = HallConductivitySim(
            N=20000,             
            mu=0.909427, 
            U=5,
            deltam=0.374206,
            t2=0.5
        )
    sim.run(c=0, d=0)
    sim.plot_results(mode="spin")
    sim.save_data(mode="spin")
    
    # raw_data = np.loadtxt('delta_m_vs_U2.txt', skiprows=1)
            
    # Us = raw_data[:, 0]
    # mus = raw_data[:, 1]
    # deltas = raw_data[:, 2]


    # for i in range(len(Us)):

    #     sim = HallConductivitySim(
    #         N=4000,             # 建议先用 200 测试速度，正式计算用 400
    #         mu=mus[i], 
    #         U=Us[i],
    #         deltam=deltas[i]
    #     )
        
    #     # 3. 运行核心计算 (以 xx 分量 c=0, d=0 为例)
    #     sim.run(c=0, d=0)
        
    #     # 4. 绘图与保存 (Spin 模式)
    #     sim.plot_results(mode="spin")
    #     sim.save_data(mode="spin")
        
    #     # # 5. 绘图与保存 (Charge 模式)
    #     # sim.plot_results(mode="charge")
    #     # sim.save_data(mode="charge")