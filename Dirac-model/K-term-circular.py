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
        
        
        self.omega = np.linspace(0.0001, 10, 500)
        self.kx = np.linspace(-np.pi/a, np.pi/a, N)
        self.ky = np.linspace(-np.pi/a, np.pi/a, N)
        self.dk = self.kx[1] - self.kx[0]
        self.factor = 2.32
        
        
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
                    hz = 2 * t2 * delta_param * (np.cos(kx_v*a + ky_v*a) - np.cos(kx_v*a - ky_v*a)) - J * s + 4*np.sin(kx_v*a)*np.sin(ky_v*a)
                    return h0 * eye + hx * sig_x + hz * sig_z

                # center point
                H_center = get_h(k_x, k_y)
                eigval, eigvec = np.linalg.eigh(H_center)
                e_diff = eigval[1] - eigval[0]
                if e_diff < 1e-6: continue

                # velocity matrix
                vx_mat = (get_h(k_x + dr, k_y) - get_h(k_x - dr, k_y)) / (2 * dr)
                vy_mat = (get_h(k_x, k_y + dr) - get_h(k_x, k_y - dr)) / (2 * dr)
                
                va_01 = eigvec[:, 0].conj() @ (vx_mat @ eigvec[:, 1]) 
                vb_10 =  eigvec[:, 1].conj() @ (vy_mat @ eigvec[:, 0]) 
                
                
                vc =  eigvec[:, 1].conj() @ (vx_mat @ eigvec[:, 0])
                vd =  eigvec[:, 0].conj() @ (vy_mat @ eigvec[:, 1])
                
                Omega_diff = 4*(vc*vd/e_diff**2).imag
                
                integrand = va_01*vb_10*Omega_diff
                
                f0 = 1.0 / (1.0 + np.exp((eigval[0] - mu) / 0.0001))
                f1 = 1.0 / (1.0 + np.exp((eigval[1] - mu) / 0.0001))
                f_diff = f0 - f1

                for k in range(num_omega):
                    delta_fun = (1.0 / (0.1 * np.sqrt(np.pi))) * np.exp(-((e_diff - omega[k])**2) / 0.01)
                    sigma[k] += f_diff * (delta_fun * integrand).imag* prefactor
                    
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
            N=2000,            
            mu=0.90695178, 
            U=5,
            deltam=0.374,
            t2=0.50
        )
    sim.run(c=0, d=np.pi/2)
    sim.plot_results(mode="spin")
    # sim.save_data(mode="charge")