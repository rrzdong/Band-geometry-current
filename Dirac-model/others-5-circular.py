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
        self.factor = 2.32*(-2) #*10**(-11)
        
        
        self.results = {}

    @staticmethod
    @njit(parallel=True)
    def _compute_core(omega, kx, ky, dk, s,  c, d, t1, t2, delta_param, J, a, mu):
        
        num_omega = len(omega)
        N_grid = len(kx)
        sigma = np.zeros(num_omega)
        dx_diff = dk / 100.0
        
        
        sig_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sig_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        eye = np.eye(2, dtype=np.complex128)
        prefactor = (dk**2) / (4 * np.pi**2)

        for i in prange(N_grid):
            for j in range(N_grid):
                k_x, k_y = kx[i], ky[j]

                
                def get_h(kx_v, ky_v):
                    h0 = -2 * t2 * (np.cos(kx_v*a + ky_v*a) + np.cos(kx_v*a - ky_v*a))
                    hx = -2 * t1 * (np.cos(kx_v*a) + np.cos(ky_v*a))
                    hz = 2 * t2 * delta_param * (np.cos(kx_v*a + ky_v*a) - np.cos(kx_v*a - ky_v*a)) - J * s
                    return h0 * eye + hx * sig_x + hz * sig_z

                
                H = get_h(k_x, k_y)
                eigval, eigvec = np.linalg.eigh(H)
                
                
                vx_mat = (get_h(k_x + dx_diff, k_y) - get_h(k_x - dx_diff, k_y)) / (2 * dx_diff)
                vy_mat = (get_h(k_x, k_y + dx_diff) - get_h(k_x, k_y - dx_diff)) / (2 * dx_diff)
                
                
                va = vx_mat 
                vb = vy_mat 
                
                vc = vx_mat * np.cos(c) + vy_mat * np.sin(c)
                vd = vx_mat * np.cos(d) + vy_mat * np.sin(d)
                
                
                va_10 = eigvec[:, 1].conj() @ (va @ eigvec[:, 0])
                va_00 = eigvec[:, 0].conj() @ (va @ eigvec[:, 0])
                va_11 = eigvec[:, 1].conj() @ (va @ eigvec[:, 1])
                va_diff = va_11 - va_00
                
                vb_01 = eigvec[:, 0].conj() @ (vb @ eigvec[:, 1])
                vb_00 = eigvec[:, 0].conj() @ (vb @ eigvec[:, 0])
                vb_11 = eigvec[:, 1].conj() @ (vb @ eigvec[:, 1])
                vb_diff = vb_11 - vb_00
                
                vc_01 = eigvec[:, 0].conj() @ (vc @ eigvec[:, 1])
                vc_10 = np.conj(vc_01)
                
                vd_00 =eigvec[:, 0].conj() @ (vd @ eigvec[:, 0])
                vd_11 =eigvec[:, 1].conj() @ (vd @ eigvec[:, 1])
                vd_diff = vd_11 - vd_00
                
        
                e_diff = eigval[1] - eigval[0]
                flag = 1
                if e_diff <0.001:
                    flag = 0.000000000001

                f0 = 1.0 / (1.0 + np.exp((eigval[0] - mu) / 0.0001))
                f1 = 1.0 / (1.0 + np.exp((eigval[1] - mu) / 0.0001))
                f_diff = f0 - f1

                
                for k in range(num_omega):
                   
                    d=0.1
                    delta_fun = (1.0 / (d * np.sqrt(np.pi))) * np.exp(-((e_diff - omega[k])**2) / d**2)
                    sigma[k] += f_diff*(delta_fun*vd_diff/e_diff*(vb_01*vc_10*va_diff/e_diff - va_10*vc_01*vb_diff/e_diff)/e_diff**2*flag).real*prefactor
        
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
        filename = f'TIC-5-{mode.lower()}-t2={self.t2}.txt'
        header = f'omega(eV)  {mode}_result_real'
        np.savetxt(filename, output_array, header=header, fmt='%.8e')
        print(f"Data successfully saved to: {filename}")


if __name__ == "__main__":

    
    sim = HallConductivitySim(
          N=5000,             
          mu=0.909427, 
          U=5,
          deltam=0.374206,
          t2=0.5
        )
    sim.run(c=0, d=np.pi/2)
    sim.plot_results(mode="charge")
    sim.save_data(mode="charge")
    
    
