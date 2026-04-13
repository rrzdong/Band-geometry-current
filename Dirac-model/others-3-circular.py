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
    def _compute_core(omega, kx, ky, dk, s, c_angle, d_angle, t1, t2, delta_param, J, a, mu):
        num_omega = len(omega)
        N_grid = len(kx)
        sigma = np.zeros(num_omega)
        dr = dk / 100.0  
        
        sig_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sig_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        eye = np.eye(2, dtype=np.complex128)
        prefactor = (dk**2) / (4 * np.pi**2)

        
        cos_c, sin_c = np.cos(c_angle), np.sin(c_angle)
        cos_d, sin_d = np.cos(d_angle), np.sin(d_angle)

        for i in prange(N_grid):
            for j in range(N_grid):
                k_x, k_y = kx[i], ky[j]

                
                def get_h(kx_v, ky_v):
                    h0 = -2 * t2 * (np.cos(kx_v*a + ky_v*a) + np.cos(kx_v*a - ky_v*a))
                    hx = -2 * t1 * (np.cos(kx_v*a) + np.cos(ky_v*a))
                    hz = 2 * t2 * delta_param * (np.cos(kx_v*a + ky_v*a) - np.cos(kx_v*a - ky_v*a)) - J * s
                    return h0 * eye + hx * sig_x + hz * sig_z

                # center point
                H_c = get_h(k_x, k_y)
                eigval, eigvec = np.linalg.eigh(H_c)
                e_diff = eigval[1] - eigval[0]
                if e_diff < 1e-6: continue

                f0 = 1.0 / (1.0 + np.exp((eigval[0] - mu) / 0.0001))
                f1 = 1.0 / (1.0 + np.exp((eigval[1] - mu) / 0.0001))
                f_diff = f0 - f1

                
                def get_r_elements(knx, kny):
                    H_n = get_h(knx, kny)
                    ev_n, ec_n = np.linalg.eigh(H_n)
                    ed_n = ev_n[1] - ev_n[0]
                    
                    
                    vx_n = (get_h(knx + dr, kny) - get_h(knx - dr, kny)) / (2 * dr)
                    vy_n = (get_h(knx, kny + dr) - get_h(knx, kny - dr)) / (2 * dr)
                    
                    # r_mn^x (01)
                    v01_x = ec_n[:, 0].conj() @ (vx_n @ ec_n[:, 1])
                    r01_x = v01_x / (-1j * ed_n)
                    
                    # r_nm^y (10)
                    v01_y = ec_n[:, 1].conj() @ (vy_n @ ec_n[:, 0])
                    r01_y = v01_y / (-1j * ed_n)
                    
                    return r01_x, r01_y

                r01x_x_p, r01y_x_p = get_r_elements(k_x + dr, k_y )
                r01x_x_m, r01y_x_m = get_r_elements(k_x - dr, k_y )

                
                r01x_y_p, r01y_y_p = get_r_elements(k_x , k_y + dr)
                r01x_y_m, r01y_y_m = get_r_elements(k_x , k_y - dr)

                
                dr01x_dkc = (r01x_x_p - r01x_x_m) / (2 * dr)*cos_c + (r01x_y_p - r01x_y_m) / (2 * dr)*sin_c
                dr01x_dkd = (r01x_x_p - r01x_x_m) / (2 * dr)*cos_d + (r01x_y_p - r01x_y_m) / (2 * dr)*sin_d
                
                dr01y_dkc = (r01y_x_p - r01y_x_m) / (2 * dr)*cos_c + (r01y_y_p - r01y_y_m) / (2 * dr)*sin_c
                dr01y_dkd = (r01y_x_p - r01y_x_m) / (2 * dr)*cos_d + (r01y_y_p - r01y_y_m) / (2 * dr)*sin_d
                
                dr10y_dkc=np.conj(dr01y_dkc)
                dr10y_dkd=np.conj(dr01y_dkd)
                
               
                integrand = (dr01x_dkc * dr10y_dkd - dr01x_dkd * dr10y_dkc)
                for k in range(num_omega):
                    d=0.1
                    gauss = (1.0 / (d * np.sqrt(np.pi))) * np.exp(-((e_diff - omega[k])**2) / d**2)
                    sigma[k] += -f_diff * integrand.real * gauss * prefactor
                    
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
        plt.ylabel(r'Signal ($10^{-6} Am/V^3$)')
        plt.title(f'{mode.capitalize()} Current (t1={self.t1}, t2={self.t2}, mu={self.mu})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
    def save_data(self, mode="spin"):
        """保存数据：可选 'spin' 或 'charge'"""
        data = self._get_combined_data(mode)
        if data is None:
            print("No data available to save.")
            return

        output_array = np.column_stack((self.omega, data.real))
        filename = f'TIC-3-{mode.lower()}-t2={self.t2}.txt'
        header = f'omega(eV)  {mode}_result_real'
        np.savetxt(filename, output_array, header=header, fmt='%.8e')
        print(f"Data successfully saved to: {filename}")

# --- 主程序入口 ---
if __name__ == "__main__":

    
    sim = HallConductivitySim(
            N=5000,             # 建议先用 200 测试速度，正式计算用 400
            mu=0.909427, 
            U=5,
            deltam=0.374206,
            t2=0.5
        )
    sim.run(c=0, d=np.pi/2)
    sim.plot_results(mode="charge")
    sim.save_data(mode="charge")
    
