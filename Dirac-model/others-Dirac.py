#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:43:26 2026

@author: dongruizhi
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(parallel=True)
def compute_injection_term_band_grad(omega, kx, ky, dk, hbar_vF, m):
    num_omega = len(omega)
    N_grid = len(kx)
    sigma_res = np.zeros(num_omega)
    
    prefactor = (dk**2) / (4.0 * np.pi**2)
    hvF2 = hbar_vF**2
    m2 = m**2

    for i in prange(N_grid):
        for j in range(N_grid):
            k_x, k_y = kx[i], ky[j]
            k2 = k_x**2 + k_y**2
            if k2 < 1e-12: continue
            
            
            Ek = np.sqrt(hvF2 * k2 + m2)
            w_mn = 2.0 * Ek
            
            
            dw_dy = 2.0 * hvF2 * k_y / Ek
            dw_dx = 2.0 * hvF2 * k_x / Ek 
            
            
            r_pre = hbar_vF / (2.0 * Ek)
            k_mod = np.sqrt(k2)
            r_px = r_pre * (m * k_x / (Ek * k_mod) + 1j * k_y / k_mod)
            r_py = r_pre * (m * k_y / (Ek * k_mod) - 1j * k_x / k_mod)
            r_nx = np.conj(r_px) 

            
            factor_out = dw_dy / w_mn
            term1 = (r_nx * r_px) * (dw_dy / w_mn)
            term2 = (r_py * r_nx) * (dw_dx / w_mn)
            
            total_val = (factor_out * (term1 - term2)).real
            
            
            gauss_width = 0.1
            for k in range(num_omega):
                diff = w_mn - omega[k]
                if abs(diff) > 5 * gauss_width: continue
                
                delta_fun = (1.0 / (gauss_width * np.sqrt(np.pi))) * \
                            np.exp(-(diff**2) / (gauss_width**2))
                sigma_res[k] += (total_val * delta_fun * prefactor)
                
    return sigma_res

class DiracInjectionBandGradSim:
    def __init__(self, hbar_vF=52660.0, m=8.5, N=2000, k_max=0.05):
        self.hbar_vF = hbar_vF
        self.m = m
        self.N = N
        self.k_max = k_max
        self.omega = np.linspace(0.1, 50, 400)
        self.kx = np.linspace(-k_max, k_max, N)
        self.ky = np.linspace(-k_max, k_max, N)
        self.dk = self.kx[1] - self.kx[0]
        self.factor = 7.6471/6*2 #(10**-20)
        self.results = None

    def run(self):
        
        self.results = compute_injection_term_band_grad(
            self.omega, self.kx, self.ky, self.dk, self.hbar_vF, self.m
        )
        

    def save_to_txt(self):
        if self.results is None: return
        
        data_to_save = np.column_stack((self.omega, -self.results * self.factor))
        
        np.savetxt("others-term.txt", data_to_save, fmt='%.10e')
       

    def plot(self):
        if self.results is None: return
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(self.omega, -self.results * self.factor, color='firebrick', lw=2)
        plt.axvline(2*self.m, color='blue', ls='--', label=f'Gap={2*self.m} meV')
        plt.xlabel(r'Photon Energy $\omega$ (meV)')
        plt.ylabel(r'Response $F(\omega)(10^{-20})$')
        plt.title('Injection Current: others')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    sim = DiracInjectionBandGradSim(N=60000, k_max=0.05)
    sim.run()
    sim.save_to_txt()
    sim.plot()