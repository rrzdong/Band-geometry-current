#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:37:59 2026

@author: dongruizhi
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

class DiracTheoryResponseSim:
    def __init__(self, hbar_vF=5.266*10000, m=8.5, N=1000, k_max=0.05):
        """
        单位：meV, Angstrom
        hbar_vF: 52660 meV*A
        m: 8.5 meV (Gap = 17 meV)
        N: 网格点数
        k_max: 0.05 A^-1 
        """
        self.hbar_vF = hbar_vF
        self.m = m
        self.N = N
        self.k_max = k_max
        
        
        self.omega = np.linspace(0.1, 50, 400)
        
        
        self.kx = np.linspace(-k_max, k_max, N)
        self.ky = np.linspace(-k_max, k_max, N)
        self.dk = self.kx[1] - self.kx[0]
        self.results = None
        
        self.factor = 7.6471/6 #(10**-20)

    @staticmethod
    @njit(parallel=True)
    def _compute_core_theory(omega, kx, ky, dk, hbar_vF, m):
        num_omega = len(omega)
        N_grid = len(kx)
        sigma_res = np.zeros(num_omega)
        
        prefactor = (dk**2) / (4.0 * np.pi**2)
        m2 = m**2
        hvF2 = hbar_vF**2
        hvF4 = hbar_vF**4

        for i in prange(N_grid):
            for j in range(N_grid):
                k2 = kx[i]**2 + ky[j]**2
                
                #  E_diff = 2 * sqrt((hvF*k)^2 + m^2)
                e_diff = 2.0 * np.sqrt(hvF2 * k2 + m2)
                
                 
                term = 16.0 * m2 * hvF4 / (e_diff**6)

                
                gauss_width = 0.1 
                for k in range(num_omega):
                    diff = e_diff - omega[k]
                    
                    if abs(diff) > 5 * gauss_width:
                        continue
                        
                    delta_fun = (1.0 / (gauss_width * np.sqrt(np.pi))) * \
                                np.exp(-(diff**2) / (gauss_width**2))
                    sigma_res[k] += (term * delta_fun * prefactor)
                    
        return sigma_res

    def run(self):
        
        self.results = self._compute_core_theory(
            self.omega, self.kx, self.ky, self.dk, 
            self.hbar_vF, self.m
        )
        

    def save_to_txt(self):
        
        if self.results is None:
            return
        
        data_to_save = np.column_stack((self.omega, self.results * self.factor))
        
        np.savetxt("K-term.txt", data_to_save, fmt='%.10e')
       

    def plot_results(self):
        if self.results is None: return
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(self.omega, self.results*self.factor, color='royalblue', lw=2, label='Theory Formula')
        plt.axvline(2*abs(self.m), color='orange', linestyle='--', label=f'Gap={2*abs(self.m)} meV')
        plt.xlabel(r'Photon energy $\hbar\omega$ (meV)')
        plt.ylabel(r'Response $F(\omega)(10^{-20})$')
        plt.title(f'Massive Dirac Optical Response (N={self.N})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__": 
    sim = DiracTheoryResponseSim(N=60000, k_max=0.05) 
    sim.run()
    sim.save_to_txt()
    sim.plot_results()