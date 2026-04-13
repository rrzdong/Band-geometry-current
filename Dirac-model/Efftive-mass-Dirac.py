#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:16:58 2026

@author: dongruizhi
"""

import numpy as np
import matplotlib.pyplot as plt

class DiracTheoryResponseSim:
    def __init__(self, hbar_vF=5.266*10000, m=8.5, N=1000, k_max=0.05, tilt_x=0.0):
        """
        单位说明：
        hbar_vF: 52660 meV*A
        m: 8.5 meV (Gap = 17 meV)
        N: k空间采样点数 (N x N)
        k_max: 0.05 A^-1
        tilt_x: 倾斜项系数 (默认0，若设为如 5000 则会打破对称性使虚部积分非零)
        """
        self.hbar_vF = hbar_vF
        self.m = m
        self.N = N
        self.k_max = k_max
        self.tilt_x = tilt_x 
        
        
        self.omega = np.linspace(0.1, 50, 400)
        
        
        self.kx_lin = np.linspace(-k_max, k_max, N)
        self.ky_lin = np.linspace(-k_max, k_max, N)
        self.dk = self.kx_lin[1] - self.kx_lin[0]
        self.KX, self.KY = np.meshgrid(self.kx_lin, self.ky_lin)
        
        self.results = None
        
        self.factor = 7.6471/6

    def compute_imaginary_response(self):
        """
        计算虚部项: Im(r_{+-}^x r_{-+}^y) * d2E/dkxky
        公式推导结果: 32 * m * vF^6 * kx * ky / E_diff^6
        """
        print(f"开始计算虚部响应 (N={self.N}, Tilt={self.tilt_x})...")
        
        hvF = self.hbar_vF
        hvF2 = hvF**2
        hvF6 = hvF**6
        m = self.m
        
        k2 = self.KX**2 + self.KY**2
        e_diff = 2.0 * np.sqrt(hvF2 * k2 + m**2)
        term = 32.0 * m * hvF6 * self.KX * self.KY / (e_diff**6)
        
        sigma_res = np.zeros(len(self.omega))
        gauss_width = 0.1 
        prefactor = (self.dk**2) / (4.0 * np.pi**2)
        
        
        for i, w in enumerate(self.omega):
            
            diff = e_diff - w
            
            delta_fun = (1.0 / (gauss_width * np.sqrt(np.pi))) * \
                        np.exp(-(diff**2) / (gauss_width**2))
            
            sigma_res[i] = np.sum(term * delta_fun) * prefactor
            
        self.results = sigma_res
        print("计算完成。")

    def save_to_txt(self, filename="K-term_imag.txt"):
        if self.results is None: return
        data_to_save = np.column_stack((self.omega, self.results * self.factor))
        np.savetxt(filename, data_to_save, fmt='%.10e')
        print(f"数据已保存至 {filename}")

    def plot_results(self):
        if self.results is None: return
        
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(self.omega, self.results * self.factor, color='darkorange', lw=2, 
                 label=r'Im($r_{xy} \cdot \partial_{xy} E_{+-}$)')
        
        plt.axvline(2*abs(self.m), color='black', linestyle='--', label=f'Gap={2*abs(self.m)} meV')
        plt.axhline(0, color='gray', lw=1)
        
        plt.xlabel(r'Photon energy $\hbar\omega$ (meV)')
        plt.ylabel(r'Response $F(\omega) (10^{-20}) $ ')
        plt.title('Efftive mass in Dirac')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__":

    sim = DiracTheoryResponseSim(N=5000, m=8.5, tilt_x=0.0)
    sim.compute_imaginary_response()
    sim.save_to_txt()
    sim.plot_results()