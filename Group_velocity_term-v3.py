# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 23:25:35 2026

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

class HallConductivitySim:
    def __init__(self, t1=1.0, t2=0.5, delta=0.5, U=5.0, deltam=0.3, N=400, a=2, mu=0.1):
        # 物理参数初始化
        self.t1 = t1
        self.t2 = t2
        self.delta = delta
        self.J = U*deltam
        self.N = N
        self.a = a
        self.mu = mu
        self.U = U
        
        # 网格与常数设置
        self.omega = np.linspace(0.1, 4, 500)
        self.kx = np.linspace(-np.pi/a, np.pi/a, N)
        self.ky = np.linspace(-np.pi/a, np.pi/a, N)
        self.dk = self.kx[1] - self.kx[0]
        self.factor = 17.75 / 2
        
        # 存储原始计算结果: {s_value: complex_array}
        self.results = {}

    @staticmethod
    @njit(parallel=True)
    def _compute_core(omega, kx, ky, dk, s, c, d, t1, t2, delta_param, J, a, mu):
        """
        核心计算逻辑：使用 Numba 强力加速
        """
        num_omega = len(omega)
        N_grid = len(kx)
        sigma = np.zeros(num_omega, dtype=np.complex128)
        dx_diff = dk / 100.0
        
        # 定义 Pauli 矩阵
        sig_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sig_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        eye = np.eye(2, dtype=np.complex128)
        prefactor = (dk**2) / (4 * np.pi**2)

        for i in prange(N_grid):
            for j in range(N_grid):
                k_x, k_y = kx[i], ky[j]

                # 内部哈密顿量函数
                def get_h(kx_v, ky_v):
                    h0 = -2 * t2 * (np.cos(kx_v*a + ky_v*a) + np.cos(kx_v*a - ky_v*a))
                    hx = -2 * t1 * (np.cos(kx_v*a) + np.cos(ky_v*a))
                    hz = 2 * t2 * delta_param * (np.cos(kx_v*a + ky_v*a) - np.cos(kx_v*a - ky_v*a)) - J * s
                    return h0 * eye + hx * sig_x + hz * sig_z

                # 求解特征值与特征向量
                H = get_h(k_x, k_y)
                eigval, eigvec = np.linalg.eigh(H)
                
                # 有限差分计算速度算符
                vx_mat = (get_h(k_x + dx_diff, k_y) - get_h(k_x - dx_diff, k_y)) / (2 * dx_diff)
                vy_mat = (get_h(k_x, k_y + dx_diff) - get_h(k_x, k_y - dx_diff)) / (2 * dx_diff)

                # 投影速度
                vc = vx_mat * np.cos(c) + vy_mat * np.sin(c)
                vd = vx_mat * np.cos(d) + vy_mat * np.sin(d)
                
                # 矩阵元计算
                v01 = eigvec[:, 0].conj() @ (vx_mat @ eigvec[:, 1])
                v10 = eigvec[:, 1].conj() @ (vy_mat @ eigvec[:, 0])
                trans_val = v01 * v10

                vcc0 = eigvec[:, 0].conj() @ (vc @ eigvec[:, 0])
                vdd0 = eigvec[:, 0].conj() @ (vd @ eigvec[:, 0])
                vcc1 = eigvec[:, 1].conj() @ (vc @ eigvec[:, 1])
                vdd1 = eigvec[:, 1].conj() @ (vd @ eigvec[:, 1])
                vv_val = vcc0 * vdd0 - vcc1 * vdd1

                # 能量差与费米占据数差 (T -> 0 近似)
        
                e_diff = eigval[1] - eigval[0]
                flag = 1
                if e_diff <0.001:
                    flag = 0.000000000001

                f0 = 1.0 / (1.0 + np.exp((eigval[0] - mu) / 0.0001))
                f1 = 1.0 / (1.0 + np.exp((eigval[1] - mu) / 0.0001))
                f_diff = f0 - f1

                # 能量谱累加
                for k in range(num_omega):
                    # 高斯函数模拟 Delta 函数
                    d=0.01
                    gauss = (1.0 / (d * np.sqrt(np.pi))) * np.exp(-((e_diff - omega[k])**2) / d**2)
                    # sigma[k] += f_diff * trans_val * vv_val * (gauss / (omega[k]**2)) * prefactor*flag
                    sigma[k] += f_diff * trans_val * vv_val * (gauss / (omega[k]**2)) * prefactor*flag
        return sigma

    def run(self, s_values=[1, -1], c=0, d=0):
        """执行计算并存储 s=1 和 s=-1 的原始数据"""
        for s in s_values:
            print(f"Calculating: s={s}, t1={self.t1:.4f}, mu={self.mu:.4f}...")
            self.results[s] = self._compute_core(
                self.omega, self.kx, self.ky, self.dk, s, c, d,
                self.t1, self.t2, self.delta, self.J, self.a, self.mu
            )

    def _get_combined_data(self, mode):
        """内部辅助函数：处理相加或相减逻辑"""
        if 1 not in self.results or -1 not in self.results:
            return None
        
        if mode.lower() == "spin":
            return (self.results[1] - self.results[-1]) * self.factor
        elif mode.lower() == "charge":
            return (self.results[1] + self.results[-1]) * self.factor
        return None

    def plot_results(self, mode="spin"):
        """绘图：可选 'spin' (相减) 或 'charge' (相加)"""
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
        filename = f'{mode.lower()}-U={self.U}.txt'
        header = f'omega(eV)  {mode}_result_real'
        np.savetxt(filename, output_array, header=header, fmt='%.8e')
        print(f"Data successfully saved to: {filename}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 从外部文件读取参数

    raw_data = np.loadtxt('delta_m_vs_U2.txt', skiprows=1)
            
    Us = raw_data[:, 0]
    mus = raw_data[:, 1]
    deltas = raw_data[:, 2]


    for i in range(len(Us)):

        sim = HallConductivitySim(
            N=3000,             # 建议先用 200 测试速度，正式计算用 400
            mu=mus[i], 
            U=Us[i],
            deltam=deltas[i]
        )
        
        # 3. 运行核心计算 (以 xx 分量 c=0, d=0 为例)
        sim.run(c=0, d=0)
        
        # 4. 绘图与保存 (Spin 模式)
        sim.plot_results(mode="spin")
        sim.save_data(mode="spin")
        
    #     # # 5. 绘图与保存 (Charge 模式)
    #     # sim.plot_results(mode="charge")
    #     # sim.save_data(mode="charge")
    
    # sim = HallConductivitySim(
    #         N=4000,             # 建议先用 200 测试速度，正式计算用 400
    #         mu=0.612480, 
    #         U=5,
    #         deltam=0.385365,
    #         t2=0.5
    #     )
    # sim.run(c=0, d=0)
    # sim.plot_results(mode="spin")
    # # sim.save_data(mode="spin")