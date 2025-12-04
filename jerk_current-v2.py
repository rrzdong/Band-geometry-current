#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:23:42 2025

@author: dongruizhi
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import matplotlib.lines as mlines

# Constants
t1 = 1
t2 = 0.5
delta = 0.5
J = 5*0.371574
N = 400
a= 2
mu = 0.833307
T=0.000001
h = 0.1
mgOpen = 1

# Grid setup
omega = np.linspace(0.0001, 8, 200)
kx = np.linspace(-np.pi/a, np.pi/a, N)
ky = np.linspace(-np.pi/a, np.pi/a, N)
dx = (kx[1] - kx[0]) / 100
KX, KY = np.meshgrid(kx, ky)  # Vectorized grid

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

@njit
def hamilton(kx, ky, h, J, s):
    h_0 = -2 * t2 * (np.cos(kx*a + ky*a) + np.cos(kx*a - ky*a)) -h*mgOpen*s
    h_x = -2 * t1 * (np.cos(kx*a) + np.cos(ky*a))
    h_z = 2 * t2 * delta * (np.cos(kx*a + ky*a) - np.cos(kx*a - ky*a)) - J * s
    return h_0 * identity + h_x * sigma_x + h_z * sigma_z

@njit
def vx(kx, ky, h, J, s):

    return (hamilton(kx + dx, ky, s) - hamilton(kx - dx, ky, s)) / (2 * dx)

@njit
def vy(kx, ky, h, J, s):

    return (hamilton(kx, ky + dx, s) - hamilton(kx, ky - dx, s)) / (2 * dx)

@njit
def delta_f(E0, omega, delta=0.1):

    return 1 / (delta * np.sqrt(np.pi)) * np.exp(-((E0 - omega) ** 2) / (delta ** 2))

@njit
def mass(kx, ky, h, J, s, theta, phi):

    # Positive displacement
    H_p = hamilton(kx + dx * np.cos(theta), ky + dx * np.sin(theta), s)
    eigval_p, eigvec_p = np.linalg.eigh(H_p)
    vx_p = vx(kx + dx * np.cos(theta), ky + dx * np.sin(theta), s)
    vy_p = vy(kx + dx * np.cos(theta), ky + dx * np.sin(theta), s)
    
    vc_p = (np.vdot(eigvec_p[:, 1], vx_p @ eigvec_p[:, 1]) - 
            np.vdot(eigvec_p[:, 0], vx_p @ eigvec_p[:, 0])) * np.cos(phi) + \
           (np.vdot(eigvec_p[:, 1], vy_p @ eigvec_p[:, 1]) - 
            np.vdot(eigvec_p[:, 0], vy_p @ eigvec_p[:, 0])) * np.sin(phi)
    
    # Negative displacement
    H_m = hamilton(kx - dx * np.cos(theta), ky - dx * np.sin(theta), s)
    eigval_m, eigvec_m = np.linalg.eigh(H_m)
    vx_m = vx(kx - dx * np.cos(theta), ky - dx * np.sin(theta), s)
    vy_m = vy(kx - dx * np.cos(theta), ky - dx * np.sin(theta), s)
    
    vc_m = (np.vdot(eigvec_m[:, 1], vx_m @ eigvec_m[:, 1]) - 
            np.vdot(eigvec_m[:, 0], vx_m @ eigvec_m[:, 0])) * np.cos(phi) + \
           (np.vdot(eigvec_m[:, 1], vy_m @ eigvec_m[:, 1]) - 
            np.vdot(eigvec_m[:, 0], vy_m @ eigvec_m[:, 0])) * np.sin(phi)
    
    return (vc_p - vc_m) / (2 * dx)

def fermi(energy, mu):
        arg = (energy - mu) / ( T)
        return 1.0 / (np.exp(arg) + 1)


def compute_current(s, c , d, y, h, mu, J, omega):

    sigma = np.zeros(len(omega), dtype=np.complex128)
    
    for i in prange(N):
        for j in prange(N):
            kx_ = KX[i, j]
            ky_ = KY[i, j]
            
            # Hamiltonian and velocity operators
            H = hamilton(kx_, ky_, s) 
            eigval, eigvec = np.linalg.eigh(H)
            vx_ = vx(kx_, ky_, s)
            vy_ = vy(kx_, ky_, s)
            
            # Transition matrix elements
            if y == 0:
               va = vx_
               vb = vx_
            else:
                va = vy_
                vb = vy_
            transition_ = np.vdot(eigvec[:, 0], va @ eigvec[:, 1]) * \
                          np.vdot(eigvec[:, 1], vb @ eigvec[:, 0])
            
            # Mass term
            m_ = mass(kx_, ky_, s, c, d)
            
            # Sum over omega
            E_diff = eigval[1] - eigval[0]
            f_diff = fermi(eigval[0], mu) - fermi(eigval[1], mu)
            if E_diff < 0:
                print ('error')
            for k in range(len(omega)):
                sigma[k] += f_diff*transition_ * m_ * delta_f(E_diff, omega[k])/omega[k]**2 * (kx[1]-kx[0])**2/(2*np.pi)**2
    
    return sigma

# result_xxy = {}
# for s in [1, -1]:
#     result_xxy[s] = compute_current(s, c = 0, d = np.pi/2, y=0) 
    
# result_xxx = {}
# for s in [1, -1]:
#     result_xxx[s] = compute_current(s, c = 0, d = 0, y=0) 
    
# result_yxy = {}
# for s in [1, -1]:
#     result_yxy[s] = compute_current(s, c = 0, d = np.pi/2, y=1) 
    
# result_yxx = {}
# for s in [1, -1]:
#     result_yxx[s] = compute_current(s, c = 0, d = 0, y=1)

# # plot
# fsize = 26

# plt.figure(figsize=(8, 6), dpi=250)

# plt.plot(omega, (result_xxy[1] + result_xxy[-1]) * 3.55 / 2, color='r', lw=2.5)
# plt.plot(omega, (result_xxx[1] + result_xxx[-1]) * 3.55 / 2, color='g', lw=2.5)
# plt.plot(omega, (result_yxy[1] + result_yxy[-1]) * 3.55 / 2, color='b', linestyle='--', lw=2.5)
# plt.plot(omega, (result_yxx[1] + result_yxx[-1]) * 3.55 / 2, color='k', linestyle='--', lw=2.5)

# plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=0)
# plt.xlabel('Photon energy $\\hbar\\omega (eV)$', fontsize=fsize+2)
# plt.ylabel(r'$10^{-8}Am/V^3$', fontsize=fsize+2)
# plt.title('Linear \'effective mass\' ', fontsize=fsize+2)
# plt.xlim(0, 8)
# plt.xticks(fontsize=fsize - 2)
# plt.yticks(fontsize=fsize - 2)

# plt.legend( fontsize=fsize - 4,
#            title='Spin current', title_fontsize=fsize-4)

# plt.show()

# # save datas
# data_xxy = ( (result_xxy[1]+ result_xxy[-1]) * 3.55 / 2).real  
# data_xxx = ((result_xxx[1] + result_xxx[-1]) * 3.55 / 2).real
# data_yxy = ((result_yxy[1] + result_yxy[-1]) * 3.55 / 2).real
# data_yxx = ((result_yxx[1] + result_yxx[-1]) * 3.55 / 2).real

# output_array = np.column_stack((omega, data_xxy, data_xxx, data_yxy, data_yxx))

# header = 'omega result_xxy result_xxx result_yxy result_yxx'
# np.savetxt('output_data_charge.txt', output_array, header=header, comments='', fmt='%.8e')

# result_xxy = {}
# for s in [1, -1]:
#     result_xxy[s] = compute_current(s, c = 0, d = np.pi/2, y=0) 
    
# plt.plot(omega, (result_xxy[1] - result_xxy[-1]) * 3.55 / 2, color='r', lw=2.5)
# data_xxy = ( (result_xxy[1]- result_xxy[-1]) * 3.55 / 2).real

# output_array = np.column_stack((omega, data_xxy))

# header = 'omega result_xxy'
# np.savetxt(f'output_data_spin-h={h*mgOpen}.txt', output_array, header=header, comments='', fmt='%.8e')

omega = 3
data = np.loadtxt('scf_results.txt', skiprows=1)
h = data[:, 0]
mu = data[:, 1]
deltam = data[:, 2]

J = 5 * deltam

spin = []  # 用列表存储

for i in range(len(J)):
    up_ = compute_current(1, c=0, d=np.pi/2, y=0, h=h[i], mu=mu[i], J=J[i], omega=omega)
    down_ = compute_current(-1, c=0, d=np.pi/2, y=0, h=h[i], mu=mu[i], J=J[i], omega=omega)
    spin_ = (up_ - down_).real * 3.55 / 2
    
    spin.append([h[i], spin_])

spin = np.array(spin)  # 转成 numpy 数组便于保存
np.savetxt('spin.txt', spin, comments='')




