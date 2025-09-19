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
t1 = 0.5
t2 = 1
delta = 0.5
J = 1
N = 400
a=2

# Grid setup
omega = np.linspace(0.0001, 10, 200)
kx = np.linspace(-np.pi/a, np.pi/a, N)
ky = np.linspace(-np.pi/a, np.pi/a, N)
dx = (kx[1] - kx[0]) / 100
KX, KY = np.meshgrid(kx, ky)  # Vectorized grid

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

@njit
def hamilton(kx, ky, s):

    h_0 = -2 * t1 * (np.cos(kx*a + ky*a) + np.cos(kx*a - ky*a))
    h_x = -2 * t2 * (np.cos(kx*a) + np.cos(ky*a))
    h_z = 2 * t1 * delta * (np.cos(kx*a + ky*a) - np.cos(kx*a - ky*a)) - J * s
    return h_0 * identity + h_x * sigma_x + h_z * sigma_z

@njit
def vx(kx, ky, s):

    return (hamilton(kx + dx, ky, s) - hamilton(kx - dx, ky, s)) / (2 * dx)

@njit
def vy(kx, ky, s):

    return (hamilton(kx, ky + dx, s) - hamilton(kx, ky - dx, s)) / (2 * dx)

@njit
def delta_f(E0, omega, delta=0.1):

    return 1 / (delta * np.sqrt(np.pi)) * np.exp(-((E0 - omega) ** 2) / (delta ** 2))


def compute_current(s, c , d):

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
            
            vc = vx_*np.cos(c) + vy_*np.sin(c)
            vd = vx_*np.cos(d) + vy_*np.sin(d)
            
            # Transition matrix elements
            va = vx_
            vb = vy_
            transition_ = np.vdot(eigvec[:, 0], va @ eigvec[:, 1]) * \
                          np.vdot(eigvec[:, 1], vb @ eigvec[:, 0])
                          
            vv = np.vdot(eigvec[:, 0], vc @ eigvec[:, 0]) * np.vdot(eigvec[:, 0], vd @ eigvec[:, 0]) - \
                 np.vdot(eigvec[:, 1], vc @ eigvec[:, 1]) * np.vdot(eigvec[:, 1], vd @ eigvec[:, 1])
            # Sum over omega
            E_diff = eigval[1] - eigval[0]
            if E_diff < 0:
                print ('error')
            for k in range(len(omega)):
                sigma[k] += transition_ * vv * delta_f(E_diff, omega[k])/omega[k]**2 * (kx[1]-kx[0])**2/(2*np.pi)**2
    
    return sigma

result_xy = {}
for s in [1, -1]:
    result_xy[s] = compute_current(s, c = 0, d = np.pi/2)
    
result_xx = {}
for s in [1, -1]:
    result_xx[s] = compute_current(s, c = 0, d = 0)
    
result_yx = {}
for s in [1, -1]:
    result_yx[s] = compute_current(s, c = np.pi/2, d = 0 )
    
result_yy = {}
for s in [1, -1]:
    result_yy[s] = compute_current(s, c = np.pi/2, d = np.pi/2)
    

fsize = 26
plt.figure(figsize=(8, 6), dpi=250)
plt.plot(omega, (result_xy[1]-result_xy[-1])*17.75/2, color='r',lw=2.5)
plt.plot(omega, (result_xx[1]-result_xx[-1])*17.75/2, color='g',lw=2.5)
plt.plot(omega, (result_yx[1]-result_yx[-1])*17.75/2, color='b',linestyle='--',lw=2.5)
plt.plot(omega, (result_yy[1]-result_yy[-1])*17.75/2, color='k',linestyle='--',lw=2.5)


plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=0)
plt.xlabel('Photon energy $\\hbar\\omega (eV)$', fontsize=fsize+2)
plt.ylabel(r'$10^{-6}Am/V^3$', fontsize=fsize+2)
plt.title('Circular \'group velocity\'', fontsize=fsize+2)
plt.xlim(0, 9)
plt.xticks(fontsize=fsize-2)
plt.yticks(fontsize=fsize-2)
plt.legend( fontsize=fsize - 4,
            title='Spin current', title_fontsize=fsize-4)

plt.show()


# data_xy = ((result_xy[1] + result_xy[-1]) * 17.75 / 2).real  # 实部
# data_xx = ((result_xx[1] + result_xx[-1]) * 17.75 / 2).real
# data_yx = ((result_yx[1] + result_yx[-1]) * 17.75 / 2).real
# data_yy = ((result_yy[1] + result_yy[-1]) * 17.75 / 2).real

# output_array = np.column_stack((omega, data_xy, data_xx, data_yx, data_yy))

# header = 'omega result_xy result_xx result_yx result_yy'
# np.savetxt('output_data_charge.txt', output_array, header=header, comments='', fmt='%.8e')



    



