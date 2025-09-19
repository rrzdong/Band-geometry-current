# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 11:59:22 2025

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
t1 = 0.5
t2 = 1
delta = 0.5 
J = 1.84


def E(kx, ky, s):
    h_0 = -2*t1*(np.cos(kx+ky)+np.cos(kx-ky))
    h_x = -2*t2*(np.cos(kx)+np.cos(ky))
    h_z = 2*t1*delta*(np.cos(kx+ky) - np.cos(kx-ky)) - J*s
    return h_0 + np.sqrt(h_x**2 + h_z**2), h_0 - np.sqrt(h_x**2 + h_z**2)

def v(kx, ky, s):
    dx = np.pi/(10*200)
    E_plus_p, E_minus_p = E(kx + dx, ky, s)
    E_plus_m, E_minus_m = E(kx - dx, ky, s)
    vx_plus = (E_plus_p - E_plus_m) / (2*dx)
    vx_minus = (E_minus_p - E_minus_m) / (2*dx)

    E_plus_p, E_minus_p = E(kx, ky + dx, s)
    E_plus_m, E_minus_m = E(kx, ky - dx, s)
    vy_plus = (E_plus_p - E_plus_m) / (2*dx)
    vy_minus = (E_minus_p - E_minus_m) / (2*dx)

    return (vx_plus, vx_minus), (vy_plus, vy_minus)

def m_xy(kx, ky, s):
    dx = np.pi/(10*200)

    return (v(kx,ky+dx,s)[0][0]-v(kx,ky-dx,s)[0][0]-v(kx,ky+dx,s)[0][1]+v(kx,ky-dx,s)[0][1])/(2*dx)

# Sampling points
N = 200
dx = np.pi/(10*N)

# Define paths
kx_path1 = np.linspace(0, -np.pi/2, N)
ky_path1 = np.linspace(np.pi, np.pi/2, N)
kx_path2 = np.linspace(-np.pi/2, 0, N)
ky_path2 = np.linspace(np.pi/2, 0, N)
kx_path3 = np.linspace(0, np.pi/2, N)
ky_path3 = np.linspace(0, np.pi/2, N)
kx_path4 = np.linspace(np.pi/2, 0, N)
ky_path4 = np.linspace(np.pi/2, np.pi, N)

# Concatenate path points
kx_path = np.concatenate([kx_path1, kx_path2, kx_path3, kx_path4])
ky_path = np.concatenate([ky_path1, ky_path2, ky_path3, ky_path4])

# Calculate energy and effective mass
E_up_1, E_up_0 = E(kx_path, ky_path, s=1)
E_down_1, E_down_0 = E(kx_path, ky_path, s=-1)
m_up = m_xy(kx_path, ky_path, s=1)
m_down = m_xy(kx_path, ky_path, s=-1)

# x-axis for path points
x_vals = np.arange(len(kx_path))
xticks_pos = [0, N, 2*N, 3*N, 4*N-1]
xticks_label = ["Y", "S'", "Γ", "S", "Y"]


fsize1 = 28
fsize2 = 24
lw=2.5
# Create stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=250, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

# Top subplot: Energy bands
ax1.plot(x_vals, E_up_1, 'r-', label='Spin up',lw=lw)
ax1.plot(x_vals, E_up_0, 'r-',lw=lw)
ax1.plot(x_vals, E_down_1, 'b--', label='Spin down',lw=lw)
ax1.plot(x_vals, E_down_0, 'b--',lw=lw)
ax1.set_ylabel('Energy(eV)',fontsize=fsize1)
ax1.legend(fontsize=17)
ax1.tick_params(axis='y', labelcolor='black',labelsize=fsize2)


# Bottom subplot: Effective mass
ax2.plot(x_vals, m_up, 'r-',lw=lw)
ax2.plot(x_vals, m_down, 'b--',lw=lw)

ax2.set_ylabel('$\\Delta 1/m_{xy}(\\AA^2eV)$',fontsize=fsize1)
ax2.set_xlabel('k-Path',fontsize=fsize1)

ax2.tick_params(axis='y', labelcolor='black',labelsize=fsize2)


# Add vertical lines at high-symmetry points
for pos in xticks_pos[1:-1]:
    ax1.axvline(x=pos, color='gray', linestyle='-', alpha=0.5, zorder=0)
    ax2.axvline(x=pos, color='gray', linestyle='-', alpha=0.5, zorder=0)

# Add horizontal line at y=0 for both axes
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, zorder=0)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, zorder=0)

# Set x-axis ticks and labels
plt.xticks(xticks_pos, xticks_label,fontsize=fsize1)
plt.xlim(0,len(kx_path))

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
