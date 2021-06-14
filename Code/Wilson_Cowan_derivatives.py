#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:21:58 2021

@author: pierrehouzelstein
"""

import numpy as np
from numpy import loadtxt, pi, sqrt, cos, sin, arctan, zeros, linspace, mod, arctan2
import matplotlib.pyplot as plt
from WilsonCowanv3 import EulerInteg

# Plot derivatives in phase space

limit_cycle_data = loadtxt('./data/WilsonCowan/limCycleData0.1')
grad_x = loadtxt('./data/WilsonCowan/gradTheta_x.txt')
grad_y = loadtxt('./data/WilsonCowan/gradTheta_y.txt')
shifted_LC = loadtxt('./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_shifted_LC_S=1.6.txt')

LC_x = limit_cycle_data[:,0]
LC_y = limit_cycle_data[:,1]
max_index = np.argmax(LC_x)
phase_0 = np.array([LC_x[max_index], LC_y[max_index]])

#Integrate trajectory for alveus
h=0.01; N = 100 #average each trajectory over N periods to get mean phase
gee = 10.; gei = -10.; gie = 12.; gii = -10.
tau_e = 3.; tau_i = 8.; tau_stim = 6.
ae = -2.; ai = -3.5
noise_amplitude = 0.
    
#Simulation #1 (fig 2e): parameters
T = 33.126711029860324
numsteps = T//h
#Fig.2d:σ= 0.13, θ=85°,S0 =1, 1.5, 4 and 10
#Fig. 3b: σ = 0.2–0.5, θ = 80°, S0 = 0.4, 1.6, 3 and 12 - alveus
#fig 3d: σ = 0.4–0.6, θ = 60°, S0 = 0.5, 2.2, 4.1 and 8 - dentate
sigma = 0.4
S0 = 1.6; theta = 80*(pi/180)
pulse = np.array([S0*cos(theta), S0*sin(theta)])

time_list_1, trajectory_1 = EulerInteg(phase_0, h, numsteps, noise_amplitude, pulse, sigma, tau_e, tau_i,\
               tau_stim, gee, gei, gie, gii, ae, ai)

#Integrate trajectory for dentate
S0 = 2.2; theta = 60*(pi/180)
pulse = np.array([S0*cos(theta), S0*sin(theta)])

time_list_2, trajectory_2 = EulerInteg(phase_0, h, numsteps, noise_amplitude, pulse, sigma, tau_e, tau_i,\
               tau_stim, gee, gei, gie, gii, ae, ai)


yp = 0.75; ym = -0.25
xp = 1.0;  xm = -0.25
y = linspace(ym, yp, 100)
x = linspace(xm, xp, 100)

plt.pcolormesh(x, y, grad_x, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(LC_x[max_index], LC_y[max_index], "ro")
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')

shifted_LC = loadtxt('./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_shifted_LC_S=2.2.txt')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'royalblue')

index_end_stim = int(tau_stim//h)
plt.plot(trajectory_1[:,0][index_end_stim], trajectory_1[:,1][index_end_stim], "ko")
plt.plot(trajectory_2[:,0][index_end_stim], trajectory_2[:,1][index_end_stim], "ko")

plt.title("Wilson-Cowan phase space: x-derivatives")
plt.legend(["Limit cycle", "Phase 0 point", "Positions after stimulus: alveus", "Positions after stimulus: dentate", "Position of phase 0 point after stimulation"])
plt.xlabel("E")
plt.ylabel("I")
cbar=plt.colorbar(label="grad x", orientation="vertical")
#cbar.set_ticks([1, 2, 3, 4, 5, 6])
#cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
plt.show()

shifted_LC = loadtxt('./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_shifted_LC_S=1.6.txt')
plt.pcolormesh(x, y, grad_y, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(LC_x[max_index], LC_y[max_index], "ro")
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')

shifted_LC = loadtxt('./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_shifted_LC_S=2.2.txt')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'white')
plt.plot(trajectory_1[:,0][index_end_stim], trajectory_1[:,1][index_end_stim], "wo")
plt.plot(trajectory_2[:,0][index_end_stim], trajectory_2[:,1][index_end_stim], "wo")

plt.title("Wilson-Cowan phase space: y-derivatives")
plt.legend(["Limit cycle", "Phase 0 point", "Positions after stimulus: alveus", "Positions after stimulus: dentate", "Position of phase 0 point after stimulation"])
plt.xlabel("E")
plt.ylabel("I")
cbar=plt.colorbar(label="grad y", orientation="vertical")
#cbar.set_ticks([1, 2, 3, 4, 5, 6])
#cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
plt.show()