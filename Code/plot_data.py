#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:32:08 2021

@author: pierrehouzelstein
"""
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre
from numpy import loadtxt, pi, sqrt, cos, sin, arctan, zeros, linspace, mod, arctan2
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from WilsonCowanv3 import EulerInteg
from mpl_toolkits import mplot3d
from matplotlib import cm
"""
isochrone = loadtxt('./data/WilsonCowan/isocronesD0.1')
isochrones_real = loadtxt('./data/WilsonCowan/realValuesD0.1')
isochrones_im = loadtxt('./data/WilsonCowan/imagValuesD0.1')
limit_cycle_data = loadtxt('./data/WilsonCowan/limCycleData0.1')
    
#interpolate
#setting up grid for the phase spaces
yp = 0.75; ym = -0.25
xp = 1.0;  xm = -0.25
y = linspace(ym, yp, 100)
x = linspace(xm, xp, 100)
real_isochrone_func = interpolate.interp2d(\
                    x, y, isochrones_real, kind = 'cubic')
im_isochrone_func = interpolate.interp2d(\
                    x, y, isochrones_im, kind = 'cubic')

#Plot alveus results
fig = plt.figure(1)

free_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=0.4/WilsonCowan_free_phase_S=0.4.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=0.4/WilsonCowan_pert_phase_S=0.4.txt")
PRC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=0.4/WilsonCowan_PRC_S=0.4.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=0.4/WilsonCowan_shifted_LC_S=0.4.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 1) # 4 rows, 2 columns, index 1
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.title("Phase space")
plt.legend(["Limit cycle", "Positions after stimulus"])
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])

plt.subplot(4, 3, 2) 
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.title("Phases of the alveus oscillations")
plt.legend(["Initial phase", "Phase one period after stimulus"])

plt.subplot(4, 3, 3)
plt.plot(free_phase, PRC, "g+")
plt.title("Alveus PRC")

free_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=1.6/WilsonCowan_free_phase_S=1.6.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=1.6/WilsonCowan_pert_phase_S=1.6.txt")
PRC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=1.6/WilsonCowan_PRC_S=1.6.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=1.6/WilsonCowan_shifted_LC_S=1.6.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
plt.subplot(4, 3, 4)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 5)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")

plt.subplot(4, 3, 6)
plt.plot(free_phase, PRC, "g+")

free_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=3/WilsonCowan_free_phase_S=3.0.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=3/WilsonCowan_pert_phase_S=3.0.txt")
PRC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=3/WilsonCowan_PRC_S=3.0.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=3/WilsonCowan_shifted_LC_S=3.0.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 7)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 8)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")

plt.subplot(4, 3, 9)
plt.plot(free_phase, PRC, "g+")

free_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=12/WilsonCowan_free_phase_S=12.0.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=12/WilsonCowan_pert_phase_S=12.0.txt")
PRC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=12/WilsonCowan_PRC_S=12.0.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/alveus_figure_3b/S=12/WilsonCowan_shifted_LC_S=12.0.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 10)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.xlabel("E")
plt.ylabel("I")
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 11)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.xlabel("Initial $\Theta$")
plt.ylabel("New $\Theta$")

plt.subplot(4, 3, 12)
plt.plot(free_phase, PRC, "g+")
plt.xlabel("Initial $\Theta$")
plt.ylabel("Shift")

#fig.tight_layout()
fig.set_size_inches(w=12,h=11)
plt.savefig("./data/WilsonCowan/output/alveus_results.jpg")
plt.show()

#Plot dentate results
fig = plt.figure(1)

free_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=0.5/WilsonCowan_free_phase_S=0.5.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=0.5/WilsonCowan_pert_phase_S=0.5.txt")
PRC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=0.5/WilsonCowan_PRC_S=0.5.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=0.5/WilsonCowan_shifted_LC_S=0.5.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 1) # 4 rows, 2 columns, index 1
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.title("Phase space")
plt.legend(["Limit cycle", "Positions after stimulus"])
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])

plt.subplot(4, 3, 2) 
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.title("Phases of the dentate oscillations")
plt.legend(["Initial phase", "Phase one period after stimulus"])

plt.subplot(4, 3, 3)
plt.plot(free_phase, PRC, "g+")
plt.title("Dentate PRC")

free_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=2.2/WilsonCowan_free_phase_S=2.2.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=2.2/WilsonCowan_pert_phase_S=2.2.txt")
PRC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=2.2/WilsonCowan_PRC_S=2.2.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=2.2/WilsonCowan_shifted_LC_S=2.2.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
plt.subplot(4, 3, 4)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 5)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")

plt.subplot(4, 3, 6)
plt.plot(free_phase, PRC, "g+")

free_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=4.1/WilsonCowan_free_phase_S=4.1.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=4.1/WilsonCowan_pert_phase_S=4.1.txt")
PRC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=4.1/WilsonCowan_PRC_S=4.1.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=4.1/WilsonCowan_shifted_LC_S=4.1.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 7)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 8)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")

plt.subplot(4, 3, 9)
plt.plot(free_phase, PRC, "g+")

free_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=8/WilsonCowan_free_phase_S=8.0.txt")
shifted_phase = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=8/WilsonCowan_pert_phase_S=8.0.txt")
PRC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=8/WilsonCowan_PRC_S=8.0.txt")
shifted_LC = loadtxt("data/WilsonCowan/output/dentate_figure_3d/S=8/WilsonCowan_shifted_LC_S=8.0.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 10)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.xlabel("E")
plt.ylabel("I")
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 11)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.xlabel("Initial $\Theta$")
plt.ylabel("New $\Theta$")

plt.subplot(4, 3, 12)
plt.plot(free_phase, PRC, "g+")
plt.xlabel("Initial $\Theta$")
plt.ylabel("Shift")

#fig.tight_layout()
fig.set_size_inches(w=13,h=13)
plt.savefig("./data/WilsonCowan/output/dentate_results.jpg")
plt.show()

#Plot AR2 results
fig = plt.figure(1)

#load data
isochrone = loadtxt('./data/AR2/isocronesD0.01')
isochrones_real = loadtxt('./data/AR2/realValuesD0.01')
isochrones_im = loadtxt('./data/AR2/imagValuesD0.01')
limit_cycle_data = loadtxt('./data/AR2/limCycleData0.01')

#interpolate
 #setting up grid for the phase spaces
yp = 0.5; ym = -0.5
xp = 0.5; xm = -0.5
y = linspace(ym, yp, 80+1)
x = linspace(xm, xp, 80+1)
real_isochrone_func = interpolate.interp2d(\
                x, y, isochrones_real, kind = 'cubic')
im_isochrone_func = interpolate.interp2d(\
                x, y, isochrones_im, kind = 'cubic')

free_phase = loadtxt("data/AR2/output/pulse=0.025/AR2_free_phase_pulse=0.025.txt")
shifted_phase = loadtxt("data/AR2/output/pulse=0.025/AR2_pert_pulse=0.025.txt")
PRC = loadtxt("data/AR2/output/pulse=0.025/AR2_PRC_pulse=0.025.txt")
shifted_LC = loadtxt("data/AR2/output/pulse=0.025/AR2_shifted_LC_pulse=0.025.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 1) # 3 rows, 3 columns, index 1
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.title("AR2 phase space")
plt.text(0., -0.4, "pulse of 0.05", color = "white")
plt.legend(["Limit cycle", "Positions after stimulus"])
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])

plt.subplot(4, 3, 2) 
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.title("Phases of the oscillations")
plt.legend(["Initial phase", "Phase one period after stimulus"])

plt.subplot(4, 3, 3)
plt.plot(free_phase, PRC, "g+")
plt.title("AR2 model: PRC")
        
free_phase = loadtxt("data/AR2/output/pulse=0.05/AR2_free_phase_pulse=0.05.txt")
shifted_phase = loadtxt("data/AR2/output/pulse=0.05/AR2_pert_pulse=0.05.txt")
PRC = loadtxt("data/AR2/output/pulse=0.05/AR2_PRC_pulse=0.05.txt")
shifted_LC = loadtxt("data/AR2/output/pulse=0.05/AR2_shifted_LC_pulse=0.05.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 4) # 3 rows, 3 columns, index 1
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.title("AR2 phase space")
plt.text(0., -0.4, "pulse of 0.05", color = "white")
plt.legend(["Limit cycle", "Positions after stimulus"])
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])

plt.subplot(4, 3, 5) 
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.title("Phases of the oscillations")
plt.legend(["Initial phase", "Phase one period after stimulus"])

plt.subplot(4, 3, 6)
plt.plot(free_phase, PRC, "g+")
plt.title("AR2 model: PRC")

free_phase = loadtxt("data/AR2/output/pulse=0.1/AR2_free_phase_pulse=0.1.txt")
shifted_phase = loadtxt("data/AR2/output/pulse=0.1/AR2_pert_pulse=0.1.txt")
PRC = loadtxt("data/AR2/output/pulse=0.1/AR2_PRC_pulse=0.1.txt")
shifted_LC = loadtxt("data/AR2/output/pulse=0.1/AR2_shifted_LC_pulse=0.1.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
plt.subplot(4, 3, 7)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.text(0., -0.4, "pulse of 0.1", color = "white")
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 8)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")

plt.subplot(4, 3, 9)
plt.plot(free_phase, PRC, "g+")

free_phase = loadtxt("data/AR2/output/pulse=0.3/AR2_free_phase_pulse=0.3.txt")
shifted_phase = loadtxt("data/AR2/output/pulse=0.3/AR2_pert_pulse=0.3.txt")
PRC = loadtxt("data/AR2/output/pulse=0.3/AR2_PRC_pulse=0.3.txt")
shifted_LC = loadtxt("data/AR2/output/pulse=0.3/AR2_shifted_LC_pulse=0.3.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.subplot(4, 3, 10)
plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
plt.plot(shifted_LC[:,0], shifted_LC[:,1], color = 'grey')
plt.text(0., -0.4, "pulse of 0.3", color = "white")
plt.xlabel("E")
plt.ylabel("I")
cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    
plt.subplot(4, 3, 11)
plt.plot(free_phase, free_phase, "b+")
plt.plot(free_phase, shifted_phase, "r+")
plt.xlabel("Initial $\Theta$")
plt.ylabel("New $\Theta$")

plt.subplot(4, 3, 12)
plt.plot(free_phase, PRC, "g+")
plt.xlabel("Initial $\Theta$")
plt.ylabel("Shift")

#fig.tight_layout()
fig.set_size_inches(w=13,h=13)
plt.savefig("./data/AR2/output/AR2_results.jpg")
plt.show()

free_phase = loadtxt("data/AR2/output/pulse=0.025/AR2_free_phase_pulse=0.025.txt")
PRC = loadtxt("data/AR2/output/pulse=0.025/AR2_PRC_pulse=0.025.txt")

plt.plot(free_phase, PRC, "g+")
plt.title("AR2 using the points on the limit cycle; E-shift of 0.025, one period")
plt.xlabel("Initial $\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.savefig("./data/AR2/output/AR2_PRC_pulse=0.025.jpg")
plt.show()
"""
"""
#figure 4: plot weak stim PRCs of alveus and dentate

free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_free_phase_S=1.6.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_PRC_S=1.6.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowen_variance_S=1.6.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.tick_params("both", direction = "in")

ax.plot(free_phase/(2*pi), PRC/(2*pi), "r-", label="_Hidden_1")
ax.plot(free_phase/(2*pi), PRC/(2*pi), "r.", label = "Alveus", markersize=10)
ax.fill_between(free_phase/(2*pi), (PRC+variance)/(2*pi), (PRC-variance)/(2*pi), color="orangered", alpha=.4)
"""
"""
plt.plot(free_phase/(2*pi), PRC/(2*pi), "r-", label="_Hidden_1")
plt.plot(free_phase/(2*pi), PRC/(2*pi), "r.", label = "Alveus", markersize=10)
plt.fill_between(free_phase/(2*pi), (PRC+variance)/(2*pi), (PRC-variance)/(2*pi), color="orangered", alpha=.4)
"""
"""

free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_free_phase_S=2.2.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_PRC_S=2.2.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowen_variance_S=2.2.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
"""
"""
plt.plot(free_phase/(2*pi), PRC/(2*pi), "b-", label = "_Hidden_2")
plt.plot(free_phase/(2*pi), PRC/(2*pi), "b.", label = "Dentate", markersize=10)
plt.fill_between(free_phase/(2*pi), (PRC+variance)/(2*pi), (PRC-variance)/(2*pi), color="blue", alpha=.4)
"""
"""
ax.plot(free_phase/(2*pi), PRC/(2*pi), "b-", label="_Hidden_2")
ax.plot(free_phase/(2*pi), PRC/(2*pi), "b.", label = "Dentate", markersize=10)
ax.fill_between(free_phase/(2*pi), (PRC+variance)/(2*pi), (PRC-variance)/(2*pi), color="blue", alpha=.4)
plt.axhline(y=0, xmin=0, xmax=1, color="black")

ax.set_xlabel("Initial $\Theta$")
ax.set_title('sine')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([-0.5, 0, 0.5])
ax.set_yticklabels(['','0','0.5'])

plt.title("Computed PRCs using data from the paper")
plt.legend()
plt.xlabel("Initial $\Theta$")
plt.ylabel("$\Delta \Theta$")

ax.set_xlim([0, 1])
#ax.set_ylim([50, 100])

plt.show()

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
"""
iso_0_phase = loadtxt("./data/AR2/output/iso_0/AR2_free_phase_pulse=0.025.txt")
iso_0_PRC = loadtxt("./data/AR2/output/iso_0/AR2_PRC_pulse=0.025.txt")
p0 =  0.002729897365001017

for i in range(len(iso_0_PRC)):
    if iso_0_PRC[i]>pi: iso_0_PRC[i] += -2*pi
    if iso_0_PRC[i]<-pi: iso_0_PRC[i] += 2*pi
    
iso_0_phase = iso_0_phase/(2*pi)
iso_0_PRC = iso_0_PRC/(2*pi)

#plt.plot(iso_0_phase, iso_0_PRC, "k+")

iso_001_phase = loadtxt("./data/AR2/output/iso_001/AR2_free_phase_pulse=0.025.txt")
iso_001_PRC = loadtxt("./data/AR2/output/iso_001/AR2_PRC_pulse=0.025.txt")
p1 = 5.560350945980166e-05

for i in range(len(iso_001_PRC)):
    if iso_001_PRC[i]>pi: iso_001_PRC[i] += -2*pi
    if iso_001_PRC[i]<-pi: iso_001_PRC[i] += 2*pi

#plt.plot(iso_001_phase, iso_001_PRC, "b+")
iso_001_phase = iso_001_phase/(2*pi)
iso_001_PRC = iso_001_PRC/(2*pi)

iso_002_phase = loadtxt("./data/AR2/output/iso_002/AR2_free_phase_pulse=0.025.txt")
iso_002_PRC = loadtxt("./data/AR2/output/iso_002/AR2_PRC_pulse=0.025.txt")
p2 = 1.1444825053683138e-06

for i in range(len(iso_002_PRC)):
    if iso_002_PRC[i]>pi: iso_002_PRC[i] += -2*pi
    if iso_002_PRC[i]<-pi: iso_002_PRC[i] += 2*pi

iso_002_phase = iso_002_phase/(2*pi)
iso_002_PRC = iso_002_PRC/(2*pi)
#plt.plot(iso_002_phase, iso_002_PRC, "g+")

iso_0005_phase = loadtxt("./data/AR2/output/iso_0005/AR2_free_phase_pulse=0.025.txt")
iso_0005_PRC = loadtxt("./data/AR2/output/iso_0005/AR2_PRC_pulse=0.025.txt")
p3 = 0.0003883752095350655

for i in range(len(iso_0005_PRC)):
    if iso_0005_PRC[i]>pi: iso_0005_PRC[i] += -2*pi
    if iso_0005_PRC[i]<-pi: iso_0005_PRC[i] += 2*pi

iso_0005_phase = iso_0005_phase/(2*pi)
iso_0005_PRC = iso_0005_PRC/(2*pi)
#plt.plot(iso_0005_phase, iso_0005_PRC, "r+")

iso_0001_phase = loadtxt("./data/AR2/output/iso_-0001/AR2_free_phase_pulse=0.025.txt")
iso_0001_PRC = loadtxt("./data/AR2/output/iso_-0001/AR2_PRC_pulse=0.025.txt")
p4 = 0.0040331849153021895

for i in range(len(iso_0001_PRC)):
    if iso_0001_PRC[i]>pi: iso_0001_PRC[i] += -2*pi
    if iso_0001_PRC[i]<-pi: iso_0001_PRC[i] += 2*pi

iso_0001_phase = iso_0001_phase/(2*pi)
iso_0001_PRC = iso_0001_PRC/(2*pi)
#plt.plot(iso_0001_phase, iso_0001_PRC, "c+")

iso_0002_phase = loadtxt("./data/AR2/output/iso_-0002/AR2_free_phase_pulse=0.025.txt")
iso_0002_PRC = loadtxt("./data/AR2/output/iso_-0002/AR2_PRC_pulse=0.025.txt")
p5 = 0.005969547947752948

for i in range(len(iso_0002_PRC)):
    if iso_0002_PRC[i]>pi: iso_0002_PRC[i] += -2*pi
    if iso_0002_PRC[i]<-pi: iso_0002_PRC[i] += 2*pi

iso_0002_phase = iso_0002_phase/(2*pi)
iso_0002_PRC = iso_0002_PRC/(2*pi)
#plt.plot(iso_0002_phase, iso_0002_PRC, "m+")

iso_00024_phase = loadtxt("./data/AR2/output/iso_-00024/AR2_free_phase_pulse=0.025.txt")
iso_00024_PRC = loadtxt("./data/AR2/output/iso_-00024/AR2_PRC_pulse=0.025.txt")
p6 = 0.0069724944761951

for i in range(len(iso_00024_PRC)):
    if iso_00024_PRC[i]>pi: iso_00024_PRC[i] += -2*pi
    if iso_00024_PRC[i]<-pi: iso_00024_PRC[i] += 2*pi

iso_00024_phase = iso_00024_phase/(2*pi)
iso_00024_PRC = iso_00024_PRC/(2*pi)
#plt.plot(iso_00024_phase, iso_00024_PRC, "y+")

#x = np.array(np.array[iso_00024_phase, iso_0002_phase, iso_0001_phase, iso_0_phase, iso_0005_phase, iso_001_phase, iso_002_phase])
#y = np.array(np.array[iso_00024_PRC, iso_0002_PRC, iso_0001_PRC, iso_0_PRC, iso_0005_PRC, iso_001_PRC, iso_002_PRC])
#z = np.array(np.array[-0.00024, -0.0002, -0.0001, 0.0, 0.0005, 0.001, 0.002])
fig = plt.figure()
ax = plt.axes(projection ='3d')

x = iso_00024_phase
y = []
for i in range(len(x)):
    y.append(-0.00024)
sigma_00024 = np.array(y)
z = iso_00024_PRC
ax.plot3D(x, y, z, 'g+')

x = iso_0002_phase
y = []
for i in range(len(x)):
    y.append(-0.0002)
sigma_0002 = np.array(y)
z = iso_0002_PRC
ax.plot3D(x, y, z, 'b+')

x = iso_0001_phase
y = []
for i in range(len(x)):
    y.append(-0.0001)
sigma_0001 = np.array(y)
z = iso_0001_PRC
ax.plot3D(x, y, z, 'r+')

x = iso_0_phase
y = []
for i in range(len(x)):
    y.append(0.)
sigma_0 = np.array(y)
z = iso_0_PRC
ax.plot3D(x, y, z, 'k+')

x = iso_0005_phase
y = []
for i in range(len(x)):
    y.append(0.0005)
sigma_0005 = np.array(y)
z = iso_0005_PRC
ax.plot3D(x, y, z, 'm+')

x = iso_001_phase
y = []
for i in range(len(x)):
    y.append(0.001)
sigma_001 = np.array(y)
z = iso_001_PRC
ax.plot3D(x, y, z, 'y+')


x = iso_002_phase
y = []
for i in range(len(x)):
    y.append(0.002)
sigma_002 = np.array(y)
z = iso_002_PRC
ax.plot3D(x, y, z, 'c+')

plt.title("PRCs according to sigma", fontsize=30)
#ax.set_zticks([0, 0.002])
ax.set_xlabel('$Initial \Theta$', fontsize=30)
ax.set_ylabel("$\sigma$", fontsize=30,)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$\Delta \Theta$', fontsize=30, rotation = 0)

fig.set_size_inches(w=13,h=13)
plt.show()

all_phases = np.concatenate((iso_00024_phase, iso_0002_phase, iso_0001_phase,\
                    iso_0_phase, iso_0005_phase, iso_001_phase, iso_002_phase))
all_PRCs = np.concatenate((iso_00024_PRC, iso_0002_PRC, iso_0001_PRC,\
                    iso_0_PRC, iso_0005_PRC, iso_001_PRC, iso_002_PRC))
all_sigmas = np.concatenate((sigma_00024, sigma_0002, sigma_0001,\
                    sigma_0, sigma_0005, sigma_001, sigma_002))

ptot = p0 + p1 + p2 + p3 + p4 + p5 + p6
p0 = p0/ptot; p1 = p1/ptot; p2 = p2/ptot; p3 = p3/ptot; p4 = p4/ptot; p5 = p5/ptot; p6 = p6/ptot
ptot = p0 + p1 + p2 + p3 + p4 + p5 + p6
print(ptot)

"""
phase = p0*iso_0_phase + p1*iso_001_phase + p2*iso_002_phase + p3*iso_0005_phase + p4*iso_0001_phase + p5*iso_0002_phase + p6*iso_00024_phase
PRC_weighted = p0*iso_0_PRC + p1*iso_001_PRC + p2*iso_002_PRC + p3*iso_0005_PRC + p4*iso_0001_PRC + p5*iso_0002_PRC + p6*iso_00024_PRC

plt.plot(phase, PRC_weighted, "k+")
plt.title("Weighted PRC of the AR2 model")
#plt.legend(["Initial limit cycle", "Iso 0.001", "Iso 0.002", "Iso 0.0005", "Iso -0.0001", "Iso -0.0002", "Iso -0.00024"])
plt.xlabel("Initial $\Theta$")
plt.ylabel("Phase shift")
plt.show()
"""
phase = np.linspace(0, 1, 100)

#plt.plot(iso_0_phase, iso_0_PRC, "k+")
fit_0 = Polynomial.fit(iso_0_phase, iso_0_PRC, deg=7)
"""
plt.plot(phase, fit_0(phase), "b-")
plt.title("Limit cycle: Type 1 PRC and fit")
plt.legend(["PRC", "Fit"])
plt.show()
"""
#plt.plot(iso_001_phase, iso_001_PRC, "b+")
fit_1 = Polynomial.fit(iso_001_phase, iso_001_PRC, deg=7)
#plt.plot(phase, fit_1(phase), "r-")
#plt.show()

#plt.plot(iso_002_phase, iso_002_PRC, "g+")
fit_2 = Polynomial.fit(iso_002_phase, iso_002_PRC, deg=7)
#plt.plot(phase, fit_2(phase), "r-")
#plt.show()

#plt.plot(iso_0005_phase, iso_0005_PRC, "r+")
fit_3 = Polynomial.fit(iso_0005_phase, iso_0005_PRC, deg=7)
#plt.plot(phase, fit_3(phase), "r-")
#plt.show()


fit_4 = Polynomial.fit(iso_0001_phase, iso_0001_PRC, deg=7)
#plt.plot(iso_0001_phase, iso_0001_PRC, "c+")
#plt.plot(phase, fit_4(phase), "r-")
#plt.title("Iso -0.0001: Type 0 PRC and fit")
#plt.legend(["PRC", "Fit"])
#plt.show()

#plt.plot(iso_0002_phase[1:15], iso_0002_PRC[1:15], "m+")
fit_5_1 = Polynomial.fit(iso_0002_phase[1:15], iso_0002_PRC[1:15], deg=6)

#plt.plot(iso_0002_phase[16:], iso_0002_PRC[16:], "m+")

#plt.plot(iso_0002_phase, iso_0002_PRC, "b+")

phase_002 = iso_0002_phase[15:]; np.append(phase_002, 0)
PRC_002 = iso_0002_PRC[15:]; np.append(PRC_002, 0)

#plt.plot(phase_002, PRC_002, "g+")
#plt.plot(phase, fit_5_2(phase), "r-")

fit_5_2 = Polynomial.fit(phase_002, PRC_002, deg=9)


def fit_5(phase):
    fit_5 = np.zeros(len(phase))
    for i in range(len(phase)):
        if phase[i]<0.51: fit_5[i]= fit_5_2(phase[i])
        if phase[i]>0.51: fit_5[i]= fit_5_1(phase[i])
    return fit_5

#plt.plot(phase, fit_5(phase), "r-")

#plt.xlim((0, 1))
#plt.ylim((-0.5, 0.5))
#plt.title("Iso-0.0002: Type 0 PRC and fit")
#plt.legend(["PRC", "Fit"])
#plt.show()

#plt.plot(iso_0002_phase, iso_0002_PRC, "m+")
#plt.plot(phase, fit_5(phase), "r-")
#plt.xlim((0, 1))
#plt.ylim((-0.5, 0.5))
#plt.title("Limit cycle: Type 0 PRC and fit")
#plt.legend(["PRC", "Fit 1", "Fit 2"])
#plt.show()

fit_6_1 = Polynomial.fit(iso_00024_phase[1:8], iso_00024_PRC[1:8], deg=6)
fit_6_2 = Polynomial.fit(iso_00024_phase[8:], iso_00024_PRC[8:], deg=6)

def fit_6(phase):
    fit_6 = np.zeros(len(phase))
    for i in range(len(phase)):
        if phase[i]<0.51: fit_6[i]= fit_6_2(phase[i])
        if phase[i]>0.51: fit_6[i]= fit_6_1(phase[i])
    return fit_6

#plt.plot(iso_00024_phase, iso_00024_PRC, "b+")
#plt.plot(phase, fit_6(phase), "r-")
#plt.title("Iso -0.00024: Type 0 PRC and fit")
#plt.legend(["PRC", "Fit"])
#plt.xlim((0, 1))
#plt.ylim((-0.5, 0.5))
#plt.show()

fit = p0*fit_0(phase) + p1*fit_1(phase) + p2*fit_2(phase) + p3*fit_3(phase) + p4*fit_4(phase) + p5*fit_5(phase) + p6*fit_6(phase)

plt.plot(phase, fit, "k+")
plt.title("Weighted PRC of the AR2 model")
#plt.legend(["Initial limit cycle", "Iso 0.001", "Iso 0.002", "Iso 0.0005", "Iso -0.0001", "Iso -0.0002", "Iso -0.00024"])
plt.xlabel("Initial $\Theta$")
plt.ylabel("Phase shift")

plt.show()
