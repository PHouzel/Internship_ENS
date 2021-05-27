#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:32:08 2021

@author: pierrehouzelstein
"""
import numpy as np
from numpy import loadtxt, pi, sqrt, cos, sin, arctan, zeros, linspace, mod, arctan2
from scipy import interpolate
import matplotlib.pyplot as plt
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
#figure 4: plot weak stim PRCs of alveus and dentate

free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_free_phase_S=1.6.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_PRC_S=1.6.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowen_variance_S=1.6.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.plot(free_phase/(2*pi), PRC, "k-", label="_Hidden_1")
plt.plot(free_phase/(2*pi), PRC, "r.", label = "Alveus", markersize=10)
plt.fill_between(free_phase/(2*pi), PRC+variance, PRC-variance, color="orangered", alpha=.4)


free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_free_phase_S=2.2.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_PRC_S=2.2.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowen_variance_S=2.2.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
plt.plot(free_phase/(2*pi), PRC, "k-", label = "_Hidden_2")
plt.plot(free_phase/(2*pi), PRC, "b.", label = "Dentate", markersize=10)
plt.fill_between(free_phase/(2*pi), PRC+variance, PRC-variance, color="blue", alpha=.4)

plt.title("Computed PRCs using data from the paper")
plt.legend()
plt.xlabel("Initial $\Theta$")
plt.ylabel("$\Delta \Theta$")

plt.show()

free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_free_phase_S=1.0.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_PRC_S=1.0.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowen_variance_S=1.0.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.plot(free_phase/(2*pi), PRC, "k-", label="_Hidden_1")
plt.plot(free_phase/(2*pi), PRC, "r.", label = "Alveus", markersize=10)
plt.fill_between(free_phase/(2*pi), PRC+variance, PRC-variance, color="orangered", alpha=.4)


free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_free_phase_S=1.0.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_PRC_S=1.0.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowen_variance_S=1.0.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
plt.plot(free_phase/(2*pi), PRC, "k-", label = "_Hidden_2")
plt.plot(free_phase/(2*pi), PRC, "b.", label = "Dentate", markersize=10)
plt.fill_between(free_phase/(2*pi), PRC+variance, PRC-variance, color="blue", alpha=.4)

plt.title("Computed PRCs using data from the paper")
plt.legend()
plt.xlabel("Initial $\Theta$")
plt.ylabel("$\Delta \Theta$")

plt.show()
