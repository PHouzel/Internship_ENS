#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:09:43 2021

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

phase = np.linspace(0, 1, 100)

plt.plot(iso_0_phase, iso_0_PRC, "k+")
fit_0 = Polynomial.fit(iso_0_phase, iso_0_PRC, deg=7)
plt.plot(phase, fit_0(phase), "b-")
plt.title("Limit cycle: Type 1 PRC and fit")
plt.legend(["PRC", "Fit"])
plt.show()

plt.plot(iso_001_phase, iso_001_PRC, "b+")
fit_1 = Polynomial.fit(iso_001_phase, iso_001_PRC, deg=7)
plt.plot(phase, fit_1(phase), "r-")
plt.show()

plt.plot(iso_002_phase, iso_002_PRC, "g+")
fit_2 = Polynomial.fit(iso_002_phase, iso_002_PRC, deg=7)
plt.plot(phase, fit_2(phase), "r-")
plt.show()

plt.plot(iso_0005_phase, iso_0005_PRC, "r+")
fit_3 = Polynomial.fit(iso_0005_phase, iso_0005_PRC, deg=7)
plt.plot(phase, fit_3(phase), "r-")
plt.show()


fit_4 = Polynomial.fit(iso_0001_phase, iso_0001_PRC, deg=7)
plt.plot(iso_0001_phase, iso_0001_PRC, "c+")
plt.plot(phase, fit_4(phase), "r-")
plt.title("Iso -0.0001: Type 0 PRC and fit")
plt.legend(["PRC", "Fit"])
plt.show()

fit_5_1 = Polynomial.fit(iso_0002_phase[1:15], iso_0002_PRC[1:15], deg=6)

phase_002 = iso_0002_phase[15:]; np.append(phase_002, 0)
PRC_002 = iso_0002_PRC[15:]; np.append(PRC_002, 0)
fit_5_2 = Polynomial.fit(phase_002, PRC_002, deg=9)

def fit_5(phase):
    fit_5 = np.zeros(len(phase))
    for i in range(len(phase)):
        if phase[i]<0.51: fit_5[i]= fit_5_2(phase[i])
        if phase[i]>0.51: fit_5[i]= fit_5_1(phase[i])
    return fit_5

plt.plot(iso_0002_phase, iso_0002_PRC, "m+")
plt.plot(phase, fit_5(phase), "r-")
plt.title("Iso 0.0002: Type 0 PRC and fit")
plt.legend(["PRC", "Fit 1"])
plt.show()

fit_6_1 = Polynomial.fit(iso_00024_phase[1:8], iso_00024_PRC[1:8], deg=6)
fit_6_2 = Polynomial.fit(iso_00024_phase[8:], iso_00024_PRC[8:], deg=6)

def fit_6(phase):
    fit_6 = np.zeros(len(phase))
    for i in range(len(phase)):
        if phase[i]<0.51: fit_6[i]= fit_6_2(phase[i])
        if phase[i]>0.51: fit_6[i]= fit_6_1(phase[i])
    return fit_6

plt.plot(iso_00024_phase, iso_00024_PRC, "b+")
plt.plot(phase, fit_6(phase), "r-")
plt.title("Iso -0.00024: Type 0 PRC and fit")
plt.legend(["PRC", "Fit"])
plt.show()

fit = p0*fit_0(phase) + p1*fit_1(phase) + p2*fit_2(phase) + p3*fit_3(phase) + p4*fit_4(phase) + p5*fit_5(phase) + p6*fit_6(phase)

plt.plot(phase, fit, "k+")
plt.title("Weighted PRC of the AR2 model")
#plt.legend(["Initial limit cycle", "Iso 0.001", "Iso 0.002", "Iso 0.0005", "Iso -0.0001", "Iso -0.0002", "Iso -0.00024"])
plt.xlabel("Initial $\Theta$")
plt.ylabel("Phase shift")

plt.show()