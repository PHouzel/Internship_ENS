#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:32:08 2021

@author: pierrehouzelstein
"""
from numpy import loadtxt, pi, sqrt, cos, sin, arctan, zeros, linspace, mod, arctan2
from scipy import interpolate
import matplotlib.pyplot as plt

def Hopf_theoretical(phase, beta):
    phase = phase/(2*pi)
    
    a = -sin(2*pi*phase)
    b = 2*pi*sqrt(abs(beta))
    Hopf = a/b
    return Hopf

def omega(phase, m):
    a = m*sin(pi*phase)
    b = sqrt(m**2 -1)*cos(pi*phase) + sin(pi*phase)
    omega = 2*arctan(a/b)
    return omega

def SNIC_theoretical(phase, beta, m):
    phase = phase/(2*pi)
    
    om = omega(phase, m)
    a = sqrt(m**2 - 1)*sin(om)
    b = 2*pi*sqrt(beta)*(m - sin(om))
    SNIC = -a/b
    return SNIC

yp = 1.5; ym = -1.5
xp = 1.5; xm = -1.5
y = linspace(ym, yp, 100)
x = linspace(xm, xp, 100)
isochrones_real = loadtxt('./Data/hopf/above/data/realValuesD0.01125')
isochrones_im = loadtxt('./Data/hopf/above/data/imagValuesD0.01125')
#interpolate
real_isochrone_func = interpolate.interp2d(x, y, isochrones_real, kind = 'cubic')
im_isochrone_func = interpolate.interp2d(x, y, isochrones_im, kind = 'cubic')

#case 1: Hopf above bifurcation
beta = 1;pulse = 0.3
scale = 2*pi*pulse

phase_0 = loadtxt('./Data/hopf/output/Hopf_LC_phase_0T0.3.txt')
PRC_0 = loadtxt('./Data/hopf/output/Hopf_LC_shift_0T0.3.txt')
phase = loadtxt('./Data/hopf/output/Hopf_LC_phase_1T0.3.txt')
PRC = loadtxt('./Data/hopf/output/Hopf_LC_shift_1T0.3.txt')

print(phase)

plt.figure(figsize=(9, 11))
plt.plot(phase, scale*Hopf_theoretical(phase, beta), "g-")
plt.plot(phase_0, -PRC_0, "r-")
plt.plot(phase, -PRC, "b+")
plt.title("Hopf LC (above bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Theoretical PRC", "PRC at time 0", "PRC at T"])
#plt.xlim([0, 1])
#plt.ylim([-0.25, 0.25])
plt.savefig("./Data/graphs/Hopf_LC_PRC.jpg")
plt.show()

#case 2: hopf focus
beta = -0.1
scale = 0.6

phase_0 = loadtxt('./Data/hopf/output/Hopf_focus_phase_0T0.3.txt')
PRC_0 = loadtxt('./Data/hopf/output/Hopf_focus_shift_0T0.3.txt')
phase = loadtxt('./Data/hopf/output/Hopf_focus_phase_1T0.3.txt')
PRC = loadtxt('./Data/hopf/output/Hopf_focus_shift_1T0.3.txt')

plt.figure(figsize=(9, 11))
plt.plot(phase, scale*Hopf_theoretical(phase, beta), "g-")
plt.plot(phase_0, PRC_0, "r-")
plt.plot(phase, PRC, "b+")
plt.title("Hopf focus (below bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Theoretical PRC", "PRC at time 0", "PRC at T"])
#plt.ylim([-0.1, 0.05])
plt.savefig("./Data/graphs/Hopf_focus_PRC.jpg")
plt.show()

#Case 3: SNIC lc
beta = 1; m = 1.1; scale = 1.

polar_phase_0 = loadtxt('./Data/snic/output/SNIC_LC_phase_0T0.3.txt')
PRC_0 = loadtxt('./Data/snic/output/SNIC_LC_shift_0T0.3.txt')
polar_phase = loadtxt('./Data/snic/output/SNIC_LC_phase_1T0.3.txt')
PRC = loadtxt('./Data/snic/output/SNIC_LC_shift_1T0.3.txt')

pulse = 0.3

plt.figure(figsize=(9, 11))
plt.plot(polar_phase, 2*pi*pulse*SNIC_theoretical(polar_phase, beta, m), "g-")
plt.plot(polar_phase_0, PRC_0, "r-")
plt.plot(polar_phase, PRC, "b+")
plt.title("SNIC LC (below bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Theoretical PRC", "PRC at time 0", "PRC at T"])
#plt.xlim([0, 1])
#plt.ylim([-2, 0.1])
plt.savefig("./Data/graphs/SNIC_LC_PRC.jpg")
plt.show()

polar_phase_0 = loadtxt('./Output_files/phase_0T.txt')
PRC_0 = loadtxt('./Output_files/shift_0T.txt')
polar_phase = loadtxt('./Output_files/phase_1T.txt')
PRC = loadtxt('./Output_files/shift_1T.txt')

#plt.figure(figsize=(9, 11))
#plt.plot(polar_phase, 2*pi*pulse*SNIC_theoretical(polar_phase, beta, m), "g-")
plt.plot(polar_phase_0[9:], PRC_0[9:], "r--")
plt.plot(polar_phase, PRC, "b+")
plt.title("AR2 model: PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["PRC at time 0", "PRC at T"])
#plt.xlim([0, 1])
#plt.ylim([-2, 0.1])
plt.savefig("./Output_files/AR2_PRC_1T.jpg")
plt.show()

polar_phase_0 = loadtxt('./Output_files/phase_0T.txt')
PRC_0 = loadtxt('./Output_files/shift_0T.txt')
polar_phase = loadtxt('./Output_files/phase_2T.txt')
PRC = loadtxt('./Output_files/shift_2T.txt')

#plt.figure(figsize=(9, 11))
#plt.plot(polar_phase, 2*pi*pulse*SNIC_theoretical(polar_phase, beta, m), "g-")
plt.plot(polar_phase_0[9:], PRC_0[9:], "r--")
plt.plot(polar_phase, PRC, "b+")
plt.title("AR2 model: PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["PRC at time 0", "PRC at 2T"])
#plt.xlim([0, 1])
#plt.ylim([-2, 0.1])
plt.savefig("./Output_files/AR2_PRC_2T.jpg")
plt.show()

"""
#SNIC SN
phase_0 = loadtxt('./Data/snic/output/SNIC_SN_phase_0T.txt')
PRC_0 = loadtxt('./Data/snic/output/SNIC_SN_shift_0T.txt')
phase = loadtxt('./Data/snic/output/snic_SN_phase_1T.txt')
PRC = loadtxt('./Data/snic/output/snic_SN_shift_1T.txt')

plt.figure(figsize=(9, 11))
plt.plot(phase_0, PRC_0, "r-")
plt.plot(phase, PRC, "b+")
plt.title("SNIC SN (below bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Reference PRC", "PRC at T"])
#plt.xlim([0, 1])
#plt.ylim([-0.1, 0.1])
plt.savefig("./Data/graphs/SNIC_SN_PRC.jpg")
plt.show()
"""