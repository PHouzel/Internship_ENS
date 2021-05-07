#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:32:08 2021

@author: pierrehouzelstein
"""
from numpy import loadtxt
import matplotlib.pyplot as plt
"""
#AR2 model
phase_0 = loadtxt('./Output_files/phase_0T.txt')
PRC_0 = loadtxt('./Output_files/shift_0T.txt')
phase = loadtxt('./Output_files/phase_1T.txt')
PRC = loadtxt('./Output_files/shift_1T.txt')
"""

# Hopf LC
phase_0 = loadtxt('./Data/hopf/output/Hopf_LC_phase_0T.txt')
PRC_0 = loadtxt('./Data/hopf/output/Hopf_LC_shift_0T.txt')
phase = loadtxt('./Data/hopf/output/Hopf_LC_phase_1T.txt')
PRC = loadtxt('./Data/hopf/output/Hopf_LC_shift_1T.txt')

plt.figure(figsize=(9, 11))
plt.plot(phase_0, PRC_0, "r-")
plt.plot(phase, PRC, "b+")
plt.title("Hopf LC (above bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Reference PRC", "PRC at T"])
#plt.xlim([0, 1])
#plt.ylim([-0.1, 0.05])
plt.savefig("./Data/graphs/Hopf_LC_PRC.jpg")
plt.show()

#hopf focus
phase_0 = loadtxt('./Data/hopf/output/Hopf_focus_phase_0T.txt')
PRC_0 = loadtxt('./Data/hopf/output/Hopf_focus_shift_0T.txt')
phase = loadtxt('./Data/hopf/output/Hopf_focus_phase_1T.txt')
PRC = loadtxt('./Data/hopf/output/Hopf_focus_shift_1T.txt')

plt.figure(figsize=(9, 11))
plt.plot(phase, PRC, "b+")
plt.plot(phase_0, PRC_0, "r-")
plt.title("Hopf focus (below bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Reference PRC", "PRC at T"])
#plt.ylim([-0.1, 0.05])
plt.savefig("./Data/graphs/Hopf_focus_PRC.jpg")
plt.show()

#SNIC lc
phase_0 = loadtxt('./Data/snic/output/SNIC_LC_phase_0T.txt')
PRC_0 = loadtxt('./Data/snic/output/SNIC_LC_shift_0T.txt')
phase = loadtxt('./Data/snic/output/SNIC_LC_phase_1T.txt')
PRC = loadtxt('./Data/snic/output/SNIC_LC_shift_1T.txt')

plt.figure(figsize=(9, 11))
plt.plot(phase_0, PRC_0, "r-")
plt.plot(phase, PRC, "b+")
plt.title("SNIC LC (below bifurcation): PRC with an initial E-shift of 0.1, after one period")
plt.xlabel("$\Theta$")
plt.ylabel("$\Delta \Theta$")
plt.legend(["Reference PRC", "PRC at T"])
plt.xlim([0, 1])
#plt.ylim([-0.1, 0.1])
plt.savefig("./Data/graphs/SNIC_LC_PRC.jpg")
plt.show()

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
plt.xlim([0, 1])
#plt.ylim([-0.1, 0.1])
plt.savefig("./Data/graphs/SNIC_SN_PRC.jpg")
plt.show()