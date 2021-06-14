#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:32:08 2021

@author: pierrehouzelstein
"""

from numpy import loadtxt, pi, sqrt, cos, sin, arctan, zeros, linspace, mod, arctan2
import matplotlib.pyplot as plt

#figure 4: plot weak stim PRCs of alveus and dentate


free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_free_phase_S=1.6.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowan_PRC_S=1.6.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/alveus/WilsonCowen_variance_theo_S=1.6.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi
    
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.tick_params("both", direction = "in")

ax.plot(free_phase/(2*pi), PRC/(2*pi), "r-", label="_Hidden_1")
ax.plot(free_phase/(2*pi), PRC/(2*pi), "r.", label = "Alveus", markersize=10)
ax.fill_between(free_phase/(2*pi), (PRC+sqrt(variance))/(2*pi), (PRC-sqrt(variance))/(2*pi), color="orangered", alpha=.4)

free_phase = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_free_phase_S=2.2.txt")
PRC = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowan_PRC_S=2.2.txt")
variance = loadtxt("./data/WilsonCowan/output/comparison_figure_4/dentate/WilsonCowen_variance_theo_S=2.2.txt")

for i in range(len(PRC)):
    if PRC[i]>pi: PRC[i] += -2*pi
    if PRC[i]<-pi: PRC[i] += 2*pi

plt.plot(free_phase/(2*pi), PRC/(2*pi), "b-", label = "_Hidden_2")
plt.plot(free_phase/(2*pi), PRC/(2*pi), "b.", label = "Dentate", markersize=10)
plt.fill_between(free_phase/(2*pi), (PRC+sqrt(variance))/(2*pi), (PRC-sqrt(variance))/(2*pi), color="blue", alpha=.4)

ax.plot(free_phase/(2*pi), PRC/(2*pi), "b-", label="_Hidden_2")
ax.plot(free_phase/(2*pi), PRC/(2*pi), "b.", label = "Dentate", markersize=10)
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

