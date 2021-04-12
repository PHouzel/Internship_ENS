#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:25:50 2021

@author: pierrehouzelstein

Intoduction to dynamical systems: toy model 
x'1 = beta*x1 - x2 - x1(x1^2 + x2^2)
x'2 = x1 + beta*x2 - x2(x1^2 + x2^2)

Compute limit cycle, phase response curves
"""

import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import random

#Number of points in the phase space whose trjectory we'll study
number_of_points = 50


def vector_space(beta):
    
    x1, x2 = np.meshgrid(np.linspace(-2, 2, num=20), np.linspace(-2, 2, num=20))
    y1 = beta*x1 - x2 - x1*(x1**2 + x2**2)
    y2 = x1 + beta*x2 - x2*(x1**2 + x2**2)
            

    plt.quiver(x1, x2, y1, y2, color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def f(s,t):
    """
    Integration of x1 and x2
    s =[x1[0], x2[0]]: initial conditions
    """
    beta=1
    x1=s[0]
    x2=s[1]
    dx1dt = beta*x1 - x2 - x1*(x1**2 + x2**2)
    dx2dt = x1 + beta*x2 - x2*(x1**2 + x2**2)
    return [dx1dt, dx2dt]

#Plotting the limit cycle
t = np.linspace(0,20)
s0_list = []
for i in range(number_of_points):
    #setting random initial points
    s0_list.append([random.randint(-2, 2), random.randint(-2, 2)])
    s = odeint(f,s0_list[i],t)
    plt.plot(s[:,0], s[:,1], color='blue')
    plt.title("Phase space of the toy model with beta = 1")
vector_space(beta=1)

#shift initial points
s_list = []
s_shifted_list = []
for j in range(len(s0_list)):
    s_list.append(odeint(f,s0_list[j],t))
    s_shifted_list.append(odeint(f,[s0_list[j][0], s0_list[j][1]+1.0],t))
    
#Plot one case to see the shift
plt.plot(t,s_list[0][:,1],'b-')
plt.plot(t,s_shifted_list[0][:,1],'g-')
plt.xlabel("t")
plt.ylabel("x2")
plt.legend(["x2", "x2_shifted"])
plt.title("Example of a phase shift in the oscillations of x2 when shifting the initial position by 1.0 vertically")
plt.show()

#extract the shift
time_shift_list = []
for k in range(len(s_list)):
    """
    x2_max = np.max(s_list[k][:,1])
    x2_shifted_max = np.max(s_shifted_list[k][:,1])
    """
    index_of_x2_max = np.argmax(s_list[k][:,1])
    index_of_x2_shifted_max = np.argmax(s_shifted_list[k][:,1])
    
    theta = t[index_of_x2_max]
    theta_shifted = t[index_of_x2_shifted_max]

    time_shift_list.append(theta_shifted - theta)

#Extract initial x2 positions
initial_x2_list = []
for l in range(len(s0_list)):
    initial_x2_list.append(s0_list[l][1])

plt.scatter(initial_x2_list, time_shift_list)
plt.xlabel("Initial x2")
plt.ylabel("Time shift in the oscillations")
plt.title("Time shift obtained when shifting the original positions by 1.0 vertically")
plt.show()
