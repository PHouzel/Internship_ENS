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
#size of the initial shift in position
shift = 1.0
beta=1

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

#Plotting the trajectories
t = np.linspace(0,20)
s0_list = []
for i in range(number_of_points):
    #setting random initial points
    s0_list.append([random.randint(-2, 2), random.randint(-2, 2)])
    s = odeint(f,s0_list[i],t)
    plt.plot(s[:,0], s[:,1], color='blue')
    plt.title("Phase space of the toy model and trajectories with beta = 1")
vector_space(beta=1)

"""
#shift initial points
s_list = []
s_shifted_list = []
for j in range(len(s0_list)):
    s_list.append(odeint(f,s0_list[j],t))
    s_shifted_list.append(odeint(f,[s0_list[j][0], s0_list[j][1] + shift],t))
    
"""
"""
    
#Plot one case to see the shift
plt.plot(t,s_list[0][:,1],'b-')
plt.plot(t,s_shifted_list[0][:,1],'g-')
plt.xlabel("t")
plt.ylabel("x2")
plt.legend(["x2", "x2_shifted"])
plt.title("Example of a phase shift in the oscillations of x2 when shifting the initial position by 1.0 vertically")
plt.show()
"""
"""
#extract the shift
time_shift_list = []
for k in range(len(s_list)):
    """
    #x2_max = np.max(s_list[k][:,1])
    #x2_shifted_max = np.max(s_shifted_list[k][:,1])
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
"""

def limit_cycle(s, t):
    #Plots theoretical limit cycle; take initial position such that dx/dt and dy/dt = 0 and (x,y) =/=0
    alpha = 1
    a = 1
    x=s[0]
    y=s[1]
    dxdt = alpha*x*(1 - (x**2 + y**2)) - y*(1 + alpha*a*(x**2 + y**2))
    dydt = alpha*y*(1 - (x**2 + y**2)) + x*(1 + alpha*a*(x**2 + y**2))
    return [dxdt, dydt]

def shift_cycle_points(s, shift):
    #Shifts the points belonging to the limit cycle
    cycle_point_shifted = []
    for m in range(len(s)):
        cycle_point_shifted.append([s[m][0]+shift[0], s[m][1]+shift[1]])
    cycle_point_shifted = np.array(cycle_point_shifted)
    plt.plot(cycle_point_shifted[:,0], cycle_point_shifted[:,1], color='green')
    
    return cycle_point_shifted
    
s0 = [1, 0]
s = odeint(limit_cycle, s0, t)
initial_s = s
plt.plot(s[:,0], s[:,1], color='blue')
cycle_point_shifted = shift_cycle_points(s, shift=[0, 1])
plt.title("Theoretical shape of the limit cycle")
plt.legend(["Limit cycle", "Shifted points"])
vector_space(beta=1)

#Trajectories of initial points
trajectory_list = []
for i in range(len(initial_s)):
    trajectory = odeint(f,initial_s[i],t)
    trajectory_list.append(trajectory)
    plt.plot(initial_s[:,0], initial_s[:,1], color='blue')
plt.title("Trajectory of the initial points")
vector_space(beta=1)

trajectory_list = np.array(trajectory_list)

cycle_point_shifted = shift_cycle_points(s, shift=[0, 1])
#Trajectories of the shifted points
#Use those shifted points as initial positions
shifted_trajectory_list = []
for i in range(len(cycle_point_shifted)):
    shifted_trajectory = odeint(f,cycle_point_shifted[i],t)
    shifted_trajectory_list.append(shifted_trajectory)
    plt.plot(shifted_trajectory[:,0], shifted_trajectory[:,1], color='blue')
plt.title("Trajectory of the shifted points")
vector_space(beta=1)

shifted_trajectory_list = np.array(shifted_trajectory_list)

plt.plot(t,trajectory_list[0][:,1],'b-')
plt.plot(t,shifted_trajectory_list[0][:,1],'g-')
plt.xlabel("t")
plt.ylabel("x2")
plt.legend(["x2", "x2_shifted"])
plt.title("Example of a phase shift in the oscillations of x2 when shifting the initial position by 1.0 vertically")
plt.show()