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
from scipy import signal
import math

#Number of points in the phase space whose trjectory we'll study
number_of_points = 50
#size of the initial shift in position
shift = 1.0
#parameters in equations
beta=1
alpha = 0.1
a = 10
theta=0
#time range
tmax=100
nsamples = 100
t = np.linspace(0,20, nsamples)
#period
T0 = (2*math.pi)/(1+alpha*a)

def vector_space(beta):
    #Plots the vector field linked to the system
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
s0_list = []
for i in range(number_of_points):
    #setting random initial points
    s0_list.append([random.randint(-2, 2), random.randint(-2, 2)])
    s = odeint(f,s0_list[i],t)
    plt.plot(s[:,0], s[:,1], color='blue')
    plt.title("Phase space of the toy model and trajectories with beta = 1")
vector_space(beta=1)

def limit_cycle(s, t):
    #Plots theoretical limit cycle; take initial position such that dx/dt and dy/dt = 0 and (x,y) =/=0
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
    plt.plot(shifted_trajectory[:,0], shifted_trajectory[:,1], color='green')
plt.title("Trajectory of the shifted points")
vector_space(beta=1)

shifted_trajectory_list = np.array(shifted_trajectory_list)

#Plot a shift example
plt.plot(t,trajectory_list[0][:,1],'b-')
plt.plot(t,shifted_trajectory_list[0][:,1],'g-')
plt.xlabel("t")
plt.ylabel("x2")
plt.legend(["x2", "x2_shifted"])
plt.title("Example of a phase shift in the oscillations of x2 when shifting the initial position by 1.0 vertically")
plt.show()

"""
#Plot alls shifts
for i in range(len(cycle_point_shifted)):
    plt.plot(t,trajectory_list[i][:,1],'b-')
    plt.plot(t,shifted_trajectory_list[i][:,1],'g-')
    plt.xlabel("t")
    plt.ylabel("x2")
    plt.legend(["x2", "x2_shifted"])
    plt.title("Oscillations of x2")
    plt.show()
"""

#Extract the ohase: difference in phase between a reference oscillation and the studied point
dt = np.linspace(-t[-1], t[-1], 2*nsamples-1)
ref_trajectory = trajectory_list[1][:,1]

phase_list = []
for i in range(len(trajectory_list)):
    #cross correlation between the two signals
    corr = signal.correlate(trajectory_list[i][:,1], ref_trajectory)
    #recover the phase: peak of cross correlation array
    recovered_time_shift = dt[corr.argmax()]
    recovered_phase_shift = 2*math.pi*(((0.5 + recovered_time_shift/T0) % 1.0) - 0.5)
    phase_list.append(recovered_phase_shift)

#extract the shift between initial and shifted trajectories
phase_shift_list = []
for i in range(len(trajectory_list)):
    corr = signal.correlate(trajectory_list[i][:,1], shifted_trajectory_list[i][:,1])
    recovered_time_shift = dt[corr.argmax()]
    #Set phase to be between -pi and pi
    recovered_phase_shift = 2*math.pi*(((0.5 + recovered_time_shift/T0) % 1.0) - 0.5)
    phase_shift_list.append(recovered_phase_shift)

plt.scatter(phase_list, phase_shift_list, color='red')
plt.xlabel("Theta")
plt.ylabel("Phase shift")
plt.show()

#theoretical PRF
sigma = np.linspace(0, 4.5)
PRF_list = []
for i in range(len(sigma)):
    PRF = (((1 - 2*alpha*sigma[i])**0.5)/(2*math.pi)) * math.sin((2*math.pi*theta + 0.5*a*math.log(1 - 2*alpha*sigma[i]))) - a*math.cos(2*math.pi*theta + 0.5*a*math.log(1 - 2*alpha*sigma[i]))
    PRF_list.append(PRF)

plt.plot(sigma, PRF_list, color='red')
plt.xlabel("sigma")
plt.ylabel("PRF(0, sigma)")
plt.title("Theoretical PRF with a=10, alpha=0.1, theta=0")
plt.show()