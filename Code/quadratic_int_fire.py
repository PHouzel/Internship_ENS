#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:00:25 2021

@author: pierrehouzelstein

Goal: find reliable method to compute PRC from data
Model: quadratic integrate-and-fire neuron
Izhikevich, p.305, fig. 8.31
Izhikevich, p.307, fig. 8.32
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from numba import jit, float64, int64, types
from numpy import  meshgrid, shape, array, sqrt, zeros, random, pi, linspace, cos, sin, loadtxt, arctan, arctan2, mod, exp, sum, square
from math import atan2
from scipy import interpolate
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def quadratic_int_fire(z, I, t):

    V, u = z
    
    if V <= -65: b =10
    else: b = 2
    
    dV = (0.25*(V + 65)*(V + 45)+ I(t) - u)/40
    du = 0.015*(b*(V + 65) - u)
    return array([dV, du])


def noiseFunction(eta, amplitude):
    """
    Added noise amplitude
    """
    gx = (amplitude)*eta[0]
    gy = (amplitude)*eta[1]
    return array([gx, gy])

def update_z(z, I, t, h, eta, amplitude):
    """
    Gives z at t + h from z at t
    """
    V, u = z
    #Hard reset when treshold
    if V >= 0: 
        V_updated = -55
        u_updated = u + 50
        z_updated = array([V_updated, u_updated])
    #Normal integration otherwise
    else:
        pred = z + h*quadratic_int_fire(z, I, t) + noiseFunction(eta, amplitude)
        
        z_updated = z + 0.5*(quadratic_int_fire(z, I, t) + quadratic_int_fire(pred, I, t))*h\
                         + noiseFunction(eta, amplitude)
    return z_updated

def EulerInteg(z, I, initial_t, h, amplitude, numsteps):
    """
    Integration routine
    """
    #Initial time
    time = initial_t
    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2))
    time_list[0] = time
    trajectory[0] = z
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(h), size=2)

        z = update_z(z, I, time, h, eta, amplitude)
        time += h
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z

    return time_list, trajectory

def stimulus(time):
    #if time < 0: stimulus = 0
    #elif time > 200: stimulus = 0
    #else: stimulus = 110
    stimulus = 0.1*time
    return stimulus

z0 = array([-50, 60]); h = 0.01; amplitude = 0.; numsteps = 100000

time_list, trajectory = EulerInteg(z0, stimulus, 0, h, amplitude, numsteps)


plt.plot(time_list, trajectory[:,0], "b-")
plt.title("Quadratic integrate-and-fire neuron")
plt.legend(["V"])
plt.show()

plt.plot(time_list, trajectory[:,1], "r-")
plt.title("Quadratic integrate-and-fire neuron")
plt.legend(["u"])
plt.show()

plt.plot(trajectory[:,0], trajectory[:,1], "r-", label="_Hidden_1")
plt.plot(trajectory[:,0], trajectory[:,1], "b+")
plt.plot(trajectory[:,0][0], trajectory[:,1][0], "ko")
plt.title("Phase space")
plt.legend(["", "Trajectory", "Starting point"])
plt.show()