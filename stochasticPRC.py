#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:29:08 2021
Euler integration routine for stochastic diff equation
"""

from numpy import linspace, array, meshgrid, zeros, ndarray
from numpy import sin, cos, pi, random
from datetime import datetime
import matplotlib.pyplot as plt
from numba import jit, float64, int64, types
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

@jit((float64[:])(float64[:], float64, float64),nopython=True)
def stuartLandau(z, a, alpha):
    """
    Studied function
    """
    x, y = z
    fx = alpha*x*(1 - (x**2 + y**2)) - y*(1 + alpha*a*(x**2 + y**2))
    fy = alpha*y*(1 - (x**2 + y**2)) + x*(1 + alpha*a*(x**2 + y**2))
    return array([fx, fy])

@jit((float64[:])(float64[:]),nopython=True)
def noiseFunction(z):
    """
    Added noise
    """
    x, y = z
    gx = 1.
    gy = 1.
    return array([gx, gy])

def vector_space(a, alpha):
    """
    Plots the vector field linked to the system
    """
    x, y = meshgrid(linspace(-2, 2, num=20), linspace(-2, 2, num=20))
    fx, fy =  stuartLandau([x, y], a, alpha)
    gx, gy = noiseFunction([x, y])
    plt.quiver(x, y, fx + gx, fy + gy, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

@jit((float64[:])(float64[:], float64, float64, float64, float64),nopython=True)
def predictor(z, h, eta, a, alpha):
    """
    Computes predictor for Euler integration
    """
    pred = z + h*stuartLandau(z, a, alpha) + eta*noiseFunction(z)
    return pred

@jit((float64[:])(float64[:], float64, float64, float64, float64),nopython=True)
def update_z(z, h, eta, a, alpha):
    """
    Gives z at t + h from z at t
    """
    pred = predictor(z, h, eta, a, alpha)
    z_updated = z + 0.5*(stuartLandau(z, a, alpha) + stuartLandau(pred, a, alpha))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64, float64, float64, int64, float64),nopython=True)
def EulerInteg(h, a, alpha, numsteps, phase):
    """
    Integration routine
    """
    #Initial points
    time = 0
    z = array([cos(2*pi*phase), sin(2*pi*phase)])

    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2))
    time_list[0] = time
    trajectory[0] = z

    for i in range(numsteps):
        eta = random.normal(loc=0, scale=h, size=1)[0]

        z_updated = update_z(z, h, eta, a, alpha)

        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z

    return time_list, trajectory


@jit(types.Tuple((float64[:], float64[:,:]))(float64, float64, float64, int64, int64),nopython=True)
def stochastic_average(h, a, alpha, numsteps, numiter):
    """
    The main piece
    Repeats various integrations and uses them to average the position of the point at each t
    """
    #Same times for all trajectories
    times = zeros(numsteps+1)
    trajectories = zeros((numiter, numsteps+1, 2))

    #Initial phase
    phase = random.uniform(0, 1)
    for i in range(numiter):
        times, trajectories[i] = EulerInteg(h, a, alpha, numsteps, phase)

    stochastic_av = zeros((numsteps+1, 2))
    for j in range(numsteps+1):
        average_pos = array([0., 0.])
        for k in range(numiter):
            average_pos = average_pos + trajectories[k][j]
        stochastic_av[j] = average_pos/numiter

    return times, stochastic_av

def main():
    
    startTime = datetime.now()

    h=0.01; a, alpha = [0, 1]; numsteps = 5000; numiter = 50000

    #Example of integration
    time_list, trajectory = EulerInteg(h, a, alpha, numsteps, random.uniform(0, 1))
    plt.plot(trajectory[:,0], trajectory[:,1], color='green')
    plt.title("Typical trajectory with noise")
    vector_space(a, alpha)

    plt.plot(time_list, trajectory[:,0], color='green')
    plt.title("Typical oscillations with noise")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    times, stochastic_av = stochastic_average(h, a, alpha, numsteps, numiter)

    plt.plot(stochastic_av[:,0], stochastic_av[:,1], color='blue')
    plt.title("Average trajectory with noise")
    vector_space(a, alpha)

    plt.plot(times, stochastic_av[:,1], color='blue')
    plt.title("Average oscillations with noise")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()
