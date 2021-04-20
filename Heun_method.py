#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:29:08 2021
Euler integration routine for stochastic diff equation
"""

from numpy import linspace, array, meshgrid, zeros, ndarray, shape
from numpy import sin, cos, pi, random, average, exp, inf, sqrt, diag
from datetime import datetime
import matplotlib.pyplot as plt
from numba import jit, float64, int64
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

@jit
def stuartLandau(z, a, alpha, mu):
    """
    Studied function
    """
    x, y = z
    fx = alpha*x*(mu - (x**2 + y**2)) - y*(1 + alpha*a*(x**2 + y**2))
    fy = alpha*y*(mu - (x**2 + y**2)) + x*(1 + alpha*a*(x**2 + y**2))
    return array([fx, fy])

def noiseFunction(z):
    """
    Added noise
    """
    x, y = z
    gx = 0.1
    gy = 0.1
    return array([gx, gy])

def vector_space(a, alpha, mu):
    """
    Plots the vector field linked to the system
    """
    x, y = meshgrid(linspace(-2, 2, num=20), linspace(-2, 2, num=20))
    fx, fy =  stuartLandau([x, y], a, alpha, mu)
    gx, gy = noiseFunction([x, y])
    plt.quiver(x, y, fx + gx, fy + gy, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def predictor(z, h, eta, a, alpha, mu):
    """
    Computes predictor for Euler integration
    """
    pred = z + h*stuartLandau(z, a, alpha, mu) + eta*noiseFunction(z)
    return pred

def update_z(z, h, eta, a, alpha, mu):
    """
    Gives z at t + h from z at t
    """
    pred = predictor(z, h, eta, a, alpha, mu)
    z_updated = z + 0.5*(stuartLandau(z, a, alpha, mu) + stuartLandau(pred, a, alpha, mu))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

def EulerInteg(h, a, alpha, numsteps, phase, mu):
    """
    Integration routine
    """
    #Initial points
    time = 0
    z = array([cos(2*pi*phase), sin(2*pi*phase)])
    
    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2), dtype=ndarray)
    time_list[0] = time
    trajectory[0] = z
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(h), size=1)[0], random.normal(loc=0, scale=sqrt(h), size=1)[0]

        z_updated = update_z(z, h, eta, a, alpha, mu)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
        
    return time_list, trajectory

def stochastic_average(h, a, alpha, numsteps, numiter, phase, mu):
    """
    The main piece
    Repeats various integrations and uses them to average the position of the point at each t
    """
    #Same times for all trajectories
    times = zeros(numsteps+1)
    trajectories = zeros((numiter, numsteps+1, 2), dtype=ndarray)
    
    print("Computing the Euler integration (this might take a while)...")
    
    
    for i in tqdm(range(numiter)):
        times, trajectories[i] = EulerInteg(h, a, alpha, numsteps, phase, mu)
    
    stochastic_av = zeros((numsteps+1, 2), dtype=ndarray)
    for j in range(numsteps+1):
        average_pos = array([0., 0.])
        for k in range(numiter):
            average_pos = average_pos + trajectories[k][j]
        stochastic_av[j] = average_pos/numiter
        
    return times, stochastic_av

#exponential function for fitting
def exponential(x, a, b):
    return a * exp( -b * x)

def find_amplitudes(time, trajectory):
    """
    Find indexes of local maxima, their value, and the time t associated
    """
    #Select points in time and trajectory lists corresponding to maxima
    peaks, properties = find_peaks(trajectory, height = 0.6, prominence=1)
    times = time[peaks]
    amplitudes = trajectory[peaks]
    
    period_list = zeros(len(peaks)-1)
    for i in range(len(peaks)-1):
        period = times[i+1] - times[i]
        period_list[i] = period
    period = average(period_list)
    
    times = array(times, dtype='float64')
    amplitudes = array(amplitudes, dtype='float64')
    
    #Fit and get the fit info
    popt, pcov = curve_fit(exponential, times, amplitudes)
    
    return times, amplitudes, period, popt
        
def main():
    startTime = datetime.now()

    h=0.01; a, alpha = [0, 1]; numsteps = 10000; numiter = 1000; start_phase = random.uniform(-1, 1); mu = 1

    #Example of integration
    time_list, trajectory = EulerInteg(h, a, alpha, numsteps,  start_phase, mu)
    plt.plot(trajectory[:,0], trajectory[:,1], color='green')
    plt.title("Typical trajectory with noise")
    vector_space(a, alpha, mu)

    plt.plot(time_list, trajectory[:,0], color='green')
    plt.title("Typical oscillations with noise")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    times, stochastic_av = stochastic_average(h, a, alpha, numsteps, numiter, start_phase, mu)
    
    plt.plot(stochastic_av[:,0], stochastic_av[:,1], color='blue')
    plt.title("Average trajectory with noise")
    vector_space(a, alpha, mu)
    
    times_maxima, maxima, period, params = find_amplitudes(times,stochastic_av[:,0])
    
    plt.plot(times, stochastic_av[:,0], color='blue')
    #Check graphically if the maxima have been found
    plt.plot(times_maxima, maxima, "r+")
    plt.title("Average oscillations with noise")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()
    
    #Compare with numerical values found
    plt.plot(times_maxima, maxima, "r+")
    t = linspace(0, h*numsteps, num=numsteps)
    plt.plot(t, exponential(t, params[0], params[1]))
    plt.title("Average oscillations with noise")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    print("The average period of the oscillations is", period, "s")
    print('The rate of decay is', params[1], "s" + "\u00B2")
    
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()