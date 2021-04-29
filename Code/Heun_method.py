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
from matplotlib.axis import Axis
from numba import jit, float64, int64, types
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

@jit(float64[:](float64[:], float64, float64, float64))
def stuartLandau(z, a, alpha, mu):
    """
    Studied function
    """
    x, y = z
    fx = alpha*x*(mu - (x**2 + y**2)) - y*(1 + alpha*a*(x**2 + y**2))
    fy = alpha*y*(mu - (x**2 + y**2)) + x*(1 + alpha*a*(x**2 + y**2))
    return array([fx, fy])

@jit(float64[:](float64[:]))
def noiseFunction(z):
    """
    Added noise
    """
    x, y = z
    gx = 0.1
    gy = 0.1
    return array([gx, gy])

@jit
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

@jit(float64[:](float64[:], float64, float64[:], float64, float64, float64), nopython=True)
def predictor(z, h, eta, a, alpha, mu):
    """
    Computes predictor for Euler integration
    """
    pred = z + h*stuartLandau(z, a, alpha, mu) + eta*noiseFunction(z)
    return pred

@jit((float64[:])(float64[:], float64, float64[:], float64, float64, float64),nopython=True)
def update_z(z, h, eta, a, alpha, mu):
    """
    Gives z at t + h from z at t
    """
    pred = predictor(z, h, eta, a, alpha, mu)
    z_updated = z + 0.5*(stuartLandau(z, a, alpha, mu) + stuartLandau(pred, a, alpha, mu))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64, float64, float64, int64, float64, float64), nopython=True)
def EulerInteg(h, a, alpha, numsteps, phase, mu):
    """
    Integration routine
    """
    #Initial points
    time = 0.
    z = array([cos(2*pi*phase), sin(2*pi*phase)])
    
    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2))#, dtype=ndarray)
    time_list[0] = time
    trajectory[0] = z
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(h), size=2)

        z_updated = update_z(z, h, eta, a, alpha, mu)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
    return time_list, trajectory

@jit(types.Tuple((float64[:], float64[:,:]))(float64, float64, float64, int64, int64, float64, float64), nopython=True)
def stochastic_average(h, a, alpha, numsteps, numiter, phase, mu):
    """
    The main piece
    Repeats various integrations and uses them to average the position of the point at each t
    """
    #Same times for all trajectories
    times = zeros(numsteps+1)
    stochastic_av = zeros((numsteps+1, 2))
    for i in range(numiter):
        times, AverageTraj = EulerInteg(h, a, alpha, numsteps, phase, mu)
        stochastic_av = stochastic_av + AverageTraj
    stochastic_av = stochastic_av/numiter
    
    return times, stochastic_av

#exponential function for fitting
@jit
def exponential(x, a, b):
    return a * exp( -b * x)

@jit
def find_amplitudes(time, trajectory, sensitivity):
    """
    Find indexes of local maxima, their value, and the time t associated
    """
    #Select points in time and trajectory lists corresponding to maxima

    peaks, properties = find_peaks(trajectory, height = 0, prominence=sensitivity)

    times = time[peaks]
    amplitudes = trajectory[peaks]
    
    period_list = zeros(len(peaks) - 1)
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

    h=0.01; a, alpha = [0.0, 1.0]; numsteps = 100000; numiter = 1000; start_phase = random.uniform(-1, 1); mu = 1.0
    sensitivity = 0.9 #parameter used to find the peaks; around 1 when limit cycle

    #Example of integration
    time_list, trajectory = EulerInteg(h, a, alpha, numsteps, start_phase, mu)
    plt.plot(trajectory[:,0], trajectory[:,1], color='green')
    plt.title("Typical trajectory with noise")
    plt.show()
    #vector_space(a, alpha, mu)

    plt.plot(time_list, trajectory[:,0], color='green')
    plt.title("Typical oscillations with noise")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()
    
    times, stochastic_av = stochastic_average(h, a, alpha, numsteps, numiter, start_phase, mu)
    
    plt.plot(stochastic_av[:,0], stochastic_av[:,1], color='blue')
    plt.title("Average trajectory with noise")
    plt.show()
    #vector_space(a, alpha, mu)
    
    times_maxima, maxima, period, params = find_amplitudes(times, stochastic_av[:,0], sensitivity) #error there
    
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
    plt.plot(t, exponential(t, params[0], params[1]), color='green')
    plt.title("Average oscillations with noise")
    plt.legend(["Peaks", "Exponential fit"])
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    print("The average period of the oscillations is", period, "s")
    print('The rate of decay is', params[1], "s-1")

    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()