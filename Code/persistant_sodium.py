#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:54:14 2021

@author: pierrehouzelstein
persistent sodium + transient potassium model
Izhikevich p 90
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
from scipy.signal import find_peaks, periodogram
from matplotlib.mlab import find
import numpy as np
from statsmodels import api as sm
plt.style.use('seaborn-poster')

@jit(nopython=True)
def pers_sod(z, t, stimulus, perturbation_time):
    
    V, n = z
    
    C = 1; gL = 8; gNa = 20; gK = 10; tau = 1; ENa = 60; EK = -90
    #Low treshold case
    EL = -78; I = 40
    
    m_inf = 1/(1+exp((-20-V)/15))
    n_inf = 1/(1+exp((-45-V)/5))
    
    dV = (stimulus(t, perturbation_time) + I - gL*(V - EL) - gNa * m_inf *(V-ENa) - gK*n*(V-EK))/C
    dn = (n_inf - n)/tau
    return array([dV, dn])

@jit(nopython=True)
def noiseFunction(eta, amplitude):
    """
    Added noise amplitude
    """
    gx = (amplitude)*eta[0]
    gy = (amplitude)*eta[1]
    return array([gx, gy])

@jit(nopython=True)
def update_z(z, t, stimulus, perturbation_time, h, eta, amplitude):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*pers_sod(z, t, stimulus, perturbation_time) + noiseFunction(eta, amplitude)
        
    z_updated = z + 0.5*(pers_sod(z, t, stimulus, perturbation_time) + pers_sod(pred, t, stimulus, perturbation_time))*h\
                         + noiseFunction(eta, amplitude)
    return z_updated

@jit(nopython=True)
def EulerInteg(z, initial_t, stimulus, perturbation_time, h, amplitude, numsteps):
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

        z = update_z(z, time, stimulus, perturbation_time, h, eta, amplitude)
        time += h
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z

    return time_list, trajectory

@jit(nopython=True)
def stimulus(time, perturbation_time):
    if abs(time-perturbation_time) < 0.001: stimulus = 10000
    else: stimulus = 0
    return stimulus

@jit(nopython=True)
def no_stim(time, perturbation_time):
    return 0

z0 = array([-10, 0.3]); h = 0.001; amplitude = 0.; numsteps = 100000

#sampling frequency
fs = 1/h


time_list_1, trajectory_1 = EulerInteg(z0, 0, no_stim, 0, h, amplitude, numsteps)
amplitude = 0.2
test_time_list, test_trajectory = EulerInteg(z0, 0, no_stim, 0, h, amplitude, numsteps)

plt.plot(time_list_1, trajectory_1[:,0], "b-")
plt.show()

plt.plot(test_time_list, test_trajectory[:,0], "r-")
plt.show()

def freq_zero_crossing(sig, fs):
    """
    https://qingkaikong.blogspot.com/2017/01/signal-processing-finding-periodic.html
    Frequency estimation from zero crossing method
    sig - input signal
    fs - sampling rate
    
    return: 
    dominant period
    
    
    """
    # Find the indices where there's a crossing
    #indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    indices = np.where((sig[1:] >= 0) & (sig[:-1] < 0))

    # Let's calculate the real crossings by interpolate
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Let's get the time between each crossing
    # the diff function will get how many samples between each crossing
    # we divide the sampling rate to get the time between them
    delta_t = np.diff(crossings) / fs
    
    # Get the mean value for the period
    period = np.mean(delta_t)
    
    return period

#center around zero
clean_traj = trajectory_1[:,0] - np.mean(trajectory_1[:,0])
noisy_traj = test_trajectory[:,0] - np.mean(test_trajectory[:,0])

period_from_zero_crossing = freq_zero_crossing(clean_traj, fs)
print('The unperturbed period estimation is %.1f'%(period_from_zero_crossing))
period_from_zero_crossing = freq_zero_crossing(noisy_traj, fs)
print('The noisy period estimation is %.1f'%(period_from_zero_crossing))



# get the frequency and spectrum
f, Pxx = periodogram(clean_traj, fs = fs, window='hanning', scaling='spectrum')

plt.figure(figsize = (10, 8))
plt.plot(f, Pxx)
plt.xlim(0, 10)
plt.yscale('log')
plt.xlabel('Frequency (cycles/t)')
plt.ylabel('Spectrum Amplitude')

# print the top 6 period in the signal
print("Fourier tranform peak: clean signal:")
#for amp_arg in np.argsort(np.abs(Pxx))[::-1][1:6]:
    #time = 1 / f[amp_arg]
    #print(time)
amp_arg = np.argsort(np.abs(Pxx))[::-1][1]
time = 1 / f[amp_arg]
print(time)
    
f, Pxx = periodogram(noisy_traj, fs = fs, window='hanning', scaling='spectrum')

plt.figure(figsize = (10, 8))
plt.plot(f, Pxx)
plt.xlim(0, 10)
plt.yscale('log')
plt.xlabel('Frequency (cycles/t)')
plt.ylabel('Spectrum Amplitude')

print("Fourier tranform peak: noisy signal:")
# print the top 6 period in the signal
#for amp_arg in np.argsort(np.abs(Pxx))[::-1][1:6]:
    #time = 1 / f[amp_arg]
    #print(time)
amp_arg = np.argsort(np.abs(Pxx))[::-1][1]
period = 1 / f[amp_arg]
print(period)

#Get phase list
phase_list = ((2*pi*test_time_list)/period)
phase_list = mod(phase_list, 2*pi)

"""
#Get period
#Cut off irregular part
time_list_1_bis = time_list_1[1000:]; trajectory_1_bis = trajectory_1[1000:]
plt.plot(time_list_1_bis, trajectory_1_bis[:,0], "b-")
peaks, _ = find_peaks(trajectory_1_bis[:,0])

period = (time_list_1_bis[peaks[-1]] - \
    time_list_1_bis[peaks[0]])/(len(peaks) - 1)
    
#get a phase list
#Take a single period
time_list_1_ter = time_list_1_bis[peaks[0]:peaks[1]]
trajectory_1_ter = trajectory_1_bis[peaks[0]:peaks[1]]
phase_list = ((2*pi*time_list_1_ter)/period)
phase_list = mod(phase_list, 2*pi)

plt.plot(time_list_1_ter, trajectory_1_ter[:,0], "k-")
plt.show()

#Idea: perturb the system at different times ie phases and get shift

#Next step: compute phase shift of a single perturbed trajectory
pert_time = time_list_1_bis[2000]
test_time_list, test_trajectory = EulerInteg(z0, 0, stimulus, pert_time, h, amplitude, numsteps)
test_peaks, _ = find_peaks(test_trajectory[:,0])

plt.plot(time_list_1, trajectory_1[:,0], "b-")
plt.plot(test_time_list, test_trajectory[:,0], "r-")
plt.plot(test_time_list[test_peaks], test_trajectory[:,0][test_peaks], "k+")
plt.plot(time_list_1_bis[peaks], trajectory_1_bis[:,0][peaks], "b+")
plt.axvline(pert_time, color="black")
plt.show()

def compute_phase_shift(time_list, indice):
    #Take time list of elements between first and second peak; take certain points and perturbate
    pert_time = time_list[int(indice)]
    test_time_list, test_trajectory = EulerInteg(z0, 0, stimulus, pert_time, h, amplitude, numsteps)
    test_peaks, _ = find_peaks(test_trajectory[:,0])
    
    time_shift = test_time_list[test_peaks[-1]] - time_list_1_bis[peaks[-1]]
    phase_shift = ((2*pi*time_shift)/period)
    phase_shift = mod(phase_shift, 2*pi)
    
    return phase_shift

time_list = time_list_1_ter[::10]
phase_list = phase_list[::10]
phase_shifts = zeros(len(phase_list))
for i in tqdm(range(len(phase_list))):
    phase_shifts[i] = compute_phase_shift(time_list, i)

plt.plot(phase_list, phase_shifts, "k+")
plt.show()
"""

