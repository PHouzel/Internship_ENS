#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:21:34 2021

@author: pierrehouzelstein
"""
from numpy import array, cos, sin, pi, zeros, sqrt, random, linspace, loadtxt, meshgrid
from scipy import interpolate
from numba import jit, float64, int64, types
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Same as before: functions for Euler integration routine
@jit(float64[:](float64[:], float64, float64))
def AR2_model(z, beta_1, beta_2):
    """
    Studied function
    """
    wee = -(1 + beta_1)
    wei = beta_2 + beta_1 - 1
    wie = 1
    E, I = z
    fE = wee * E + wei * I
    fI = wie * E
    return array([fE, fI])

@jit(float64[:](float64[:]))
def noiseFunction(z):
    """
    Added noise
    """
    x, y = z
    gx = 0.01
    gy = 0.01
    return array([gx, gy])

@jit((float64, float64))
def vector_space(beta_1, beta_2):
    """
    Plots the vector field linked to the system
    """
    wee = -(1 + beta_1)
    wei = beta_2 + beta_1 - 1
    wie = 1
    
    x, y = meshgrid(linspace(-1, 1, num=20), linspace(-2, 2, num=20))
    fx, fy =  array([wee * x + wei * y, wie * x])
    gx, gy = array([0.01, 0.01])
    plt.quiver(x, y, fx + gx, fy + gy, color='red')
    plt.xlabel('E')
    plt.ylabel('I')
    plt.show()

@jit((float64[:])(float64[:], float64, float64[:], float64, float64),nopython=True)
def update_z(z, h, eta, beta_1, beta_2):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*AR2_model(z, beta_1, beta_2) + eta*noiseFunction(z)
    z_updated = z + 0.5*( AR2_model(z, beta_1, beta_2) + AR2_model(pred, beta_1, beta_2))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64, float64, float64, int64, float64), nopython=True)
def EulerInteg(h, beta_1, beta_2, numsteps, phase):
    """
    Integration routine
    """
    #Initial points
    time = 0.
    z = array([cos(phase), sin(phase)])
    
    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2))#, dtype=ndarray)
    time_list[0] = time
    trajectory[0] = z
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(h), size=2)

        z_updated = update_z(z, h, eta, beta_1, beta_2)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
    return time_list, trajectory


def compute_mean_phase(T, h, N, beta_1, beta_2, initial_phase, isochrone_func):
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_phase = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(h, beta_1, beta_2, numsteps, initial_phase)
        #Compute phase of last position
        mean_phase += isochrone_func(trajectory[-1][0], trajectory[-1][1])
    mean_phase = mean_phase/N
    return mean_phase

def compute_single_shift(T, h, N, beta_1, beta_2, initial_phase, pulse, isochrone_func):
    shifted_phase = initial_phase + pulse
    shift = compute_mean_phase(T, h, N, beta_1, beta_2, shifted_phase, isochrone_func) - compute_mean_phase(T, h, N, beta_1, beta_2, initial_phase, isochrone_func)
    return shift % 2*pi

def compute_PRC(T, h, N, beta_1, beta_2, pulse, isochrone_func, n_points):
    phase_list = linspace(0, 2*pi, n_points)
    PRC_list = []
    for i in tqdm(range(len(phase_list))):
        PRC_list.append(compute_single_shift(T, h, N, beta_1, beta_2, phase_list[i], pulse, isochrone_func))
    return phase_list, array(PRC_list)

def main():
    startTime = datetime.now()
    #h timesteps; betas = constants for function; T = period; N = numner of steps used to compute mean phase
    h=0.01; beta_1, beta_2 = [-0.9606, 1.8188]; T = 16.708; N = 100
    
    #Single integration: example
    numsteps = 100000; initial_phase = random.uniform(0, 2*pi)
    time_list, trajectory = EulerInteg(h, beta_1, beta_2, numsteps, initial_phase)
    
    plt.plot(trajectory[:,0], trajectory[:,1], color='blue')
    plt.title("Typical trajectory with noise")
    vector_space(beta_1, beta_2)

    plt.plot(time_list, trajectory[:,0], color='red')
    plt.title("Typical oscillations of E with noise")
    plt.xlabel("t")
    plt.ylabel("E")
    plt.show()
    
    #Use given data
    #Use given data to get the phase function
    yp = 0.5; ym = -0.5
    xp = 0.5; xm = -0.5
    y = linspace(ym, yp, 80+1)#;  dy = y[1] - y[0]
    x = linspace(xm, xp, 80+1)#;  dx = x[1] - x[0]
    isochrones = loadtxt('./isocronesD0.01')
    
    plt.pcolormesh(x, y, isochrones, cmap='gist_rainbow')
    plt.title('Isochrones from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.show()
    
    isochrone_func = interpolate.interp2d(x, y, isochrones)
    pulse = 0.2; n_points = 100
    phase_list, PRC_list = compute_PRC(T, h, N, beta_1, beta_2, pulse, isochrone_func, n_points)
    plt.plot(phase_list, PRC_list, color='red')
    plt.title("PRC with an initial phase shift of {}".format(pulse))
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.show()
    
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()