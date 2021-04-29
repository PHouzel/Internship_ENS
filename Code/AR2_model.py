#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:21:34 2021

@author: pierrehouzelstein
"""
from numpy import array, cos, sin, arccos, arcsin, arctan, pi, zeros, sqrt, random, linspace, loadtxt, meshgrid, shape
from scipy import interpolate
from numba import jit, float64, int64, types
import matplotlib.pyplot as plt
from datetime import datetime
from math import atan2
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
    
    x, y = meshgrid(linspace(-0.2, 0.2, num=20), linspace(-0.4, 0.4, num=20))
    fx, fy =  array([wee * x + wei * y, wie * x])
    gx, gy = array([0.01, 0.01])
    plt.quiver(x, y, fx + gx, fy + gy, color='red')
    plt.xlabel('E')
    plt.ylabel('I')

@jit((float64[:])(float64[:], float64, float64[:], float64, float64),nopython=True)
def update_z(z, h, eta, beta_1, beta_2):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*AR2_model(z, beta_1, beta_2) + eta*noiseFunction(z)
    z_updated = z + 0.5*( AR2_model(z, beta_1, beta_2) + AR2_model(pred, beta_1, beta_2))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64[:], float64, float64, float64, int64), nopython=True)
def EulerInteg(z, h, beta_1, beta_2, numsteps):
    """
    Integration routine
    """
    #Initial time
    time = 0.
    
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

"""
#V1: using simple interpolation function
def compute_mean_phase(initial_z, T, h, N, beta_1, beta_2, isochrone_func):
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_phase = 0
    mean_trajectory = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, beta_1, beta_2, numsteps)
        #Compute phase of last position
        mean_phase += isochrone_func(trajectory[-1][0], trajectory[-1][1])
        mean_trajectory += trajectory
    mean_phase = mean_phase/N
    mean_trajectory = mean_trajectory/N
    return mean_phase[0], mean_trajectory
"""

def compute_mean_phase(initial_z, T, h, N, beta_1, beta_2, real_isochrone_func, im_isochrone_func):
    #V2: get cosinus and sinus, and take phase from there
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_real = 0
    mean_im = 0
    mean_trajectory = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, beta_1, beta_2, numsteps)
        #Compute phase of last position
        real = real_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        im = im_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        mean_real += real
        mean_im += im
        mean_trajectory = mean_trajectory + trajectory
    mean_phase = atan2(mean_im, mean_real)
    if mean_phase < 0: mean_phase += 2*pi
    mean_trajectory = mean_trajectory/N
    return mean_phase, mean_trajectory


def compute_single_shift(initial_z, T, h, N, beta_1, beta_2, pulse, real_isochrone_func, im_isochrone_func):
    shifted_z = initial_z + pulse

    shifted_mean_phase, shifted_mean_trajectory = compute_mean_phase(shifted_z, T, h, N, beta_1, beta_2, real_isochrone_func, im_isochrone_func)
    initial_mean_phase, initial_mean_trajectory = compute_mean_phase(initial_z, T, h, N, beta_1, beta_2, real_isochrone_func, im_isochrone_func)
    
    shift = (shifted_mean_phase - initial_mean_phase)
    if shift < 0: shift += 2*pi
    return shift, initial_mean_trajectory, shifted_mean_trajectory

def compute_PRC(T, h, N, beta_1, beta_2, pulse, real_isochrone_func, im_isochrone_func, n_points):
    phase_list = linspace(0, 2*pi, n_points)
    PRC_list = []
    for i in tqdm(range(len(phase_list))):
        z = array([0.3*cos(phase_list[i]), 0.3*sin(phase_list[i])])
        shift, _, _ = compute_single_shift(z, T, h, N, beta_1, beta_2, pulse, real_isochrone_func, im_isochrone_func)
        PRC_list.append(shift)
    return phase_list, array(PRC_list)

def main():
    startTime = datetime.now()
    #h timesteps; betas = constants for function; T = period; N = number of steps used to compute mean phase
    h=0.01; beta_1, beta_2 = [-0.9606, 1.8188]; T = 16.708; N = 100
    pulse = array([0.1, 0.]); n_points = 1000
    
    #Single integration: example
    numsteps = 100000; initial_phase = random.uniform(0, 2*pi)
    z_test = array([0.3*cos(initial_phase), 0.3*sin(initial_phase)])
    
    time_list, trajectory = EulerInteg(z_test, h, beta_1, beta_2, numsteps)
    
    plt.plot(trajectory[:,0], trajectory[:,1], color='blue')
    plt.title("Typical trajectory with noise")
    #vector_space(beta_1, beta_2)
    plt.show()

    plt.plot(time_list, trajectory[:,0], color='red')
    plt.title("Typical oscillations of E with noise")
    plt.xlabel("t")
    plt.ylabel("E")
    plt.savefig("./Output_files/AR2_trajectory.png")
    plt.show()
    
    #Use given data
    #Use given data to get the phase function
    yp = 0.5; ym = -0.5
    xp = 0.5; xm = -0.5
    y = linspace(ym, yp, 80+1)#;  dy = y[1] - y[0]
    x = linspace(xm, xp, 80+1)#;  dx = x[1] - x[0]
    #isochrones = loadtxt('./isocronesD0.01')
    
    isochrones_real = loadtxt('./Data/realValuesD0.01')
    isochrones_im = loadtxt('./Data/imagValuesD0.01')
    isochrones_real_func = interpolate.interp2d(x, y, isochrones_real)
    isochrones_im_func = interpolate.interp2d(x, y, isochrones_im)
    """
    phase = arctan(isochrones_im/isochrones_real)
    plt.pcolormesh(x, y,  phase, cmap='gist_rainbow')
    shift, initial_mean_trajectory, shifted_mean_trajectory = compute_single_shift(z_test, T, h, N, beta_1, beta_2, pulse, isochrones_real_func, isochrones_im_func)
    phase_list = linspace(0, 2*pi, n_points)
    plt.plot(0.3*cos(phase_list), 0.3*sin(phase_list), color = 'black')
    plt.plot(0.3*cos(phase_list) + pulse[0], 0.3*sin(phase_list), color = 'gray')
    plt.plot(initial_mean_trajectory[:,0], initial_mean_trajectory[:,1])
    plt.plot(shifted_mean_trajectory[:,0], shifted_mean_trajectory[:,1])
    plt.legend(["Initial points", "Shifted points", "Initial trajectory", "Shifted trajectory"])
    plt.title('Isochrones from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.show()
    """
    phase_list, PRC_list = compute_PRC(0, h, N, beta_1, beta_2, pulse, isochrones_real_func, isochrones_im_func, n_points)
    plt.plot(phase_list, PRC_list, "r+")
    plt.title("PRC with an initial E-shift of 0.1, initial time")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.savefig("./Output_files/AR2_PRC_0T.jpg")
    plt.show()
    
    with open('./Output_files/phase_0T.txt', 'w') as output:
        content = str(phase_list)
        output.write(content)
    with open('./Output_files/shift_0T.txt', 'w') as output:
        content = str(PRC_list)
        output.write(content)        
    
    phase_list, PRC_list = compute_PRC(T, h, N, beta_1, beta_2, pulse, isochrones_real_func, isochrones_im_func, n_points)
    plt.plot(phase_list, PRC_list, "r+")
    plt.title("PRC with an initial E-shift of 0.1, after one period")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.savefig("./Output_files/AR2_PRC_1T.jpg")
    plt.show()
    
    with open('./Output_files/phase_1T.txt', 'w') as output:
        content = str(phase_list)
        output.write(content)
    with open('./Output_files/shift_1T.txt', 'w') as output:
        content = str(PRC_list)
        output.write(content)           
    
    phase_list, PRC_list = compute_PRC(2*T, h, N, beta_1, beta_2, pulse, isochrones_real_func, isochrones_im_func, n_points)
    plt.plot(phase_list, PRC_list, "r+")
    plt.title("PRC with an initial E-shift of 0.1, after two periods")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.savefig("./Output_files/AR2_PRC_2T.jpg")
    plt.show()

    with open('./Output_files/phase_2T.txt', 'w') as output:
        content = str(phase_list)
        output.write(content)
    with open('./Output_files/shift_2T.txt', 'w') as output:
        content = str(PRC_list)
        output.write(content)  
        
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()
    
    """
    isochrone_func = interpolate.interp2d(x, y, isochrones)
    
    plt.pcolormesh(x, y, isochrones_cos, cmap='gist_rainbow')
    plt.title('Isochrones cosinus from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="cos($\Theta$", orientation="vertical")
    plt.show()
    
    plt.pcolormesh(x, y, isochrones_sin, cmap='gist_rainbow')
    plt.title('Isochrones sinus from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="sin($\Theta$", orientation="vertical")
    plt.show()

    plt.pcolormesh(x, y, isochrones, cmap='gist_rainbow')
    shift, initial_mean_trajectory, shifted_mean_trajectory = compute_single_shift(z_test, T, h, N, beta_1, beta_2, pulse, isochrone_func)
    phase_list = linspace(0, 2*pi, n_points)
    plt.plot(0.3*cos(phase_list), 0.3*sin(phase_list), color = 'black')
    plt.plot(0.3*cos(phase_list) + pulse[0], 0.3*sin(phase_list), color = 'gray')
    plt.plot(initial_mean_trajectory[:,0], initial_mean_trajectory[:,1])
    plt.plot(shifted_mean_trajectory[:,0], shifted_mean_trajectory[:,1])
    plt.legend(["Initial points", "Shifted points", "Initial trajectory", "Shifted trajectory"])
    plt.title('Isochrones from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.show()
    
    phase_list, PRC_list = compute_PRC(T, h, N, beta_1, beta_2, pulse, isochrone_func, n_points)
    plt.plot(phase_list, PRC_list, "r+")
    plt.title("PRC with an initial E-shift of 0.1")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.show()
    
        plt.pcolormesh(x, y, isochrones_real, cmap='gist_rainbow')
    plt.title('Isochrones real part from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="sin($\Theta$", orientation="vertical")
    plt.show()
    
    plt.pcolormesh(x, y,  isochrones_real_func(x, y), cmap='gist_rainbow')
    plt.title('Isochrones real part using interpolated function')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="sin($\Theta$", orientation="vertical")
    plt.show()
    
    plt.pcolormesh(x, y, isochrones_im, cmap='gist_rainbow')
    plt.title('Isochrones imaginary part from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="Im(z(E, I))", orientation="vertical")
    plt.show()
    
    plt.pcolormesh(x, y,  isochrones_im_func(x, y), cmap='gist_rainbow')
    plt.title('Isochrones imaginary part using interpolated function')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="Im(z(E, I))", orientation="vertical")
    plt.show()
    
    phase = arctan(isochrones_im/isochrones_real)
    plt.pcolormesh(x, y,  phase, cmap='gist_rainbow')
    plt.title('Isochrones phase using real and imaginary parts from data')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
    plt.show()
    
    phase = arctan(isochrones_im_func(x, y)/isochrones_real_func(x, y))
    plt.pcolormesh(x, y,  phase, cmap='gist_rainbow')
    plt.title('Isochrones phase using real and imaginary parts from interpolation')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$", orientation="vertical")
    plt.show()
    """