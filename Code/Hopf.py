#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:04:35 2021

@author: pierrehouzelstein

Goal: apply PRC computation method to other systems
Above and below Hopf bifurcation
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from numba import jit, float64, int64, types
from numpy import array, sqrt, zeros, random, pi, linspace, cos, sin, loadtxt, arctan
from math import atan2
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

#Def studied systems

@jit(float64[:](float64[:], float64), nopython=True)
def Hopf(z, beta):
    x, y = z
    fx = beta*x - y - x*(x**2 + y**2)
    fy = x + beta*y - y*(x**2 + y**2)
    return array([fx,fy])

@jit(float64[:](float64[:]), nopython=True)
def noiseFunction(z):
    """
    Added noise
    """
    #D = 0.1
    #gx = sqrt(2*D)
    #gy = sqrt(2*D)
    gx = 0.1
    gy = 0.1
    return array([gx, gy])

@jit((float64[:])(float64[:], float64, float64[:], float64), nopython=True)
def update_z(z, h, eta, beta):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*Hopf(z, beta) + eta*noiseFunction(z)
    z_updated = z + 0.5*( Hopf(z, beta) + Hopf(pred, beta))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64[:], float64, float64, int64), nopython=True)
def EulerInteg(z, h, beta, numsteps):
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

        z_updated = update_z(z, h, eta, beta)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
    return time_list, trajectory

#@jit(nopython=True)
def compute_mean_phase(initial_z, T, h, N, beta, real_isochrone_func, im_isochrone_func):
    #V2: get cosinus and sinus, and take phase from there
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_real = 0
    mean_im = 0
    mean_trajectory = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, beta, numsteps)
        #Compute phase of last position
        real = real_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        im = im_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        mean_real += real
        mean_im += im
        mean_trajectory = mean_trajectory + trajectory
    mean_phase = atan2(mean_im, mean_real)
    mean_trajectory = mean_trajectory/N
    return mean_phase, mean_trajectory

#@jit(nopython=True)
def compute_single_shift(initial_z, T, h, N, beta, pulse, real_isochrone_func, im_isochrone_func):
    shifted_z = initial_z + pulse

    shifted_mean_phase, shifted_mean_trajectory = compute_mean_phase(shifted_z, T, h, N, beta, real_isochrone_func, im_isochrone_func)
    initial_mean_phase, initial_mean_trajectory = compute_mean_phase(initial_z, T, h, N, beta, real_isochrone_func, im_isochrone_func)
    
    shift = (shifted_mean_phase - initial_mean_phase)
    if shift < 0: shift += 2*pi
    if shift > pi: shift += -2*pi
    return shift, initial_mean_trajectory, shifted_mean_trajectory

#@jit(nopython=True)
def compute_PRC(T, h, N, beta, pulse, real_isochrone_func, im_isochrone_func, n_points):
    phase_list = linspace(0, 2*pi, n_points)
    PRC_list = []
    for i in tqdm(range(len(phase_list))):
        z = array([cos(phase_list[i]), sin(phase_list[i])])
        shift, _, _ = compute_single_shift(z, T, h, N, beta, pulse, real_isochrone_func, im_isochrone_func)
        PRC_list.append(shift)
    return phase_list, array(PRC_list)

def main():
    startTime = datetime.now()
    
    #setting up grid for the phase spaces
    yp = 1.5; ym = -1.5
    xp = 1.5; xm = -1.5
    y = linspace(ym, yp, 100)
    x = linspace(xm, xp, 100)
    
    #setting up integration parameters
    h = 0.01 #timestep; 
    N = 10000     #number of trajectories over which we average the phase
    pulse = array([0.1, 0]) #Perturbation in the phase space
    n_points = 100 #number of points on the PRC
    """
    #case 1: LC (above bifurcation)
    beta = 1; T = 2*pi
    
    #load data
    isochrone = loadtxt('./Data/hopf/above/data/isocronesD0.01125')
    isochrones_real = loadtxt('./Data/hopf/above/data/realValuesD0.01125')
    isochrones_im = loadtxt('./Data/hopf/above/data/imagValuesD0.01125')
    #interpolate
    isochrones_real_func = interpolate.interp2d(x, y, isochrones_real, kind = 'cubic')
    isochrones_im_func = interpolate.interp2d(x, y, isochrones_im, kind = 'cubic')
    
    #plot phase space to see if ok
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    time_list, trajectory = EulerInteg(array([1.0, 1.0]), h, beta, 10000)
    plt.plot(trajectory[:,0], trajectory[:,1], color='white')
    phase_list = linspace(0, 2*pi, n_points)
    plt.plot(cos(phase_list), sin(phase_list), color = 'black')
    plt.plot(cos(phase_list) + pulse[0], sin(phase_list), color = 'gray')
    plt.legend(["Typical trajectory", "Initial points", "Shifted points"])
    plt.title('Hopf LC (above bifurcation): Isochrones from data')
    plt.xlabel("x")
    plt.ylabel("y")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.savefig("./Data/hopf/output/Hopf_LC_isochrones.jpg")
    plt.show()

    #compute PRC
    #reference points
    phase_list_0, PRC_list_0 = compute_PRC(0, h, N, beta, pulse, isochrones_real_func, isochrones_im_func, n_points)
    phase_list, PRC_list = compute_PRC(T, h, N, beta, pulse, isochrones_real_func, isochrones_im_func, n_points)
    plt.plot(phase_list_0, PRC_list_0, "r-")
    plt.plot(phase_list, PRC_list, "b+")
    plt.title("Hopf LC (above bifurcation): PRC with an initial E-shift of 0.1, after one period")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.legend(["Reference PRC", "PRC at T"])
    plt.savefig("./Data/hopf/output/Hopf_LC_PRC.jpg")
    plt.show()

    with open('./Data/hopf/output/Hopf_LC_phase_0T.txt', 'w') as output:
        for i in range(len(phase_list_0)):
            content = str(phase_list_0[i])
            output.write(content + " ")    
    with open('./Data/hopf/output/Hopf_LC_phase_1T.txt', 'w') as output:
        for i in range(len(phase_list)):
            content = str(phase_list[i])
            output.write(content + " ")
    with open('./Data/hopf/output/Hopf_LC_shift_0T.txt', 'w') as output:
        for i in range(len(PRC_list_0)):
            content = str(PRC_list_0[i])
            output.write(content + " ")
    with open('./Data/hopf/output/Hopf_LC_shift_1T.txt', 'w') as output:
        for i in range(len(PRC_list)):
            content = str(PRC_list[i])
            output.write(content + " ")
    """
    #case 2: focus (below bifurcation)
    beta = -0.1; T = 2*pi
    
    #load data
    isochrone = loadtxt('./Data/hopf/below/data/isocronesD0.01125')
    isochrones_real = loadtxt('./Data/hopf/below/data/realValuesD0.01125')
    isochrones_im = loadtxt('./Data/hopf/below/data/imagValuesD0.01125')
    #interpolate
    isochrones_real_func = interpolate.interp2d(x, y, isochrones_real, kind = 'cubic')
    isochrones_im_func = interpolate.interp2d(x, y, isochrones_im, kind = 'cubic')
    
    #plot phase space to see if ok
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    time_list, trajectory = EulerInteg(array([1.0, 1.0]), h, beta, 10000)
    plt.plot(trajectory[:,0], trajectory[:,1], color='white')
    phase_list = linspace(0, 2*pi, n_points)
    plt.plot(cos(phase_list), sin(phase_list), color = 'black')
    plt.plot(cos(phase_list) + pulse[0], sin(phase_list), color = 'gray')
    plt.legend(["Typical trajectory", "Initial points", "Shifted points"])
    plt.title('Hopf focus (below bifurcation): Isochrones from data')
    plt.xlabel("x")
    plt.ylabel("y")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.savefig("./Data/hopf/output/Hopf_focus_isochrones.jpg")
    plt.show()

    #compute PRC
    #reference points
    phase_list_0, PRC_list_0 = compute_PRC(0, h, N, beta, pulse, isochrones_real_func, isochrones_im_func, n_points)
    phase_list, PRC_list = compute_PRC(T, h, N, beta, pulse, isochrones_real_func, isochrones_im_func, n_points)
    plt.plot(phase_list_0, PRC_list_0, "r-")
    plt.plot(phase_list, PRC_list, "b+")
    plt.title("Hopf focus (below bifurcation): PRC with an initial E-shift of 0.1, after one period")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.legend(["Reference PRC", "PRC at T"])
    plt.savefig("./Data/hopf/output/Hopf_focus_PRC.jpg")
    plt.show()
    
    with open('./Data/hopf/output/Hopf_focus_phase_0T.txt', 'w') as output:
        for i in range(len(phase_list_0)):
            content = str(phase_list_0[i])
            output.write(content + " ")
    with open('./Data/hopf/output/Hopf_focus_phase_1T.txt', 'w') as output:
        for i in range(len(phase_list)):
            content = str(phase_list[i])
            output.write(content + " ")
    with open('./Data/hopf/output/Hopf_focus_shift_0T.txt', 'w') as output:
        for i in range(len(PRC_list_0)):
            content = str(PRC_list_0[i])
            output.write(content + " ")
    with open('./Data/hopf/output/Hopf_focus_shift_1T.txt', 'w') as output:
        for i in range(len(PRC_list)):
            content = str(PRC_list[i])
            output.write(content + " ")

    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()