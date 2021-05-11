#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:01:47 2021

@author: pierrehouzelstein
"""

from numpy import zeros, random, sqrt, loadtxt, linspace, array
from SNIC import update_z, compute_mean_phase, EulerInteg
from math import atan2
from scipy import interpolate
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

def single_phase_evol(z, h, beta, m, numsteps, real_isochrone_func, im_isochrone_func):
    """
    Integration routine
    """
    #Initial time
    time = 0.
    
    #Setting up lists
    time_list = zeros(numsteps+1)
    phase_list = zeros(numsteps+1)
    time_list[0] = time
    
    x = real_isochrone_func(z[0], z[1])
    y = im_isochrone_func(z[0], z[1])
    
    phase_list[0] = atan2(y, x)
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(h), size=2)

        z_updated = update_z(z, h, eta, beta, m)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        x = real_isochrone_func(z[0], z[1])
        y = im_isochrone_func(z[0], z[1])
        phase_list[i+1] = atan2(y, x)
        
    return time_list, phase_list

def mean_phase_evol(z, h, beta, m, numsteps, N, real_isochrone_func, im_isochrone_func):
    mean_phase = zeros(numsteps+1)
    for i in tqdm(range(N)):
        time_list, phase_list = single_phase_evol(z, h, beta, m, numsteps, real_isochrone_func, im_isochrone_func)
        mean_phase = mean_phase + phase_list
    mean_phase = mean_phase/N
    return time_list, mean_phase

        
def main():
    startTime = datetime.now()
    
    #setting up grid for the phase spaces
    yp = 1.5; ym = -1.5
    xp = 1.5; xm = -1.5
    y = linspace(ym, yp, 100)
    x = linspace(xm, xp, 100)
    
    #setting up integration parameters
    h = 0.01 #timestep; 

    #case 1: LC (above bifurcation)
    beta = 1; m = 1.1; T = 110
    
    z = array([1.0, 1.0])
    numsteps = 10000
    N =100
    
    isochrone = loadtxt('./Data/snic/above/data/isocronesD0.01125')
    isochrones_real = loadtxt('./Data/snic/above/data/realValuesD0.01125')
    isochrones_im = loadtxt('./Data/snic/above/data/imagValuesD0.01125')
    #interpolate
    isochrone_func = interpolate.interp2d(x, y, isochrone, kind = 'cubic')
    isochrones_real_func = interpolate.interp2d(x, y, isochrones_real, kind = 'cubic')
    isochrones_im_func = interpolate.interp2d(x, y, isochrones_im, kind = 'cubic')
 
    time_list, phase_list = single_phase_evol(z, h, beta, m, numsteps, isochrones_real_func, isochrones_im_func)
    
    plt.plot(time_list, phase_list, "b+")
    plt.title("SNIC: Evolution of a single phase $\Theta$ as a function of time")
    plt.xlabel("$t$")
    plt.ylabel("$\Theta$")
    plt.savefig("./Output_files/SNIC_single_phase.jpg")
    plt.show()
    
    time_list, mean_phase = mean_phase_evol(z, h, beta, m, numsteps, N, isochrones_real_func, isochrones_im_func)
    
    plt.plot(time_list, mean_phase, "b+")
    plt.title("SNIC: Evolution of the mean phase $\Theta_{mean}$ as a function of time")
    plt.xlabel("$t$")
    plt.ylabel("$\Theta_{mean}$")
    plt.savefig("./Output_files/SNIC_mean_phase.jpg")
    plt.show()

    time_list, trajectory = EulerInteg(z, h, beta, m, numsteps)
    plt.plot(trajectory[:,0], trajectory[:,1], "r+")
    plt.title("SNIC: Single trajectory")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("./Output_files/SNIC_single_traj.jpg")
    plt.show()
    
    time_list, trajectory = compute_mean_phase(z, T, h, N, beta, m, isochrones_real_func, isochrones_im_func)
    plt.plot(trajectory[:,0], trajectory[:,1], "r+")
    plt.title("SNIC: Single trajectory")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("./Output_files/SNIC_mean_traj.jpg")
    plt.show()

    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
if __name__ == '__main__':
    main()
    