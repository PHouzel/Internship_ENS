#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:18:55 2021

@author: pierrehouzelstein
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from numba import jit, float64, int64, types
from numpy import array, sqrt, zeros, random, pi, linspace, cos, sin, loadtxt, arctan, arctan2, mod, exp
from math import atan2
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

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

@jit(float64[:](float64[:], float64), nopython=True)
def noiseFunction(eta, amplitude):
    """
    Added noise
    """
    gx = amplitude*eta[0]
    gy = amplitude*eta[1]
    return array([gx, gy])

@jit((float64[:])(float64[:], float64, float64[:], float64, float64, float64),nopython=True)
def update_z(z, h, eta, amplitude, beta_1, beta_2):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*AR2_model(z, beta_1, beta_2) + noiseFunction(eta, amplitude)
    
    z_updated = z + 0.5*( AR2_model(z, beta_1, beta_2)\
                         + AR2_model(pred, beta_1, beta_2))*h\
                         + noiseFunction(eta, amplitude)
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64[:], float64, float64,float64,\
                                             float64, int64), nopython=True)
def EulerInteg(z, h, amplitude, beta_1, beta_2, numsteps):
    """
    Integration routine
    """
    #Initial time
    time = 0.
    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2))
    time_list[0] = time
    trajectory[0] = z
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(h), size=2)

        z_updated = update_z(z, h, eta, amplitude, beta_1, beta_2)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
    return time_list, trajectory

def compute_single_mean_phase(initial_z, T, h, N, noise_amplitude, beta_1, beta_2,\
                              real_isochrone_func, im_isochrone_func):
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_real = 0
    mean_im = 0
    mean_trajectory = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, noise_amplitude, beta_1, beta_2, numsteps)
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

def compute_mean_phases(limit_cycle_data, T, h, N, noise_amplitude, beta_1, beta_2,\
                               real_isochrone_func, im_isochrone_func):

    all_x = limit_cycle_data[:,0]
    all_y = limit_cycle_data[:,1]
    phase_list = zeros(len(all_x))

    for i in tqdm(range(len(all_x))):
        z = array([all_x[i], all_y[i]])
        mean_phase,_ = compute_single_mean_phase(z, T, h, N, noise_amplitude, beta_1, beta_2,\
                              real_isochrone_func, im_isochrone_func)
        phase_list[i] = mean_phase

    return phase_list

def get_shifted_points(limit_cycle_data, h, pulse, beta_1, beta_2):
    
    positions = []
    start_x = limit_cycle_data[:,0]
    start_y = limit_cycle_data[:,1]
    for i in range(len(start_x)):
        z = array([start_x[i], start_y[i]])
        z += pulse
        positions.append(z)
    return array(positions)

def main():
    startTime = datetime.now()
    
    #load data
    isochrone = loadtxt('./data/AR2/isocronesD0.01')
    isochrones_real = loadtxt('./data/AR2/realValuesD0.01')
    isochrones_im = loadtxt('./data/AR2/imagValuesD0.01')
    limit_cycle_data = loadtxt('./data/AR2/limCycleData0.01')
    
    
    
    iso_0 = loadtxt("./data/AR2/LC/isoData0.0")
    iso_001 = loadtxt("./data/AR2/LC/isoData0.001")
    iso_002 = loadtxt("./data/AR2/LC/isoData0.002")
    iso_0005 = loadtxt("./data/AR2/LC/isoData0.0005")
    iso_0001 = loadtxt("./data/AR2/LC/isoData-0.0001")
    iso_0002 = loadtxt("./data/AR2/LC/isoData-0.0002")
    iso_00024 = loadtxt("./data/AR2/LC/isoData-0.00024")
    
    limit_cycle_data = iso_0001
    
    #interpolate
    #setting up grid for the phase spaces
    yp = 0.5; ym = -0.5
    xp = 0.5; xm = -0.5
    y = linspace(ym, yp, 80+1)
    x = linspace(xm, xp, 80+1)
    real_isochrone_func = interpolate.interp2d(\
                    x, y, isochrones_real, kind = 'cubic')
    im_isochrone_func = interpolate.interp2d(\
                    x, y, isochrones_im, kind = 'cubic')
    
    #Constants parameters for all simulations
    h=0.01; N = 1000 #average each trajectory over N periods to get mean phase
    beta_1, beta_2 = [-0.9606, 1.8188]
    T = 16.70847735002074
    noise_amplitude = 0.01
    pulse = array([0.025, 0.025])
    
    limit_cycle_shifted = get_shifted_points(limit_cycle_data, h, pulse, beta_1, beta_2)
    
    #Plot one example
    #z = array([limit_cycle_shifted [0][0], limit_cycle_shifted [0][1]])
    numsteps = 15000
    #time_list, trajectory = EulerInteg(z, h, noise_amplitude, beta_1, beta_2, numsteps)
        
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    
    plt.plot(iso_0[:,0], iso_0[:,1], color = 'black')
    plt.plot(iso_0[:,0] + pulse[0], iso_0[:,1] + pulse[1], color = 'grey')
    
    plt.plot(iso_001[:,0], iso_001[:,1], color = 'black')
    plt.plot(iso_001[:,0] + pulse[0], iso_001[:,1] + pulse[1], color = 'grey')
    
    plt.plot(iso_002[:,0], iso_002[:,1], color = 'black')
    plt.plot(iso_002[:,0] + pulse[0], iso_002[:,1] + pulse[1], color = 'grey')
    
    plt.plot(iso_0005[:,0], iso_0005[:,1], color = 'black')
    plt.plot(iso_0005[:,0] + pulse[0], iso_0005[:,1] + pulse[1], color = 'grey')
    
    plt.plot(iso_0001[:,0], iso_0001[:,1], color = 'black')
    plt.plot(iso_0001[:,0] + pulse[0], iso_0001[:,1] + pulse[1], color = 'grey')
    
    plt.plot(iso_0002[:,0], iso_0002[:,1], color = 'black')
    plt.plot(iso_0002[:,0] + pulse[0], iso_0002[:,1] + pulse[1], color = 'grey')
    
    plt.plot(iso_00024[:,0], iso_00024[:,1], color = 'black')
    plt.plot(iso_00024[:,0] + pulse[0], iso_00024[:,1] + pulse[1], color = 'grey')
    #plt.plot(trajectory[:,0], trajectory[:,1], color='white')
    #plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
    #plt.plot(limit_cycle_shifted[:,0], limit_cycle_shifted[:,1], color = 'gray')
    plt.legend(["Initial limit cycles", "Shifted limit cycles"])
    plt.title('AR2 model: phase plane')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.xlim = ([xm, xp])
    plt.ylim = ([ym, yp])
    plt.savefig("./data/AR2/output/AR2_phase_space_pulse=" + str(pulse[0]) + ".jpg")
    plt.show()
    
    #Get PRC
    initial_theta = limit_cycle_data[:,2]
    theta_perturbed = compute_mean_phases(limit_cycle_shifted, T, h, N, noise_amplitude, beta_1, beta_2,\
                               real_isochrone_func, im_isochrone_func)
    theta_free = initial_theta #only if integrating over t = n*T; loss in precision but faster computing
    #theta_free = compute_mean_phases(limit_cycle_data, T, h, N, noise_amplitude, 0*pulse, sigma,\
                #tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
    
    plt.plot(initial_theta, theta_perturbed, "r+")
    plt.plot(initial_theta, theta_free, "b+")
    plt.plot(initial_theta, theta_perturbed-theta_free, "g+")
    plt.title("Phases and PRC of the AR2 model")
    plt.legend(["Final phases of perturbed points", "Final phases of unperturbed points", "PRC"])
    plt.show()
    
    with open('./data/AR2/output/AR2_pert_pulse=' +  str(pulse[0]) + '.txt', 'w') as output:
        for i in range(len(theta_perturbed)):
            content = theta_perturbed[i]
            content = str(content)
            output.write(content + " ")    
    with open('./data/AR2/output/AR2_free_phase_pulse=' +  str(pulse[0])+ '.txt', 'w') as output:
        for i in range(len(theta_free)):
            content = theta_free[i]
            content = str(content)
            output.write(content + " ")    
    with open('./data/AR2/output/AR2_PRC_pulse=' +  str(pulse[0]) + '.txt', 'w') as output:
        for i in range(len(theta_perturbed)):
            content = theta_perturbed[i] - theta_free[i]
            content = str(content)
            output.write(content + " ")
    with open('./data/AR2/output/AR2_shifted_LC_pulse=' +  str(pulse[0]) + '.txt', 'w') as output:
        for i in range(len(limit_cycle_shifted)):
            content_1 = str(limit_cycle_shifted[i][0])
            content_2 = str(limit_cycle_shifted[i][1])
            output.write(content_1 + " " + content_2  + "\n")
            
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    
if __name__ == '__main__':
    main()
