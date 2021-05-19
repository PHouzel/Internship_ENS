#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:01:48 2021

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

@jit(float64[:](float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64),nopython=True)
def WilsonCowan(z, pulse, tau_e, tau_i, gee, gei, gie, gii, ae, ai):
    E, I = z
    Se, Si = pulse
    a_1 = gee*E + gei*I + Se + ae
    a_2 = gie*E + gii*I + Si + ai
    fE = (1/tau_e)*(-E + (1/(1 + exp(-a_1))))
    fI = (1/tau_i)*(-I + (1/(1 + exp(-a_2))))
    return array([fE,fI])

@jit(float64[:](float64[:]), nopython=True)
def noiseFunction(z):
    """
    Added noise amplitude
    """
    #D = 0.1
    #gx = sqrt(2*D)
    #gy = sqrt(2*D)
    gx = 0.1
    gy = 0.1
    return array([gx, gy])

@jit((float64[:])(float64[:], float64, float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64), nopython=True)
def update_z(z, h, eta, pulse, tau_e, tau_i, gee, gei, gie, gii, ae, ai):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*WilsonCowan(z, pulse, tau_e, tau_i, gee, gei, gie, gii, ae, ai) + eta*noiseFunction(z)
    z_updated = z + 0.5*(WilsonCowan(z, pulse, tau_e, tau_i, gee, gei, gie, gii, ae, ai) + WilsonCowan(pred, pulse, tau_e, tau_i, gee, gei, gie, gii, ae, ai))*h + 0.5*(noiseFunction(z) + noiseFunction(pred))*eta
    return z_updated

def EulerInteg(z, h, numsteps, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai):
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
        eta = random.normal(loc=0, scale=sqrt(sigma), size=2)

        z_updated = update_z(z, h, eta, pulse, tau_e, tau_i, gee, gei, gie, gii, ae, ai)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
    return time_list, trajectory

def compute_mean_phase(initial_z, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func):
    #V2: get cosinus and sinus, and take phase from there
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_real = 0
    mean_im = 0
    mean_trajectory = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, numsteps, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai)
        #Compute phase of last position
        real = real_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        im = im_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        mean_real += real
        mean_im += im
        mean_trajectory = mean_trajectory + trajectory
    mean_phase = atan2(mean_im, mean_real)
    #if mean_phase < 0: mean_phase += 2*pi
    mean_trajectory = mean_trajectory/N
    return mean_phase, mean_trajectory

def compute_single_shift(initial_z, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func):
    shifted_z = initial_z + pulse

    shifted_mean_phase, shifted_mean_trajectory = compute_mean_phase(shifted_z, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
    initial_mean_phase, initial_mean_trajectory = compute_mean_phase(initial_z, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
    
    shift = (shifted_mean_phase - initial_mean_phase)
    if shift < 0: shift += 2*pi
    if shift > pi: shift += -2*pi
    return shift, initial_mean_trajectory, shifted_mean_trajectory

def compute_PRC(limit_cycle_data, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func):
    all_x = limit_cycle_data[:,0]
    all_y = limit_cycle_data[:,1]
    PRC_list = []
    phase_list2 = zeros(len(all_x))
    for i in tqdm(range(len(all_x))):
        z = array([all_x[i], all_y[i]])
        shift, _, _ = compute_single_shift(z, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
        PRC_list.append(shift)
        real = real_isochrone_func(all_x[i], all_y[i])
        im = im_isochrone_func(all_x[i], all_y[i])
        phase_list2[i] = mod(arctan2(im, real),2*pi)
    return phase_list2, array(PRC_list)

def main():
    startTime = datetime.now()
    
    #setting up grid for the phase spaces
    yp = 0.75; ym = -0.25
    xp = 1.0;  xm = -0.25
    y = linspace(ym, yp, 100)
    x = linspace(xm, xp, 100)
    
    #setting up integration parameters
    h = 0.01 #timestep; 
    N = 100 #number of trajectories over which we average the phase
    T = 33.126711029860324
    
    #First set of parameters
    gee = 10; gei = -10; gie = 12; gii = -10
    tau_e = 3; tau_i = 8
    ae = -2; ai = -3.5
    sigma = 0.13
    
    #load data
    isochrone = loadtxt('./Data/WilsonCowan/isocronesD0.1')
    isochrones_real = loadtxt('./Data/WilsonCowan/realValuesD0.1')
    isochrones_im = loadtxt('./Data/WilsonCowan/imagValuesD0.1')
    limit_cycle_data = loadtxt('./Data/WilsonCowan/limCycleData0.1')
    #interpolate
    isochrones_real_func = interpolate.interp2d(x, y, isochrones_real, kind = 'cubic')
    isochrones_im_func = interpolate.interp2d(x, y, isochrones_im, kind = 'cubic')
    
    S0 = 1; theta = 85*(pi/180)
    #pulse = array([S0*cos(theta), S0*sin(theta)])
    pulse = array([-0.1, 0.1])
    
    phase_list_0, PRC_list_0 = compute_PRC(limit_cycle_data, 0, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, isochrones_real_func, isochrones_im_func)
    #phase_list, PRC_list = compute_PRC(limit_cycle_data, T, h, N, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai, isochrones_real_func, isochrones_im_func)
    
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    #time_list, trajectory = EulerInteg(array([limit_cycle_data[:,0][0] + pulse[0], limit_cycle_data[:,0][1]+ pulse[1]]), h, 1000, pulse, sigma, tau_e, tau_i, gee, gei, gie, gii, ae, ai)
    #plt.plot(trajectory[:,0], trajectory[:,1], color='blue')
    plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
    plt.plot(limit_cycle_data[:,0] + pulse[0], limit_cycle_data[:,1] + pulse[1], color = 'gray')
    plt.legend(["Typical trajectory", "Initial points", "Shifted points"])
    plt.title('SNIC LC (above bifurcation): Isochrones from data')
    plt.xlabel("x")
    plt.ylabel("y")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.xlim = ([xm, xp])
    plt.ylim = ([ym, yp])
    plt.show()
    

    #plt.plot(phase_list_0, PRC_list_0, "r+")
    #plt.plot(phase_list, PRC_list, "b.")
    #plt.title("SNIC LC (above bifurcation): PRC with an initial E-shift of " + str(pulse[0]) + ", after one period")
    #plt.xlabel("$\Theta$")
    #plt.ylabel("$\Delta \Theta$")
    #plt.legend(["Reference PRC", "PRC at T"])
    #plt.savefig("./Data/snic/output/SNIC_LC_PRC" + str(pulse[0]) + ".jpg")
    #plt.show()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    