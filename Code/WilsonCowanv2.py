#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:00:22 2021

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

@jit(float64[:](float64[:], float64, float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64),nopython=True)
def WilsonCowan(z, t, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai,):
    E, I = z
    Se, Si = pulse*exp(-t/tau_stim)
    a_1 = gee*E + gei*I + Se + ae
    a_2 = gie*E + gii*I + Si + ai
    fE = (1/tau_e)*(-E + (1/(1 + exp(-a_1))))
    fI = (1/tau_i)*(-I + (1/(1 + exp(-a_2))))
    return array([fE,fI])

@jit(float64[:](float64[:], float64, float64, float64), nopython=True)
def noiseFunction(eta, amplitude, tau_e, tau_i):
    """
    Added noise amplitude
    """
    #D = 0.1
    #gx = sqrt(2*D)
    #gy = sqrt(2*D)
    gx = (amplitude/tau_e)*eta[0]
    gy = (amplitude/tau_i)*eta[1]
    return array([gx, gy])

@jit((float64[:])(float64[:], float64, float64, float64[:], float64, float64[:], float64, float64,\
                  float64, float64, float64, float64, float64, float64, float64), nopython=True)
def update_z(z, t, h, eta, amplitude, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai):
    """
    Gives z at t + h from z at t
    """
    pred = z + h*WilsonCowan(z, t, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai)\
                         + noiseFunction(eta, amplitude,tau_e, tau_i)*sqrt(h)
    z_updated = z + 0.5*(WilsonCowan(z, t, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai)\
                         + WilsonCowan(pred, t, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai))*h\
                         + noiseFunction(eta, amplitude, tau_e, tau_i)*sqrt(h)
    return z_updated

@jit(types.Tuple((float64[:], float64[:,:]))(float64[:], float64, int64, float64, float64[:],\
                                             float64, float64, float64, float64, float64,\
                                             float64, float64, float64, float64, float64),\
                                             nopython=True)
def EulerInteg(z, h, numsteps, noise_amplitude, pulse, sigma, tau_e, tau_i,\
               tau_stim, gee, gei, gie, gii, ae, ai):
    """
    Integration routine
    """
    #Initial time
    time = 0.
    t = time
    #Setting up lists
    time_list = zeros(numsteps+1)
    trajectory = zeros((numsteps+1, 2))
    time_list[0] = time
    trajectory[0] = z
    
    for i in range(numsteps):
        eta = random.normal(loc=0, scale=sqrt(sigma), size=2)

        z_updated = update_z(z, t, h, eta, noise_amplitude, pulse, tau_e,\
                             tau_i, tau_stim, gee, gei, gie, gii, ae, ai)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
        t = time
    return time_list, trajectory

def get_shifted_points(limit_cycle_data, h, pulse, sigma, tau_e, tau_i, tau_stim,\
                       gee, gei, gie, gii, ae, ai):
    
    numsteps = int(tau_stim//h)
    positions = []
    start_x = limit_cycle_data[:,0]
    start_y = limit_cycle_data[:,1]
    for i in range(len(start_x)):
        z = array([start_x[i], start_y[i]])
        _, trajectory = EulerInteg(z, h, numsteps, 0, pulse, sigma, tau_e, tau_i,\
                                   tau_stim, gee, gei, gie, gii, ae, ai)
        positions.append(trajectory[-1])
    return array(positions)

def compute_mean_phase(initial_z, T, h, N, noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim,\
                       gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func):
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_real = 0
    mean_im = 0
    mean_trajectory = 0
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, numsteps, noise_amplitude, pulse, sigma, tau_e,\
                                           tau_i, tau_stim, gee, gei, gie, gii, ae, ai)
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

def compute_single_shift(initial_z, shifted_z, T, h, N,  noise_amplitude, pulse, sigma, tau_e, tau_i,\
                         tau_stim, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func):

    shifted_mean_phase, shifted_mean_trajectory = compute_mean_phase(shifted_z, T, h, N,\
                        noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim, gee, gei, gie, gii,\
                        ae, ai, real_isochrone_func, im_isochrone_func)
    
    initial_mean_phase, initial_mean_trajectory = compute_mean_phase(initial_z, T, h, N,\
                        noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim, gee, gei, gie, gii,\
                        ae, ai, real_isochrone_func, im_isochrone_func)
    
    shift = shifted_mean_phase - initial_mean_phase
    
    if shift < -pi: shift += 2*pi
    if shift > pi: shift += -2*pi
    
    return shift, initial_mean_trajectory, shifted_mean_trajectory

def compute_PRC(limit_cycle_data, shifted_cycle_data, T, h, N, noise_amplitude, pulse, sigma,\
                tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func):

    all_x = limit_cycle_data[:,0]
    all_y = limit_cycle_data[:,1]
    shifted_x = shifted_cycle_data[:,0]
    shifted_y = shifted_cycle_data[:,1]
    phase_list = limit_cycle_data[:,2]
    PRC_list = []
    #phase_list2 = zeros(len(all_x))
    
    
    for i in tqdm(range(len(all_x))):
        
        z = array([all_x[i], all_y[i]])
        shifted_z = array([shifted_x[i], shifted_y[i]])
        
        shift, _, _ = compute_single_shift(z, shifted_z, T, h, N,  noise_amplitude,\
                                           pulse, sigma, tau_e, tau_i, tau_stim, gee, gei, gie,\
                                           gii, ae, ai, real_isochrone_func, im_isochrone_func)

        PRC_list.append(shift)
        #real = real_isochrone_func(all_x[i], all_y[i])
        #im = im_isochrone_func(all_x[i], all_y[i])
        #phase_list2[i] = mod(arctan2(im, real), 2*pi)

    return phase_list, array(PRC_list)


def main():
    startTime = datetime.now()
    
    #load data
    isochrone = loadtxt('./Data/WilsonCowan/isocronesD0.1')
    isochrones_real = loadtxt('./Data/WilsonCowan/realValuesD0.1')
    isochrones_im = loadtxt('./Data/WilsonCowan/imagValuesD0.1')
    limit_cycle_data = loadtxt('./Data/WilsonCowan/limCycleData0.1')
    
    #interpolate
    #setting up grid for the phase spaces
    yp = 0.75; ym = -0.25
    xp = 1.0;  xm = -0.25
    y = linspace(ym, yp, 100)
    x = linspace(xm, xp, 100)
    real_isochrone_func = interpolate.interp2d(\
                    x, y, isochrones_real, kind = 'cubic')
    im_isochrone_func = interpolate.interp2d(\
                    x, y, isochrones_im, kind = 'cubic')
    
    #Constants parameters for all simulations
    h=0.01; N = 10 #average each trajectory over N periods to get mean phase
    gee = 10.; gei = -10.; gie = 12.; gii = -10.
    tau_e = 3.; tau_i = 8.; tau_stim = 6.
    ae = -2.; ai = -3.5
    noise_amplitude = 0.1
    
    #Trajectory when no perturbation
    z = array([limit_cycle_data[0][0], limit_cycle_data[0][1]])
    numsteps = 10000; sigma = 0.13
    pulse =array([0., 0.])
    time_list, trajectory = EulerInteg(z, h, numsteps, noise_amplitude, pulse, sigma,\
                                       tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai)
    
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    plt.plot(trajectory[:,0], trajectory[:,1], color='white')
    plt.show()
    
    plt.plot(time_list, trajectory[:,0], "k-")
    plt.plot(time_list, trajectory[:,1], "r-")
    plt.show()
    #Simulation #1 (fig 2e): parameters
    T = 33.126711029860324; T = 100
    sigma = 0.13
    S0 = 1.; theta = 85*(pi/180)
    pulse = array([S0*cos(theta), S0*sin(theta)])
    
    #get shift
    shifted_positions = get_shifted_points(limit_cycle_data, h, pulse, sigma, tau_e, tau_i, tau_stim,\
                                           gee, gei, gie, gii, ae, ai)

    #get example trajectory: parameters
    z = array([shifted_positions[0][0], shifted_positions[0][1]])
    pulse = array([0., 0.]) #impulsion has waded by the time the system is completely shifted
    numsteps = 10000
    
    time_list, trajectory = EulerInteg(z, h, numsteps, noise_amplitude, pulse, sigma,\
                                       tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai)

    plt.plot(time_list, trajectory[:,0], "k-")
    plt.plot(time_list, trajectory[:,1], "r-")
    plt.show()
    
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    plt.plot(trajectory[:,0], trajectory[:,1], color='white')
    plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
    plt.plot(shifted_positions[:,0], shifted_positions[:,1], color = 'grey')
    plt.legend(["Typical trajectory", "Initial points", "Shifted points"])
    plt.title('Wilson-Cowan: Computed Isochrones')
    plt.xlabel("x")
    plt.ylabel("y")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.xlim = ([xm, xp])
    plt.ylim = ([ym, yp])
    plt.show()
    
    #compute PRC
    phase_list_0, PRC_list_0 = compute_PRC(limit_cycle_data, shifted_positions, 0, h, N,\
                                noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim,\
                                gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
        
    phase_list, PRC_list = compute_PRC(limit_cycle_data, shifted_positions, T, h, N,\
                                noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim,\
                                gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
    
    plt.plot(phase_list_0, PRC_list_0, "r+")
    plt.plot(phase_list, PRC_list, "b.")
    plt.title("Wilson-Cowan: PRC after one period")
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta \Theta$")
    plt.legend(["Reference PRC", "PRC at T"])
    #plt.savefig("./Data/snic/output/SNIC_LC_PRC" + str(pulse[0]) + ".jpg")
    plt.show()
        
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    
if __name__ == '__main__':
    main()