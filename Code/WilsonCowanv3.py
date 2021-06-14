#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:24:20 2021

@author: pierrehouzelstein
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from numba import jit, float64, int64, types
from numpy import array, sqrt, zeros, random, pi, linspace, cos, sin, loadtxt, arctan, arctan2, mod, exp, sum, square
from math import atan2
from scipy import interpolate
from scipy.integrate import trapz
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
                         + noiseFunction(eta, amplitude,tau_e, tau_i)
    z_updated = z + 0.5*(WilsonCowan(z, t, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai)\
                         + WilsonCowan(pred, t, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai))*h\
                         + noiseFunction(eta, amplitude, tau_e, tau_i)
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
        eta = random.normal(loc=0, scale=sqrt(h), size=2)

        z_updated = update_z(z, t, h, eta, noise_amplitude, pulse, tau_e,\
                             tau_i, tau_stim, gee, gei, gie, gii, ae, ai)
        
        time += h
        z = z_updated
        # i+1 because 0 is set at initial conditions
        time_list[i+1] = time
        trajectory[i+1] = z
        t = time
    return time_list, trajectory

def compute_single_mean_phase(initial_z, T, h, N, noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim,\
                       gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func, grad_x_func, grad_y_func):
    #Compute over one period
    numsteps = int(T//h)
    #Get mean phase at T
    mean_real = 0
    mean_im = 0
    mean_trajectory = 0
    mean_variance_x = 0
    mean_variance_y = 0
    mean_variance_stoch = 0
    phase_list = zeros(N)
    for i in range(N):
        time_list, trajectory = EulerInteg(initial_z, h, numsteps, noise_amplitude, pulse, sigma, tau_e,\
                                           tau_i, tau_stim, gee, gei, gie, gii, ae, ai)

        
        #Compute phase of last position
        real = real_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        im = im_isochrone_func(trajectory[-1][0], trajectory[-1][1])
        mean_real += real
        mean_im += im
        mean_trajectory = mean_trajectory + trajectory
        
        #Variance
        #Theoretical
        phase_list[i]= atan2(im, real)
        if phase_list[i]< 0: phase_list[i] += 2*pi
        if phase_list[i]< 0: phase_list[i] += 2*pi

        #Separate x and y trajectories
        x_traj = trajectory[:,0]
        y_traj = trajectory[:,1]
        #Get the derivative of each point of the trajectory
        for i in range(len(x_traj)):
            x_traj[i]= grad_x_func(x_traj[i], y_traj[i])
            y_traj[i]= grad_y_func(x_traj[i], y_traj[i])
        #Multiply by D and square each element
        #noise : sqrt(2D) = noise amplitude so D = (noise amplitude)^2/2
        D = ((noise_amplitude)**2)/2
        x_traj = square(D*x_traj)
        y_traj = square(D*y_traj)
        #Integrate
        mean_variance_x += trapz(x_traj, dx = h)
        mean_variance_y += trapz(y_traj, dx = h)

    mean_phase = atan2(mean_im, mean_real)
    if mean_phase < 0: mean_phase += 2*pi
    mean_trajectory = mean_trajectory/N
    
    mean_variance_x = mean_variance_x/N
    mean_variance_y =mean_variance_y/N
    mean_variance_stoch = mean_variance_x + mean_variance_y
    
    mean_variance_theo = sum(square(phase_list - mean_phase))/N
    
    return mean_phase, mean_trajectory, mean_variance_stoch, mean_variance_theo

def compute_mean_phases(limit_cycle_data, T, h, N, noise_amplitude, pulse, sigma,\
                tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func, grad_x_func, grad_y_func):

    all_x = limit_cycle_data[:,0]
    all_y = limit_cycle_data[:,1]
    phase_list = zeros(len(all_x))
    variance_list_stoch = zeros(len(all_x))
    variance_list_theo = zeros(len(all_x))

    for i in tqdm(range(len(all_x))):
        z = array([all_x[i], all_y[i]])
        mean_phase,_, mean_variance_stoch, mean_variance_theo = compute_single_mean_phase(z, T, h, N, noise_amplitude, pulse, sigma, tau_e, tau_i, tau_stim,\
                gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func, grad_x_func, grad_y_func)
        phase_list[i] = mean_phase
        variance_list_stoch[i] = mean_variance_stoch
        variance_list_theo[i] = mean_variance_theo

    return phase_list, variance_list_stoch, variance_list_theo

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

def main():
    startTime = datetime.now()
    
    #load data
    isochrone = loadtxt('./data/WilsonCowan/isocronesD0.1')
    isochrones_real = loadtxt('./data/WilsonCowan/realValuesD0.1')
    isochrones_im = loadtxt('./data/WilsonCowan/imagValuesD0.1')
    limit_cycle_data = loadtxt('./data/WilsonCowan/limCycleData0.1')[::5]
    grad_x = loadtxt('./data/WilsonCowan/gradTheta_x.txt')
    grad_y = loadtxt('./data/WilsonCowan/gradTheta_y.txt')
    
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
        
    #Get grad functions for variance
    grad_x_func = interpolate.interp2d(\
                    x, y, grad_x, kind = 'cubic')
    grad_y_func = interpolate.interp2d(\
                    x, y, grad_y, kind = 'cubic')
    
    #Constants parameters for all simulations
    h=0.01; N = 1000 #average each trajectory over N periods to get mean phase
    gee = 10.; gei = -10.; gie = 12.; gii = -10.
    tau_e = 3.; tau_i = 8.; tau_stim = 6.
    ae = -2.; ai = -3.5
    noise_amplitude = 0.1
    
    #Simulation #1 (fig 2e): parameters
    T = 33.126711029860324
    #Fig.2d:σ= 0.13, θ=85°,S0 =1, 1.5, 4 and 10
    #Fig. 3b: σ = 0.2–0.5, θ = 80°, S0 = 0.4, 1.6, 3 and 12
    #fig 3d: σ = 0.4–0.6, θ = 60°, S0 = 0.5, 2.2, 4.1 and 8
    sigma = 0.4
    S0 = 1.6; theta = 80*(pi/180)
    #S0 = 2.2; theta = 60*(pi/180)
    pulse = array([S0*cos(theta), S0*sin(theta)])
    
    #get shift
    shifted_positions = get_shifted_points(limit_cycle_data, h, pulse, sigma, tau_e, tau_i, tau_stim,\
                                           gee, gei, gie, gii, ae, ai)
    #Plot one example
    z = array([limit_cycle_data[0][0], limit_cycle_data[0][1]])
    numsteps = 10000
    time_list, trajectory = EulerInteg(z, h, numsteps, noise_amplitude, pulse, sigma,\
                                       tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai)
    
    plt.pcolormesh(x, y, isochrone, cmap='gist_rainbow')
    plt.plot(trajectory[:,0], trajectory[:,1], color='white')
    plt.plot(limit_cycle_data[:,0], limit_cycle_data[:,1], color = 'black')
    plt.plot(shifted_positions[:,0], shifted_positions[:,1], color = 'grey')
    plt.legend(["Typical trajectory", "Initial points", "Shifted points"])
    plt.title('Wilson-Cowan: phase plane')
    plt.xlabel("E")
    plt.ylabel("I")
    cbar=plt.colorbar(label="$\Theta$(x)", orientation="vertical")
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
    plt.xlim = ([xm, xp])
    plt.ylim = ([ym, yp])
    plt.savefig("./data/WilsonCowan/output/WilsonCowan_phase_space_S=" + str(S0) + ".jpg")
    plt.show()
    
    
    #Get PRC
    initial_theta = limit_cycle_data[:,2]
    theta_perturbed, mean_trajectory, variance_list_perturbed_stoch,\
        variance_list_perturbed_theo = compute_mean_phases(limit_cycle_data, T, h, N, noise_amplitude, pulse, sigma,\
        tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func, grad_x_func, grad_y_func)
    theta_free = initial_theta #only if integrating over t = n*T; loss in precision but faster computing
    #theta_free, variance_list_free = compute_mean_phases(limit_cycle_data, T, h, N, noise_amplitude, 0*pulse, sigma,\
                #tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, real_isochrone_func, im_isochrone_func)
    
    #plt.plot(initial_theta, variance_list_perturbed)
    #plt.errorbar(initial_theta, theta_perturbed, yerr = variance_list_perturbed)
    plt.plot(initial_theta, theta_perturbed, "b+")
    plt.fill_between(initial_theta, theta_perturbed + sqrt(variance_list_perturbed_stoch), theta_perturbed - sqrt(variance_list_perturbed_stoch))
    plt.plot(initial_theta, theta_free, "r+")
    plt.plot(initial_theta, theta_perturbed-theta_free, "g+")
    plt.title("Phases and PRC of the Wilson-Cowan model")
    plt.legend(["Final phases of perturbed points", "Final phases of unperturbed points", "PRC"])
    plt.savefig("./data/WilsonCowan/output/WilsonCowan_PRC_S=" + str(S0) + ".jpg")
    plt.show()
    
    plt.plot(initial_theta, theta_perturbed, "b+")
    plt.fill_between(initial_theta, theta_perturbed + sqrt(variance_list_perturbed_theo), theta_perturbed- sqrt(variance_list_perturbed_theo))
    plt.plot(initial_theta, theta_free, "r+")
    plt.plot(initial_theta, theta_perturbed-theta_free, "g+")
    plt.title("Phases and PRC of the Wilson-Cowan model")
    plt.legend(["Final phases of perturbed points", "Final phases of unperturbed points", "PRC"])
    plt.savefig("./data/WilsonCowan/output/WilsonCowan_PRC_S=" + str(S0) + "_var_theo.jpg")
    plt.show()

    with open('./data/WilsonCowan/output/WilsonCowan_pert_phase_S=' +  str(S0) + '.txt', 'w') as output:
        for i in range(len(theta_perturbed)):
            content = theta_perturbed[i]
            content = str(content)
            output.write(content + " ")    
    with open('./data/WilsonCowan/output/WilsonCowan_free_phase_S=' +  str(S0) + '.txt', 'w') as output:
        for i in range(len(theta_free)):
            content = theta_free[i]
            content = str(content)
            output.write(content + " ")    
    with open('./data/WilsonCowan/output/WilsonCowan_PRC_S=' +  str(S0) + '.txt', 'w') as output:
        for i in range(len(theta_perturbed)):
            content = theta_perturbed[i] - theta_free[i]
            content = str(content)
            output.write(content + " ")
    with open('./data/WilsonCowan/output/WilsonCowan_shifted_LC_S=' +  str(S0) + '.txt', 'w') as output:
        for i in range(len(shifted_positions)):
            content_1 = str(shifted_positions[i][0])
            content_2 = str(shifted_positions[i][1])
            output.write(content_1 + " " + content_2  + "\n")
    with open('./data/WilsonCowan/output/WilsonCowen_variance_stoch_S=' +  str(S0) + '.txt', 'w') as output:
        for i in range(len(variance_list_perturbed_theo)):
            content = str(variance_list_perturbed_theo[i])
            output.write(content + " ")
            
    with open('./data/WilsonCowan/output/WilsonCowen_variance_theo_S=' +  str(S0) + '.txt', 'w') as output:
        for i in range(len(variance_list_perturbed_stoch)):
            content = str(variance_list_perturbed_stoch[i])
            output.write(content + " ")
            
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    
if __name__ == '__main__':
    main()