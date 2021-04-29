#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:38:21 2021

@author: pierrehouzelstein
"""

from numpy import linspace, array, meshgrid, zeros
from numpy import sin, cos, pi
from datetime import datetime
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal.signaltools import correlate
from numba import jit
from math import fmod
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

@jit
def stuartLandau(z, t, a, alpha):
    """
    Studied function
    """
    x, y = z
    fx = alpha*x*(1 - (x**2 + y**2)) - y*(1 + alpha*a*(x**2 + y**2))
    fy = alpha*y*(1 - (x**2 + y**2)) + x*(1 + alpha*a*(x**2 + y**2))
    return [fx, fy]

def vector_space(a, alpha):
    """
    Plots the vector field linked to the system
    """
    x, y = meshgrid(linspace(-2, 2, num=20), linspace(-2, 2, num=20))
    fx, fy =  stuartLandau([x, y], None, a, alpha)
    plt.quiver(x, y, fx, fy, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def findMaxs(x):
    for i in reversed(range(0,len(x)-1)):
        if x[i] > x[i+1] and x[i] > x[i-1]:  
            return i    

def phase_shift(trajectory_1, trajectory_2, t, nsamples, period):
    """
    Gets the phase shift between two trajectories
    """
    # regularize datasets by subtracting mean and dividing by s.d.
    trajectory_1 -= trajectory_1.mean(); trajectory_1 /= trajectory_1.std()
    trajectory_2 -= trajectory_2.mean(); trajectory_2 /= trajectory_2.std()
    corr = correlate(trajectory_1, trajectory_2)
    dt = linspace(-t[-1], t[-1], 2*nsamples-1)
    recovered_time_shift = dt[corr.argmax()]
    recovered_phase_shift = recovered_time_shift/period
    return recovered_phase_shift


def main():
  
    startTime = datetime.now()
    a, alpha = [0, 1]; period = 2*pi/(1 + alpha*a)
    N = 50; phaseArray = linspace(0, 1, N)

    xCycle = cos(2*pi*phaseArray); yCycle = sin(2*pi*phaseArray)
    perturb_x = 0.5; perturb_y = 0.1
    prc_x_1 = zeros(N); prc_x_2 = zeros(N); prc_y_1 = zeros(N); prc_y_2 = zeros(N)
    
    for i in range(0, N):
        # We integrate the perturbed trajectory
        ic = array([xCycle[i] + perturb_x,  yCycle[i] + perturb_y])
        timeSteps = 2000.
        t = linspace(0, 2*period, timeSteps + 1)
        args=(a, alpha)
        
        pertOrbits = odeint(stuartLandau, ic, t, args, rtol=1e-8, atol=1e-8)
        plt.plot(pertOrbits [:,0], pertOrbits[:,1], color='green')
        
        # We integrate the unperturbed trajectory
        ic = array([xCycle[i], yCycle[i]])
        timeSteps = 2000.
        t = linspace(0, 2*period, timeSteps + 1)
        args=(a, alpha)
        
        unpertOrbits = odeint(stuartLandau, ic, t, args, rtol=1e-8, atol=1e-8)

        # We obtain the phase shifts
        # Method 1: We obtain the index of the maximas
        ix_perturbed = findMaxs(pertOrbits[:,0]); iy_perturbed = findMaxs(pertOrbits[:,1])
        ix_free = findMaxs(unpertOrbits[:,0]); iy_free = findMaxs(unpertOrbits[:,1])
        # And we compute the phase differences
        delta_thetaX = (t[ix_perturbed] - t[ix_free])/period
        delta_thetaY = (t[iy_perturbed] - t[iy_free])/period
        delta_thetaX = fmod(delta_thetaX, 1)
        delta_thetaY = fmod(delta_thetaY, 1)
        prc_x_1[i] = delta_thetaX 
        prc_y_1[i] = delta_thetaY 

        #Method 2: use cross correlation
        delta_thetaX = phase_shift(pertOrbits[:,0], unpertOrbits[:,0], t, timeSteps, period)
        delta_thetaY = phase_shift(pertOrbits[:,1], unpertOrbits[:,1], t, timeSteps, period)
        delta_thetaX = fmod(delta_thetaX, 1)
        delta_thetaY = fmod(delta_thetaY, 1)
        prc_x_2[i] = delta_thetaX 
        prc_y_2[i] = delta_thetaY 

    
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    plt.title("Trajectory of the shifted points; alpha = {}".format(alpha) + ", a = {}".format(a))
    vector_space(a, alpha)
    
    plt.plot()
    plt.plot(phaseArray, prc_x_1, 'r+')
    plt.plot(phaseArray, prc_x_2, 'b.')
    plt.title('PRC with initial x-shift of {}'.format(perturb_x) + ' and y-shift of {}'.format(perturb_y) 
              + "; alpha = {}".format(alpha) + ", a = {}".format(a))
    plt.xlabel("$\Theta_X$")
    plt.ylabel("$\Delta\Theta_X$")
    plt.legend(["Using indice of maximum", "Using cross-correlation function"])
    plt.show()
    
"""
    plt.plot()
    plt.plot(phaseArray, prc_y_1, 'r+')
    plt.plot(phaseArray, prc_y_2, 'b.')
    plt.title('PRC with initial x-shift of {}'.format(perturb_x) + ' and y-shift of {}'.format(perturb_y)
              + "; alpha = {}".format(alpha) + ", a = {}".format(a))
    plt.xlabel("$\Theta_Y$")
    plt.ylabel("$\Delta\Theta_Y$")
    plt.legend(["Using indice of maximum", "Using cross-correlation function"])
    plt.show()
"""
    
if __name__ == '__main__':
    main()
