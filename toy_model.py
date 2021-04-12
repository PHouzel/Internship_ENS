#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:25:50 2021

@author: pierrehouzelstein

Intoduction to dynamical systems: toy model 
x'1 = beta*x1 - x2 - x1(x1^2 + x2^2)
x'2 = x1 + beta*x2 - x2(x1^2 + x2^2)

Compute limit cycle, phase response curves
"""

import matplotlib.pyplot as plt
import numpy as np


def vector_space(beta):
    
    x1, x2 = np.meshgrid(np.linspace(-1, 1, num=20), np.linspace(-1, 1, num=20))
    y1 = beta*x1 - x2 - x1*(x1**2 + x2**2)
    y2 = x1 + beta*x2 - x2*(x1**2 + x2**2)
            

    plt.quiver(x1, x2, y1, y2, color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()



def attractor(beta):
    """
    Find stable points, i.e (x1, x2) such that (y1, y2) = 0

    """
    x1 = np.linspace(-1, 1, num=2000)
    x2 = np.linspace(-1, 1, num=2000)
    y1 = beta*x1 - x2 - x1*(x1**2 + x2**2)
    y2 = x1 + beta*x2 - x2*(x1**2 + x2**2)
    
    stable_x1 = []
    stable_x2 = []
    
    for i in range(len(y1)):
        for j in range(len(y2)):
            if abs(y1[i])<0.0001 and abs(y2[j])<0.0001:
                stable_x1.append(x1[i])
                stable_x2.append(x2[j])
    return stable_x1, stable_x2


stable_x1, stable_x2 = attractor(beta=1)
plt.plot(stable_x1, stable_x2)
vector_space(beta=1)
