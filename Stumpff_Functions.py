#List of Stumpff Functions

import numpy as np

def stumpff_S(z):
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x))/(x)**3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y)/(y)**3
    else:
        return (1/6)
        
def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z)))/z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1)/(-z)
    else:
        return(1/2)
          

import matplotlib.pyplot as pl

z_array = np.linspace(-10,10,200)
s_array = [stumpff_S(z) for z in z_array]
c_array = [stumpff_C(z) for z in z_array]

pl.figure(dpi=120)
pl.plot(z_array, s_array)
pl.plot(z_array, c_array)