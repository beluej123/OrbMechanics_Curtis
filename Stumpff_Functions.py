# List of Stumpff Functions
# Stumpff functions originated by Karl Stumpff, circa 1947
# Stumpff functions (C(z), S(z)) are part of a universal variable solution,
#   which is works regardless of eccentricity.
import numpy as np


def stumpff_S(z):
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x)) / (x) ** 3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y) / (y) ** 3
    else:
        return 1 / 6


def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2


import matplotlib.pyplot as plt

z_array = np.linspace(-10, 10, 200)
s_array = [stumpff_S(z) for z in z_array]
c_array = [stumpff_C(z) for z in z_array]

plt.figure(dpi=120)
plt.plot(z_array, s_array)
plt.plot(z_array, c_array)
plt.show()
