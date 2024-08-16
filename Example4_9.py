# Curtis example 4.9, p.229 in my book
#  based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#  by Howard D. Curtis
import numpy as np


def node_regression(peri, apo, i, mu, J2, R):
    # convert to radians
    i *= np.pi / 180
    e = (apo - peri) / (apo + peri)
    a = 0.5 * (apo + peri)
    A = np.sqrt(mu) * J2 * R**2
    B = ((1 - e**2) ** 2) * a**3.5
    node_r = -1.5 * (A / B) * np.cos(i)
    # Convert back to deg
    return (180 / np.pi) * node_r


def perigee_advance(peri, apo, i, mu, J2, R):
    # Convert to radians
    i *= np.pi / 180
    e = (apo - peri) / (apo + peri)
    a = 0.5 * (apo + peri)
    A = np.sqrt(mu) * J2 * R**2
    B = ((1 - e**2) ** 2) * a**3.5
    C = 2.5 * (np.sin(i)) ** 2 - 2
    peri_adv = -1.5 * (A / B) * C
    # Convert back to deg
    return (180 / np.pi) * peri_adv


# Example 4.9
# A satellite is to be launched into a sun-synchronous circular orbit with
#   a period of 100 minutes.
# Determine the required altitude (r) and orbit inclination (incl).
period = 100 * 60  # [s]
mu = 398600  # earth mu value [km^3 / s^2]
rE = 6378.0  # earth radius [km]

# T = 2pi/rt(mu) * r^1.5

r = (period * np.sqrt(mu) / (2 * np.pi)) ** (2 / 3)

d_node_r = 0.9856 * (np.pi / 180) / (24 * 3600)

cos_incl = -1 * d_node_r / (1.5 * ((np.sqrt(mu) * 0.00108263 * 6378**2) / ((r**3.5))))
incl = np.arccos(cos_incl) * (180 / np.pi)
print("orbit altitude = ", r - rE)
print("orbit inclination = ", incl, "[deg]")
