"""
Curtis [2] example 3.1 (p.163 in my book) and example 3.2 (p.164 in my book).
May shorten development; see https://github.com/jkloser/OrbitalMechanics
Given:
    geocentric elliptical orbit,
    perigee radius 9600 km,
    apogee radius 21,000 km.
Find:
    Calculate time-of-flight from perigee to true anomaly = 120 deg.

References
    ----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020).
    Fundamentals of Astrodynamics. Courier Dover Publications.
    [2] Curtis, H.W. (2009 2nd ed.), section 3.
    Orbital Mechanics for Engineering Students.
"""

import numpy as np
import scipy.optimize


def calculate_E(ecc, theta):
    A = np.sqrt((1 - ecc) / (1 + ecc))
    B = np.tan(theta / 2)
    return 2 * np.arctan(A * B)


def E_zerosolver(E, args):
    Me = args[0]
    ecc = args[1]
    return E - ecc * np.sin(E) - Me


def solve_for_E(Me, ecc):
    # iterative solution process
    sols = scipy.optimize.fsolve(E_zerosolver, x0=Me, args=[Me, ecc])
    return sols


# Example 3.1
print(f"Curtis example 3.1:")
peri = 9600  # [km]
apo = 21000  # [km]
theta = 120 * (2 * np.pi / 360)  # [rad] convert true anomaly deg->rad
mu_earth_km = 398600  # [km^3/s^2]

ecc = (apo - peri) / (apo + peri)  # eccentricity
h = np.sqrt(peri * mu_earth_km * (1 + ecc))  # [km^2/s]

# find total period T
T = (2 * np.pi / mu_earth_km**2) * (h / np.sqrt(1 - ecc**2)) ** 3
# eccentric angle/anomaly
E = calculate_E(ecc, theta)
# mean anomaly
Me = E - ecc * np.sin(E)

# time since periapsis
time_sp = Me * T / (2 * np.pi)
print(f"time since periapsis, time_sp= {time_sp:.6g}")

##Example 3.2
# Find the true anomaly at three hours after perigee passage.
# Since the time (10,800 seconds) is greater than one-half the period,
# the true anomaly must be greater than 180 degrees.
print(f"Curtis example 3.2:")

time_3h = 3 * 60 * 60
Me_3h = time_3h * 2 * np.pi / T

E_3h = solve_for_E(Me_3h, ecc)[0]  # iterative solution process
theta_3h = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E_3h / 2))
theta_3h_degrees = (180 / np.pi) * theta_3h + 360
print(f"true anomaly after 3hr, theta_3h_degrees= {theta_3h_degrees:.6g} [deg]")
