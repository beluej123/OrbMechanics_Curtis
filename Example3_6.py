# Curtis example 3.6, p.188
#  based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#  by Howard D. Curtis
import numpy as np
import scipy.optimize


def orbit_equation_h(r, mu, e, theta):
    # inspired by Curtis example 3.6
    A = r * mu
    B = 1 + e * np.cos(theta)
    return np.sqrt(A * B)


# Solve for eccentricity
def e_zerosolver(e, args):
    # inspired by Curtis example 3.6
    v_0, r_0, mu, theta_0 = args

    A = v_0
    B = orbit_equation_h(r_0, mu, e, theta_0) / r_0
    C = (mu * e * np.sin(theta_0)) / (orbit_equation_h(r_0, mu, e, theta_0))
    return A**2 - B**2 - C**2


# Universal formulation of the Kepler equation.
# The universal variable approach is a more-general approach that applies
#   regardless of orbital eccentricity value
def stumpff_S(z):
    # inspired by Curtis example 3.6
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x)) / (x) ** 3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y) / (y) ** 3
    else:
        return 1 / 6


def stumpff_C(z):
    # inspired by Curtis example 3.6
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2


# Find initial F
def F_from_theta(e, theta):
    # inspired by Curtis example 3.6
    A = np.tan(theta / 2)
    B = np.sqrt((e - 1) / (e + 1))
    return 2 * np.arctanh(A * B)


# Use Universal Kepler to find delta x
def universalx_zerosolver(x, args):
    # inspired by Curtis example 3.6
    r0, vr0, mu, dt, a = args

    A = stumpff_C((x**2) / a) * ((r0 * vr0) / (np.sqrt(mu))) * (x**2)
    B = stumpff_S((x**2) / a) * (1 - r0 / a) * (x**3)
    C = r0 * x
    D = np.sqrt(mu) * dt
    return A + B + C - D


def F_to_theta(e, F):
    # inspired by Curtis example 3.6
    A = np.sqrt((e + 1) / (e - 1))
    B = np.tanh(F / 2)
    return 2 * np.arctan(A * B)


def orbit_type(e):  # returns string
    # inspired by Curtis example 3.6
    if e > 1:
        orb_type = "hyperbola"
    elif e < 1:
        orb_type = "ellipse"
    elif e == 1:
        orb_type = "parabola"
    elif e == 0:
        orb_type = "circle"
    else:
        orb_type = "unknown"
    return orb_type


# Example 3.6, uses algorithm 3.4, find true anomaly at time
# An earth satellite has an initial true anomaly of theta_0 = 30◦,
# a radius of r0 = 10 000 km, and a speed of v0 = 10 km/s.
# Use the universal Kepler’s equation to find the change in
# universal anomaly χ after one hour and use that information
# to determine the true anomaly theta at that time.
mu = 398600  # earth mu value [km^3 / s^2]
theta_0 = 30 * (np.pi / 180)
r_0 = 10000  # [km]
v_0 = 10  # [km/s]
t_1h = 3600  # time of 1hour; but units [s]

e = scipy.optimize.fsolve(e_zerosolver, x0=1, args=[v_0, r_0, mu, theta_0])[0]
h = orbit_equation_h(r_0, mu, e, theta_0)
vr_0 = (mu * e * np.sin(theta_0)) / h

# Find semimajor axis
a_orbit = 1 / (2 / r_0 - (v_0) ** 2 / mu)

F_0 = F_from_theta(e, theta_0)

# initial estimate, x0, using the Chobotov approximation
x0_guess = t_1h * np.sqrt(mu) * np.absolute(1 / a_orbit)

x_1h = scipy.optimize.fsolve(
    universalx_zerosolver, x0=x0_guess, args=[r_0, vr_0, mu, t_1h, a_orbit]
)[0]

F_1h = F_0 + x_1h / np.sqrt(-a_orbit)

theta_1h = F_to_theta(e, F_1h)  # true anomaly in 1hour [rad]
theta_1h_deg = theta_1h * (180 / np.pi)

print("orbit eccentricity, e =", e)
print("orbit type =", orbit_type(e))
print("true anomaly, theta =", theta_1h_deg, "[deg]")
