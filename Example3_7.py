"""
Curtis [2] example 3.7 (p.192)
Find position and velocity after 60minutes

References
    ----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Curtis, H.W. (2009 2nd ed.), section 3.
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import numpy as np
import scipy.optimize


# Find x
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


def universalx_zerosolver(x, args):
    r0, vr0, mu_earth_km, dt, a = args

    A = stumpff_C((x**2) / a) * ((r0 * vr0) / (np.sqrt(mu_earth_km))) * (x**2)
    B = stumpff_S((x**2) / a) * (1 - r0 / a) * (x**3)
    C = r0 * x
    D = np.sqrt(mu_earth_km) * dt
    return A + B + C - D


# from universal formu_earth_kmlation; write f, g functions for x
def find_f_x(x, r0, a):
    A = x**2 / r0
    B = stumpff_C(x**2 / a)
    return 1 - A * B


def find_g_x(x, dt, mu_earth_km, a):
    A = x**3 / np.sqrt(mu_earth_km)
    return dt - A * stumpff_S(x**2 / a)


def find_f_dot_x(x, mu_earth_km, r, r0, a):
    A = np.sqrt(mu_earth_km) / (r * r0)
    B = stumpff_S(x**2 / a) * (x**3 / a)
    return A * (B - x)


def find_g_dot_x(x, r, a):
    A = x**2 / r
    return 1 - A * stumpff_C(x**2 / a)


def orbit_type(e):  # returns string, orbit type
    if e > 1:
        orb_type = "hyperbola"
    elif 0 < e < 1:
        orb_type = "ellipse"
    elif e == 1:
        orb_type = "parabola"
    elif e == 0:
        orb_type = "circle"
    else:
        orb_type = "unknown"
    return orb_type


# ***********************************************
"""
Example 3.7, uses algorithm 3.4.
Earth satellite moves in earth center plane.
Given:
geocentric position, velocity vectors
r0_vec=[7000, -12124, 0] [km]
v0_vec=[2.6679, 4.621, 0] [km/s]
Find:
position, velocity vectors, 2 hours later
"""

r0_vector = np.array([7000.0, -12124, 0.0])  # [km]
v0_vector = np.array([2.6679, 4.6210, 0.0])  # [km/s]
mu_earth_km = 398600  # earth mu_earth_km value [km^3 / s^2]

# Compute the position and velocity vectors of the satellite 60 minutes later
r0 = np.linalg.norm(r0_vector)  # r magnitude
v0 = np.linalg.norm(v0_vector)  # v magnitude
t_1h = 60 * 60  # converts minutes -> seconds

vr0 = np.dot(r0_vector, v0_vector) / r0
a_orbit = 1 / ((2 / r0) - (v0**2 / mu_earth_km))  # semi-major axis
x0_guess = t_1h * np.sqrt(mu_earth_km) * np.absolute(1 / a_orbit)

x_1h = scipy.optimize.fsolve(
    universalx_zerosolver, x0=x0_guess, args=[r0, vr0, mu_earth_km, t_1h, a_orbit]
)[0]


f_1h = find_f_x(x_1h, r0, a_orbit)
g_1h = find_g_x(x_1h, t_1h, mu_earth_km, a_orbit)

r_1h_vector = f_1h * r0_vector + g_1h * v0_vector
r_1h = np.linalg.norm(r_1h_vector)

f_dot_1h = find_f_dot_x(x_1h, mu_earth_km, r_1h, r0, a_orbit)
g_dot_1h = find_g_dot_x(x_1h, r_1h, a_orbit)

v_1h_vector = f_dot_1h * r0_vector + g_dot_1h * v0_vector
g_1h = np.linalg.norm(v_1h_vector)

print(f"position({t_1h}[s])= {r_1h_vector}")
print(f"velocity({t_1h}[s])= {v_1h_vector}")

# an extra: eccentricity calculation
#   does not use universal formu_earth_kmlation.  Not in Curtis example
h0_vector = np.cross(r0_vector, v0_vector)
e0_vector = (1 / mu_earth_km) * np.cross(v0_vector, h0_vector) - (r0_vector / r0)
e0 = np.linalg.norm(e0_vector)  # e magnitude
if e0 < 0.00005:
    e0 = 0.0
    theta0 = 0  # true anomaly actually undefined, here
else:
    theta0 = np.arccos(np.dot(e0_vector, r0_vector) / (e0 * r0))
    theta0_deg = theta0 * 180 / np.pi

print(f"orbit eccentricity, e= {e0:.6g}")
print(f"orbit type= {orbit_type(e0)}")
if e0 == 0.0:
    print(f"true anomaly0, theta0 = not defined; circular")
else:
    print(f"true anomaly0, theta0= {theta0_deg:.6g} [deg]")