#  based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#  by Howard D. Curtis
import numpy as np
import scipy.optimize

# Example 3.7
# An earth satellite moves in the xy plane of an inertial frame
# with origin at the earth’s center.
# Relative to that frame, the position and velocity of the
# satellite at time t0 are:

r0_vector = np.array([7000.0, -12124])  # [km]
v0_vector = np.array([2.6679, 4.6210])  # [km/s]
mu = 398600

# Compute the position and velocity vectors of the satellite 60 minutes later
r0 = np.linalg.norm(r0_vector)  # r magnitude
v0 = np.linalg.norm(v0_vector)  # v magnitude
t_1h = 60 * 60  # convert minutes -> seconds

vr0 = np.dot(r0_vector, v0_vector) / r0
a_orbit = 1 / ((2 / r0) - (v0**2 / mu))


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
    r0, vr0, mu, dt, a = args

    A = stumpff_C((x**2) / a) * ((r0 * vr0) / (np.sqrt(mu))) * (x**2)
    B = stumpff_S((x**2) / a) * (1 - r0 / a) * (x**3)
    C = r0 * x
    D = np.sqrt(mu) * dt
    return A + B + C - D


x0_guess = t_1h * np.sqrt(mu) * np.absolute(1 / a_orbit)

x_1h = scipy.optimize.fsolve(
    universalx_zerosolver, x0=x0_guess, args=[r0, vr0, mu, t_1h, a_orbit]
)[0]


# write f,g functions for x
def find_f_x(x, r0, a):
    A = x**2 / r0
    B = stumpff_C(x**2 / a)
    return 1 - A * B


def find_g_x(x, dt, mu, a):
    A = x**3 / np.sqrt(mu)
    return dt - A * stumpff_S(x**2 / a)


def find_f_dot_x(x, mu, r, r0, a):
    A = np.sqrt(mu) / (r * r0)
    B = stumpff_S(x**2 / a) * (x**3 / a)
    return A * (B - x)


def find_g_dot_x(x, r, a):
    A = x**2 / r
    return 1 - A * stumpff_C(x**2 / a)


f_1h = find_f_x(x_1h, r0, a_orbit)
g_1h = find_g_x(x_1h, t_1h, mu, a_orbit)

r_1h_vector = f_1h * r0_vector + g_1h * v0_vector
r_1h = np.linalg.norm(r_1h_vector)

f_dot_1h = find_f_dot_x(x_1h, mu, r_1h, r0, a_orbit)
g_dot_1h = find_g_dot_x(x_1h, r_1h, a_orbit)

v_1h_vector = f_dot_1h * r0_vector + g_dot_1h * v0_vector
g_1h = np.linalg.norm(v_1h_vector)

print("position(", t_1h, "[s])=", r_1h_vector)
print("velocity(", t_1h, "[s])=", v_1h_vector)
