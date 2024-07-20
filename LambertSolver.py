# Curtis part of algorithm 5.2 (p.270+ in my book). H.W. Curtis
# Orbital Mechanics for Engineering Students, 2nd ed., 2009
# Given r1, r2, and dt;
# Find v1 & v2, and orbital elements;
#   Note Gauss problem, or Lamberts theory, and solution
import numpy as np
import scipy.optimize


# Auxiliary functions
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


def y_lambert(z, r1, r2, A):
    K = (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))
    return r1 + r2 + A * K


def A_lambert(r1, r2, d_theta):
    K1 = np.sin(d_theta)
    K2 = np.sqrt((r1 * r2) / (1 - np.cos(d_theta)))
    return K1 * K2


def lambert_zerosolver(z, args):
    dt, mu, r1, r2, A = args
    K1 = ((y_lambert(z, r1, r2, A) / stumpff_C(z)) ** 1.5) * stumpff_S(z)
    K2 = A * np.sqrt(y_lambert(z, r1, r2, A))
    K3 = -1 * dt * np.sqrt(mu)
    return K1 + K2 + K3


def find_f_y(y, r1):
    return 1 - y / r1


def find_g_y(y, A, mu):
    return A * np.sqrt(y / mu)


def find_f_dot_y(y, r1, r2, mu, z):
    K1 = np.sqrt(mu) / (r1 * r2)
    K2 = np.sqrt(y / stumpff_C(z))
    K3 = z * stumpff_S(z) - 1
    return K1 * K2 * K3


def find_g_dot_y(y, r2):
    return 1 - y / r2


# Main function
# Assumes prograde trajectory, can be changed in function call
def Lambert_v1v2_solver(r1_v, r2_v, dt, mu, prograde=True):
    # step 1:
    r1 = np.linalg.norm(r1_v)
    r2 = np.linalg.norm(r2_v)

    # step 2:
    r1r2z = np.cross(r1_v, r2_v)[2]
    cos_calc = np.dot(r1_v, r2_v) / (r1 * r2)
    if prograde:
        if r1r2z < 0:
            d_theta = 2 * np.pi - np.arccos(cos_calc)
        else:
            d_theta = np.arccos(cos_calc)
    else:
        if r1r2z < 0:
            d_theta = np.arccos(cos_calc)
        else:
            d_theta = 2 * np.pi - np.arccos(cos_calc)

    # step 3:
    A = A_lambert(r1, r2, d_theta)

    # step 4:
    # find starting estimate
    z = scipy.optimize.fsolve(lambert_zerosolver, x0=1.5, args=[dt, mu, r1, r2, A])[0]

    # step 5:
    y = y_lambert(z, r1, r2, A)

    # step 6:
    f_dt = find_f_y(y, r1)
    g_dt = find_g_y(y, A, mu)
    f_dot_dt = find_f_dot_y(y, r1, r2, mu, z)
    g_dot_dt = find_g_dot_y(y, r2)

    # step 7:
    v1_v = (1 / g_dt) * (r2_v - f_dt * r1_v)
    v2_v = (g_dot_dt / g_dt) * r2_v - (
        (f_dt * g_dot_dt - f_dot_dt * g_dt) / g_dt
    ) * r1_v

    # TODO: add calculation of orbital elements; maybe a different function
    return v1_v, v2_v
