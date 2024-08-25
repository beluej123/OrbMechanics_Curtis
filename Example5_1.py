# Curtis example 5.1, p.261 in my book
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
# Determine the classical orbital elements using Gibbs’procedure; 3 position vectors
import numpy as np


# Find v1, v2, v3 given r1, r2, r3
def N_vector(r1_v, r2_v, r3_v):
    # inspired by example 5.1
    r1 = np.linalg.norm(r1_v)
    r2 = np.linalg.norm(r2_v)
    r3 = np.linalg.norm(r3_v)
    A = r1 * np.cross(r2_v, r3_v)
    B = r2 * np.cross(r3_v, r1_v)
    C = r3 * np.cross(r1_v, r2_v)
    return A + B + C


def D_vector(r1_v, r2_v, r3_v):
    # inspired by example 5.1
    A = np.cross(r1_v, r2_v)
    B = np.cross(r2_v, r3_v)
    C = np.cross(r3_v, r1_v)
    return A + B + C


def S_vector(r1_v, r2_v, r3_v):
    # inspired by example 5.1
    r1 = np.linalg.norm(r1_v)
    r2 = np.linalg.norm(r2_v)
    r3 = np.linalg.norm(r3_v)
    A = (r2 - r3) * r1_v
    B = (r3 - r1) * r2_v
    C = (r1 - r2) * r3_v
    return A + B + C


def gibbs_v_equation(r, N, D, S, mu):
    # inspired by example 5.1
    A = np.sqrt(mu / (np.linalg.norm(N) * np.linalg.norm(D)))
    B = np.cross(D, r) / np.linalg.norm(r)
    return A * (B + S)


def gibbs_r_to_v(r1_v, r2_v, r3_v, mu, zero_c=4):
    # inspired by example 5.1
    k1 = r1_v / np.linalg.norm(r1_v)
    k2 = np.cross(r2_v, r3_v)
    k3 = np.dot(k1, (k2 / np.linalg.norm(k2)))

    # zero_c is the number of decimal places the coplanar checker checks for
    if round(k3, zero_c) != 0:
        return "Vectors not coplanar"

    N = N_vector(r1_v, r2_v, r3_v)
    D = D_vector(r1_v, r2_v, r3_v)
    S = S_vector(r1_v, r2_v, r3_v)
    v1_v = gibbs_v_equation(r1_v, N, D, S, mu)
    v2_v = gibbs_v_equation(r2_v, N, D, S, mu)
    v3_v = gibbs_v_equation(r3_v, N, D, S, mu)

    return v1_v, v2_v, v3_v


r1_v_ex = np.array([-294.42, 4265.1, 5986.7])
r2_v_ex = np.array([-1365.5, 3637.6, 6346.8])
r3_v_ex = np.array([-2940.3, 2473.7, 6555.8])
mu_e = 398600

# Zero check
k1 = r1_v_ex / np.linalg.norm(r1_v_ex)
k2 = np.cross(r2_v_ex, r3_v_ex)
k3 = np.dot(k1, (k2 / np.linalg.norm(k2)))

v_vectors = gibbs_r_to_v(r1_v_ex, r2_v_ex, r3_v_ex, mu_e)
print("v_vectors", v_vectors)
