"""
State variables and orbital elements
"""

import numpy as np


def coe_from_sv(r_vec, v_vec, mu):
    """
    Compute the classical orbital elements (coe)
    from the state vector (r_vec, v_vec), Curtis [3] Algorithm 4.1.

    mu      - gravitational parameter (km^3/s^2)
    r_vec       - position vector in the geocentric equatorial frame (km)
    v_vec       - velocity vector in the geocentric equatorial frame (km/s)
    r, v    - the magnitudes of r_vec and v_vec
    vr      - radial velocity component (km/s)
    H       - the angular momentum vector (km^2/s)
    h       - the magnitude of H (km^2/s)
    incl    - inclination of the orbit (rad)
    N       - the node line vector (km^2/s)
    n       - the magnitude of N
    cp      - cross product of N and r_vec
    RA      - right ascension of the ascending node (rad)
    E       - eccentricity vector
    e       - eccentricity (magnitude of E)
    eps     - a small number below which the eccentricity is considered
              to be zero
    w       - argument of perigee (rad)
    TA      - true anomaly (rad)
    a       - semimajor axis (km)
    pi      - 3.1415926...
    coe     - vector of orbital elements [h e RA incl w TA a]

    User py-functions required: None
    """
    eps = 1.0e-10

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r

    H = np.cross(r_vec, v_vec)
    h = np.linalg.norm(H)

    # ...Equation 4.7:
    incl = np.arccos(H[2] / h)

    # ...Equation 4.8:
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)

    # ...Equation 4.9:
    if n != 0:
        RA = np.arccos(N[0] / n)
        if N[1] < 0:
            RA = 2 * np.pi - RA
    else:
        RA = 0

    # ...Equation 4.10:
    E = (1 / mu) * ((v**2 - mu / r) * r_vec - r * vr * v_vec)
    e = np.linalg.norm(E)

    # ...Equation 4.12 (incorporating the case e = 0):
    if n != 0:
        if e > eps:
            w = np.arccos(np.dot(N, E) / (n * e))
            if E[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0
    else:
        w = 0

    # ...Equation 4.13a (incorporating the case e = 0):
    if e > eps:
        TA = np.arccos(np.dot(E, r_vec) / (e * r))
        if vr < 0:
            TA = 2 * np.pi - TA
    else:
        cp = np.cross(N, r_vec)
        if cp[2] >= 0:
            TA = np.arccos(np.dot(N, r_vec) / (n * r))
        else:
            TA = 2 * np.pi - np.arccos(np.dot(N, r_vec) / (n * r))

    # ...Equation 4.62 (a < 0 for a hyperbola):
    a = h**2 / mu / (1 - e**2)
    coe = [h, e, RA, incl, w, TA, a]
    return coe
