"""
Solution of Kepler's equation.
Consolidated algorythm's 3.1, 3.2, 3.3
"""

import numpy as np
from stump import stump_c, stump_s


def kepler_E(e, M):
    """
    This function uses Newton's method to solve Kepler's
    equation E - e*sin(E) = M for the eccentric anomaly,
    given the eccentricity and the mean anomaly.

    E  - eccentric anomaly (radians)
    e  - eccentricity, passed from the calling program
    M  - mean anomaly (radians), passed from the calling program
    pi - 3.1415926...

    User py-functions required: none
    """
    # Set an error tolerance:
    error = 1.0e-8

    # Select a starting value for E:
    if M < np.pi:
        E = M + e / 2
    else:
        E = M - e / 2

    # Iterate on Equation 3.17 until E is determined to within
    # the error tolerance:
    ratio = 1
    while abs(ratio) > error:
        ratio = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - ratio

    return E


# ALGORITHM 3.2: SOLUTION OF KEPLER'S EQUATION FOR THE
# HYPERBOLA USING NEWTON'S METHOD
def kepler_H(e, M):
    """
    This function uses Newton's method to solve Kepler's equation
    for the hyperbola e*sinh(F) - F = M for the hyperbolic
    eccentric anomaly, given the eccentricity and the hyperbolic
    mean anomaly.

    F - hyperbolic eccentric anomaly (radians)
    e - eccentricity, passed from the calling program
    M - hyperbolic mean anomaly (radians), passed from the
    calling program

    User py-functions required: none
    """
    # Set an error tolerance:
    error = 1.0e-8

    # Starting value for F:
    F = M

    # Iterate on Equation 3.45 until F is determined to within
    # the error tolerance:
    ratio = 1
    while abs(ratio) > error:
        ratio = (e * np.sinh(F) - F - M) / (e * np.cosh(F) - 1)
        F = F - ratio

    return F


# ALGORITHM 3.3: SOLUTION OF THE UNIVERSAL KEPLER'S EQUATION
# USING NEWTON'S METHOD
def kepler_U(dt, ro, vro, a, mu):
    """
    This function uses Newton's method to solve the universal
    Kepler equation for the universal anomaly.

    mu   - gravitational parameter (km^3/s^2)
    x    - the universal anomaly (km^0.5)
    dt   - time since x = 0 (s)
    ro   - radial position (km) when x = 0
    vro  - radial velocity (km/s) when x = 0
    a    - reciprocal of the semimajor axis (1/km)
    z    - auxiliary variable (z = a*x^2)
    C    - value of Stumpff function C(z)
    S    - value of Stumpff function S(z)
    n    - number of iterations for convergence
    nMax - maximum allowable number of iterations

    User py-functions required: stumpC, stumpS
    """
    # Set an error tolerance and a limit on the number of iterations:
    error = 1.0e-8
    nMax = 1000

    # Starting value for x:
    x = np.sqrt(mu) * abs(a) * dt

    # Iterate on the equation until convergence occurs within the error tolerance:
    n = 0
    ratio = 1
    while abs(ratio) > error and n <= nMax:
        n += 1
        C = stump_c(a * x**2)
        S = stump_s(a * x**2)
        F = (
            ro * vro / np.sqrt(mu) * x**2 * C
            + (1 - a * ro) * x**3 * S
            + ro * x
            - np.sqrt(mu) * dt
        )
        dFdx = (
            ro * vro / np.sqrt(mu) * x * (1 - a * x**2 * S)
            + (1 - a * ro) * x**2 * C
            + ro
        )
        ratio = F / dFdx
        x -= ratio

    # Deliver a value for x, but report that nMax was reached:
    if n > nMax:
        print(f"\n **No. iterations of Kepler equation = {n}")
        print(f"\n F/dFdx = {F/dFdx}\n")
    return x
