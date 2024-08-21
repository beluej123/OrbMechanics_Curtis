"""
Curtis [3] example 3.5 (p.179 in my book). H.W. Curtis
    Orbital Mechanics for Engineering Students, 2nd ed., 2009
Given: geocentric trajectory, perigee velocity 15 km/s, perigee altitude of 300 km.
Find:
(a) radius and time when true anomaly = 100 [deg]
(b) position and speed 3 hours later
References
    ----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.)
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2013 4th ed.; i.e. my book).
        Orbital Mechanics for Engineering Students.
"""

import numpy as np
import scipy.optimize


def orbit_equation_r(h: float, mu: float, ecc: float, theta: float):
    """
    Given: eccentricity (ecc), angular momentum (h)
    Find: orbital radius

    Parameters
    ----------
    h : float
        angular momentum (h)
    mu : float
        _description_
    ecc : float
        eccentricity
    theta : float
        true angle/anomaly

    Returns
    -------
    A/B = orbital radius: float
        orbital radius
    """
    A = h**2 / mu
    B = 1 + ecc * np.cos(theta)
    return A / B


def F_from_theta(ecc, theta):
    A = np.tan(theta / 2)
    B = np.sqrt((ecc - 1) / (ecc + 1))
    return 2 * np.arctanh(A * B)


def F_to_theta(ecc, F):
    A = np.sqrt((ecc + 1) / (ecc - 1))
    B = np.tanh(F / 2)
    return 2 * np.arctan(A * B)


def F_zerosolver(F, args):
    # 2024-08-15, not sure about details used in conjunction with scipy.optimize.fsolve()
    Mh = args[0]
    ecc = args[1]
    return -F + ecc * np.sinh(F) - Mh


def solve_for_F(Mh, ecc):
    # iterate to solve for F; x0=inital guess
    sols = scipy.optimize.fsolve(F_zerosolver, x0=Mh, args=[Mh, ecc])
    return sols  # solutions...


print(f"Curtis example 3.5:")
r_p = 300 + 6378  # [km]
v_p = 15  # [km/s]
mu_earth_km = 398600  # [km^3/s^2]

# (a) the radius when the true angle/anomaly is 100 deg
# calculate primary orbital parameters: h (angular momentum), ecc (eccentricity)
theta_a = 100 * (np.pi / 180)  # [rad] convert true anomaly deg->rad
h = r_p * v_p  # [km^2/s]
ecc = (1 / np.cos(0)) * ((h**2 / (r_p * mu_earth_km)) - 1)
r_100deg = orbit_equation_r(h, mu_earth_km, ecc, theta_a)

# (b) the position and speed three hours later. (after 100 deg)
if ecc > 1:  # this is a hyperbola
    F_a = F_from_theta(ecc, theta_a)  # hyperbolic eccentric angle/anomaly
    # mean angle/anomaly at theta_a (ie. 100 [deg])
    Mh_a = ecc * np.sinh(F_a) - F_a  # Kepler's eqn for hyperbola

    # time since periapsis, at true anomaly (100 deg)
    t_a = Mh_a * (h**3 / mu_earth_km**2) * ((ecc**2 - 1) ** (-3 / 2))  # [s]
    t_3h = 3 * 60 * 60 + t_a

    # mean angle/anomaly at 3 hours
    Mh_3h = (mu_earth_km**2 / h**3) * ((ecc**2 - 1) ** 1.5) * t_3h
    F_3h = solve_for_F(Mh_3h, ecc)[0]  # iterative solution process

    theta_3h = F_to_theta(ecc, F_3h)
    theta_3h_deg = (180 / np.pi) * theta_3h

    r_3h = orbit_equation_r(h, mu_earth_km, ecc, theta_3h)
    v_3h_tangent = h / r_3h
    v_3h_radial = (mu_earth_km / h) * ecc * np.sin(theta_3h)
    v_3h = np.linalg.norm([v_3h_tangent, v_3h_radial])

elif ecc == 1:  # this is parabola
    print(f"parabola not yet implemented, 2024-Aug")
    exit()
else:  # this is ellipse
    print(f"ellipse not yet implemented, 2024-Aug")
    exit()

# print parameters of Curtis example
print(f"orbital eccentricity, ecc= {ecc:.8g}")
print(f"orbital radius (true anomaly, 100deg), r= {r_100deg:.8g} [km]")
print(f"time since periapsis (true anomaly (100 deg)), t_a= {t_a:.8g} [s]")
print(f"3hr after 100deg, t_3h= {t_3h:.8g} [s], {t_3h/(60*60):.8g} [hr]")
print(f"true anomaly after 3hr, t_3h= {theta_3h_deg:.8g} [deg]")
print(f"orbital radius after 3hr, r_3h= {r_3h:.8g} [km]")
print(f"orbital speed after 3hr, v_3h= {v_3h:.8g} [km/s]")
