"""
Curtis chapter 3, examples collection.

Notes:
----------
    This file is organized with each example as a function; example function name:
        def curtis_ex3_1():
    
    All supporting functions for all examples are collected right after this
    document block, and all example test functions are defined/enabled at the
    end of this file.  Each example function is designed to be stand-alone,
    however, you need to copy the imports and the supporting functions.

References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import numpy as np
import scipy.optimize


def calculate_E(ecc, theta):
    # inspired by Curtis example 3.1
    A = np.sqrt((1 - ecc) / (1 + ecc))
    B = np.tan(theta / 2)
    return 2 * np.arctan(A * B)


def E_zerosolver(E, args):
    # inspired by Curtis example 3.1
    Me = args[0]
    ecc = args[1]
    return E - ecc * np.sin(E) - Me


def solve_for_E(Me, ecc):
    # inspired by Curtis example 3.1
    # iterative solution process
    sols = scipy.optimize.fsolve(E_zerosolver, x0=Me, args=[Me, ecc])
    return sols


def orbit_equation_r(h: float, mu: float, ecc: float, theta: float):
    # inspired by Curtis example 3.5
    """
    Given: eccentricity (ecc), angular momentum (h)
    Find: orbital radius

    Parameters
    ----------
    h     : float, angular momentum (h)
    mu    : float,
    ecc   : float, eccentricity
    theta : float, true angle/anomaly

    Returns
    -------
    A/B = orbital radius: float
        orbital radius
    """
    A = h**2 / mu
    B = 1 + ecc * np.cos(theta)
    return A / B


def F_from_theta(ecc, theta):
    # inspired by Curtis example 3.5
    A = np.tan(theta / 2)
    B = np.sqrt((ecc - 1) / (ecc + 1))
    return 2 * np.arctanh(A * B)


def F_to_theta(ecc, F):
    # inspired by Curtis example 3.5
    A = np.sqrt((ecc + 1) / (ecc - 1))
    B = np.tanh(F / 2)
    return 2 * np.arctan(A * B)


def F_zerosolver(F, args):
    # inspired by Curtis example 3.5
    # 2024-08-15, not sure about details used in conjunction with scipy.optimize.fsolve()
    Mh = args[0]
    ecc = args[1]
    return -F + ecc * np.sinh(F) - Mh


def solve_for_F(Mh, ecc):
    # inspired by Curtis example 3.5
    # iterate to solve for F; x0=inital guess
    sols = scipy.optimize.fsolve(F_zerosolver, x0=Mh, args=[Mh, ecc])
    return sols  # solutions...


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


# from universal formu_earth_kmlation; write f, g functions for x
def find_f_x(x, r0, a):
    # inspired by Curtis example 3.7
    A = x**2 / r0
    B = stumpff_C(x**2 / a)
    return 1 - A * B


def find_g_x(x, dt, mu_earth_km, a):
    # inspired by Curtis example 3.7
    A = x**3 / np.sqrt(mu_earth_km)
    return dt - A * stumpff_S(x**2 / a)


def find_f_dot_x(x, mu_earth_km, r, r0, a):
    # inspired by Curtis example 3.7
    A = np.sqrt(mu_earth_km) / (r * r0)
    B = stumpff_S(x**2 / a) * (x**3 / a)
    return A * (B - x)


def find_g_dot_x(x, r, a):
    # inspired by Curtis example 3.7
    A = x**2 / r
    return 1 - A * stumpff_C(x**2 / a)


def curtis_ex3_1():
    """
    Curtis, p.163, example 3.1, and p.64, example 3.2.

    Given:
        geocentric elliptical orbit,
        perigee radius 9600 km,
        apogee radius 21,000 km.
    Find:
        Calculate time-of-flight from perigee to true anomaly = 120 deg.

    Notes:
    ----------
        May shorten development; see https://github.com/jkloser/OrbitalMechanics
        References: see list at file beginning.
    """
    # Example 3.1
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
    time_3h = 3 * 60 * 60
    Me_3h = time_3h * 2 * np.pi / T

    E_3h = solve_for_E(Me_3h, ecc)[0]  # iterative solution process
    theta_3h = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E_3h / 2))
    theta_3h_degrees = (180 / np.pi) * theta_3h + 360
    print(f"true anomaly after 3hr, theta_3h_degrees= {theta_3h_degrees:.6g} [deg]")

    return None  # curtis_ex3_1()


def curtis_ex3_5():
    """
    Curtis, p.179, example 3.5
    Given:
        geocentric trajectory,
        perigee velocity 15 km/s,
        perigee altitude of 300 km.
    Find:
        (a) radius and time when true anomaly = 100 [deg]
        (b) position and speed 3 hours later

    Notes:
    ----------
        References: see list at file beginning.
    """
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

    return None  # curtis_ex3_5()


def curtis_ex3_6():
    """
    TODO clean up this section
    Example 3.6, uses algorithm 3.4, find true anomaly at time
    An earth satellite has an initial true anomaly of theta_0 = 30◦,
    a radius of r0 = 10 000 km, and a speed of v0 = 10 km/s.
    Use the universal Kepler’s equation to find the change in
    universal anomaly χ after one hour and use that information
    to determine the true anomaly theta at that time.

    Given:
        TODO breakdown paramerters for this section
    Find:
        TODO breakdown paramerters for this section

    Notes:
    ----------
        References: see list at file beginning.
    """

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

    return None  # curtis_ex3_6()


def curtis_ex3_7():
    """
    Curtis, p.192, example 3.7, uses algorithm 3.4.
    Find position and velocity after 60 [min]

    Given:
        geocentric position, velocity vectors
        r0_vec=[7000, -12124, 0] [km]
        v0_vec=[2.6679, 4.621, 0] [km/s]
    Find:
        position, velocity vectors, 1 hour later

    Notes:
    ----------
        Earth satellite moves in earth center plane.
        References: see list at file beginning.
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

    return None  # curtis_ex3_7()


def test_curtis_ex3_1():
    print(f"\nTest Curtis example 3.1, ... :")
    # function does not need input parameters.
    curtis_ex3_1()
    return None


def test_curtis_ex3_5():
    print(f"\nTest Curtis example 3.5, ... :")
    # function does not need input parameters.
    curtis_ex3_5()
    return None


def test_curtis_ex3_6():
    print(f"\nTest Curtis example 3.6, ... :")
    # function does not need input parameters.
    curtis_ex3_6()
    return None


def test_curtis_ex3_7():
    print(f"\nTest Curtis example 3.7, ... :")
    # function does not need input parameters.
    curtis_ex3_7()
    return None


# use the following to test/examine functions
if __name__ == "__main__":

    # test_curtis_ex3_1()  # test curtis example 3.1
    # test_curtis_ex3_5()  # test curtis example 3.5
    # test_curtis_ex3_6()  # test curtis example 3.6
    test_curtis_ex3_7()  # test curtis example 3.7
