"""
Curtis [2] problem 3.20 (p.197 in my book); a copy of example 3.7
Given:
    geocentric position, velocity vectors
    [20000, -105000, -19000] [km]
    [0.900, -3.4000, -1.500] [km/s]
Find:
    position, velocity vectors, 2 hours later

References
    ----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020).
    Fundamentals of Astrodynamics. Courier Dover Publications.
    [2] Curtis, H.W. (2009 2nd ed.), section 3.
    Orbital Mechanics for Engineering Students.
"""

import numpy as np
import scipy.optimize

from functionCollection import stumpff_C, stumpff_S


def universalx_zerosolver(x, args):
    r0, vr0, mu, dt, a = args

    A = stumpff_C((x**2) / a) * ((r0 * vr0) / (np.sqrt(mu))) * (x**2)
    B = stumpff_S((x**2) / a) * (1 - r0 / a) * (x**3)
    C = r0 * x
    D = np.sqrt(mu) * dt
    return A + B + C - D


# from universal formulation; write f, g functions for x
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


def test_problem3_20():
    # Example 3.7, uses algorithm 3.4.
    # An earth satellite moves in the xy plane of an inertial frame
    # with origin at the earth’s center.
    # Relative to that frame, the position and velocity of the
    # satellite at time t0 are:
    print(f"Curtis problem 3.20:")

    r0_vec = np.array([20000, -105000, -19000])  # [km]
    v0_vec = np.array([0.900, -3.4000, -1.500])  # [km/s]
    mu_e = 3.986e5  # earth mu [km^3/s^2]

    # Compute the position and velocity vectors of the satellite 60 minutes later
    r0 = np.linalg.norm(r0_vec)  # r magnitude
    v0 = np.linalg.norm(v0_vec)  # v magnitude
    t_delta = (2 * 60) * 60  # convert minutes -> seconds

    vr0 = np.dot(r0_vec, v0_vec) / r0
    a_orbit = 1 / ((2 / r0) - (v0**2 / mu_e))  # semi-major axis

    x0_guess = t_delta * np.sqrt(mu_e) * np.absolute(1 / a_orbit)

    x_1h = scipy.optimize.fsolve(
        universalx_zerosolver, x0=x0_guess, args=[r0, vr0, mu_e, t_delta, a_orbit]
    )[0]

    f_1h = find_f_x(x_1h, r0, a_orbit)
    g_1h = find_g_x(x_1h, t_delta, mu_e, a_orbit)

    r_1h_vector = f_1h * r0_vec + g_1h * v0_vec
    r_1h = np.linalg.norm(r_1h_vector)

    f_dot_delta = find_f_dot_x(x_1h, mu_e, r_1h, r0, a_orbit)
    g_dot_delta = find_g_dot_x(x_1h, r_1h, a_orbit)

    v_1h_vector = f_dot_delta * r0_vec + g_dot_delta * v0_vec
    g_1h = np.linalg.norm(v_1h_vector)

    # extra: eccentricity calculation not using universal formulation.  Not in Curtis example
    h0_vector = np.cross(r0_vec, v0_vec)
    e0_vector = (1 / mu_e) * np.cross(v0_vec, h0_vector) - (r0_vec / r0)
    e0 = np.linalg.norm(e0_vector)  # e magnitude
    if e0 < 0.00005:
        e0 = 0.0
        theta0 = 0  # true anomaly actually undefined, here
    else:
        theta0 = np.arccos(np.dot(e0_vector, r0_vec) / (e0 * r0))
        theta0_deg = theta0 * 180 / np.pi

    print("orbit eccentricity, e0= {e0}")
    print(f"orbit type= {orbit_type(e0)}")
    if e0 == 0.0:
        print(f"true anomaly0, theta0 = not defined; circular")
    else:
        print(f"true anomaly0, theta0= {theta0_deg:.6g} [deg]")
    print(f"position({t_delta:.6g} [s])= {r_1h_vector} [km]")
    print(f"velocity({t_delta:.6g} [s])= {v_1h_vector} [km/s]")
    return None


# ***********************************************
"""
Curtis [2] problem 3.14 (p.196 in my book)
Given:
    geocentric launch in parabolic trajectory
    periapsis altitude 500 [km]
    
Find:
    time to leave earth SOI (sphere of influence)
    
Notes:
Review also for interplanetary flight:
https://people.unipi.it/mario_innocenti/wp-content/uploads/sites/256/2021/12/5RA_2021_ASTRO2.pdf
"""
import math

import functionCollection as funCol


def test_problem3_14():
    print(f"Curtis problem 3.14, plus a little more.")
    GM_sun_au = 3.964016e-14  # [au^3/s^2] sun
    GM_sun_km = 132712.4e6  # [km^3/s^2] sun
    GM_earth_km = 398600  # [km^3/s^2]
    au = 149.597870e6  # [km/au]
    mass_sun = 1.989e30  # [kg]
    mass_earth = 5.974e24  # [kg]
    ecc = .1  # eccentricity = 1 for parabola

    sma = au  # semi-major axis earth orbit
    SOI_earth_sun = funCol.sphere_of_influence(
        sma=sma, mass1=mass_earth, mass2=mass_sun
    )
    print(f"SOI_earth_sun= {SOI_earth_sun:.8g} [km]")

    r_p = 6378 + 500  # [km] radius at periapsis
    # slr = semilatus rectum (aka p or parameter)
    slr = r_p * (1 + ecc)  # BMWS p.21, eqn 1-46
    print(f"slr (p)= {slr:.8g} [km]")

    if ecc>=1: # TODO test for TA calculation for ellipse
        TA = math.acos(((slr / SOI_earth_sun) - 1) / ecc)  # [rad] true angle/anomaly
        print(f"true angle/anomaly, TA= {TA:.6g} [rad], {TA*180/math.pi:.6g} [deg]")

    if ecc == 1:  # parabola
        print(f"parabola:")
        # D = math.sqrt(slr) * math.tan(slr / 2)  # see BMWS p.153, eqn.4-18
        # parabolic Eccentric Angle/Anomaly; note Barker's equation
        D = math.tan(TA / 2)
        # delta_t = ((slr * D + (D**3) / 3)) / (2 * math.sqrt(GM_earth_km))
        delta_t = math.sqrt((slr**3) / GM_earth_km) * (D + (D**3) / 3) / 2

    elif ecc > 1:  # hyperbola
        print(f"hyperbola:")
        # hyperbolic semi-major axis; negative value for hyperbola
        sma_h = -r_p / (ecc - 1)
        # see BMWS p.153, eqn.4-22, 4-20
        cosh_F = (ecc + math.cos(TA)) / (1 + ecc * math.cos(TA))
        # remember in python math.log() is the natural log
        F = abs(math.log(cosh_F + math.sqrt(cosh_F**2 - 1)))
        if TA > math.pi:
            F = -F
        delta_t = (math.sqrt((-(sma_h**3)) / GM_earth_km)) * (ecc * math.sinh(F) - F)

    elif 0 < ecc < 1:
        print(f"ellipse: NOT TESTED yet, 2024-August")
        # TODO complete ellipse calculations
        # elliptic semi-major axis
        # sma_e = slr / (1-ecc**2)
        # tan_E2 = math.sqrt((1 - ecc) / (1 + ecc)) * math.tan(TA / 2)
        # E = 2 * math.atan(tan_E2)
        # delta_t = math.sqrt((sma**3) / GM_earth_km) * (E - ecc * math.sin(E))
        delta_t=0 #temporary, delete after ellipse developed
    else:
        delta_t=0

    print(f"delta_t= {delta_t:.8g} [s]")
    print(f"delta_t= {delta_t/(24*3600)} [d]")

    # TODO develope code for time-of-flight using univeral variables

    return None  # test_problem3_14()


# use the following to test/examine functions
if __name__ == "__main__":

    test_problem3_14()
    # test_problem3_20()
