"""
Curtis chapter 6, examples collection.

Notes:
----------
    This file is organized with each example as a function, and all function test
        defined/enabled at the end of this file.  Each example function is designed
        to be stand-alone, so you can copy seperately as long as the imports are
        included.
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


# TODO remove redundant functions, below, collected in the various sections.
def energy_ellipse(peri, apo, mu):
    # inspired by example 6.1
    a = (peri + apo) / 2
    return -1 * mu / (2 * a)


def v_ellipse_peri(peri, apo, mu):
    # inspired by example 6.1
    e = (apo - peri) / (apo + peri)
    h = np.sqrt(peri * mu * (1 + e))
    v_peri = h / peri
    return v_peri


def v_ellipse_apo(peri, apo, mu):
    # inspired by example 6.1
    e = (apo - peri) / (apo + peri)
    h = np.sqrt(peri * mu * (1 + e))
    v_apo = h / apo
    return v_apo


def v_circle(r, mu):
    # inspired by example 6.1
    return np.sqrt(mu / r)


def deltaV_Hohmann_circular(r_a, r_b, mu):
    # inspired by Curtis example 6.3
    # rb greater than ra
    a = r_b / r_a
    A = 1 / np.sqrt(a)
    B = -1 * (np.sqrt(2) * (1 - a)) / np.sqrt(a * (1 + a))
    C = np.sqrt(mu / r_a)
    return (A + B - 1) * C


def deltaV_Bielliptic_circular(r_a, r_b, r_c, mu):
    # inspired by Curtis example 6.3
    # rb is transfer ellipse
    a = r_c / r_a
    b = r_b / r_a
    A = np.sqrt((2 * (a + b)) / (a * b))
    B = -1 * ((1 + np.sqrt(a)) / np.sqrt(a))
    C = -1 * np.sqrt(2 / (b * (1 + b))) * (1 - b)
    D = np.sqrt(mu / r_a)
    return (A + B + C) * D


def t_circular(r, mu):
    # inspired by Curtis example 6.3
    return ((2 * np.pi) / np.sqrt(mu)) * r**1.5


def t_ellipse(r_p, r_a, mu):
    # inspired by Curtis example 6.3
    a = (r_a + r_p) / 2
    return ((2 * np.pi) / np.sqrt(mu)) * a**1.5


def E_from_theta(e, theta):
    # inspired by example 6.4
    A = np.sqrt((1 - e) / (1 + e))
    B = np.tan(theta / 2)
    return 2 * np.arctan(A * B)


def orbit_equation_h(r, mu, e, theta):
    # inspired by example 6.4
    A = r * mu
    B = 1 + e * np.cos(theta)
    return np.sqrt(A * B)


def t_from_Me(Me, mu, h, e):
    # inspired by example 6.4
    A = Me
    B = (mu**2) / (h**3)
    C = (1 - e**2) ** 1.5
    return A / (B * C)


def t_ellipse(r_p, r_a, mu):
    # inspired by example 6.4
    a = (r_a + r_p) / 2
    return ((2 * np.pi) / np.sqrt(mu)) * a**1.5


def v_ellipse_peri(peri, apo, mu):
    # inspired by example 6.4
    e = (apo - peri) / (apo + peri)
    h = np.sqrt(peri * mu * (1 + e))
    v_peri = h / peri
    return v_peri


def curtis_ex6_1():
    """
    Curtis, p.323, example 6.1; also see Orbit_from_r0v0.py
    TODO clean up this description
    A spacecraft is in a 480km by 800km earth orbit.
    (a) find the v required at perigee A to place the spacecraft in a 480km by
      16000km transfer orbit (orbit 2);
    (b) the v (apogee kick) required at B of the transfer orbit to
      establish a circular orbit of 16000km altitude (orbit 3)
    (c) total propellant if specific impulse is 300s

    Given:
        earth altitude's r1 & r2 (not vectors), dt, and delta true anomaly;
    Find:
        periapsis altitude, time to periapsis

    Notes:
    ----------
        References: see list at file beginning.
    """
    mu_e = 3.986e5  # earth mu [km^3/s^2]
    r_ea = 6378  # earth radius [km]
    mass_sc = 2000  # [kg]
    specImp = 300  # specific impulse [s]

    orbit1_peri = 480 + r_ea
    orbit1_apo = 800 + r_ea

    orbit2_peri = 480 + r_ea
    orbit2_apo = 16000 + r_ea

    orbit3_peri = 16000 + r_ea
    orbit3_apo = 16000 + r_ea

    orbit1_energy = energy_ellipse(orbit1_peri, orbit1_apo, mu_e)
    orbit2_energy = energy_ellipse(orbit2_peri, orbit2_apo, mu_e)
    orbit3_energy = energy_ellipse(orbit3_peri, orbit3_apo, mu_e)

    de_1 = orbit2_energy - orbit1_energy
    de_2 = orbit3_energy - orbit2_energy

    e_o1 = (orbit1_apo - orbit1_peri) / (orbit1_peri + orbit1_apo)
    h_o1 = np.sqrt(orbit1_peri * mu_e * (1 + e_o1))

    # part a
    v_peri_o1 = v_ellipse_peri(orbit1_peri, orbit1_apo, mu_e)
    v_peri_o2 = v_ellipse_peri(orbit2_peri, orbit2_apo, mu_e)
    dv_1 = v_peri_o2 - v_peri_o1

    # part b
    v_apo_o2 = v_ellipse_apo(orbit2_peri, orbit2_apo, mu_e)
    v_o3 = v_circle(orbit3_apo, mu_e)
    dv_2 = v_o3 - v_apo_o2
    total_dv = dv_1 + dv_2  # [km/s]
    #   remember to manage units; convert specific impulse defined for 9.807 [m/s^2] not [km/s^2]
    total_dv_m = total_dv * 1000  # convert -> [m/s]

    # part c;
    delta_mass = mass_sc * (1 - np.exp(-total_dv_m / (specImp * 9.807)))

    print("Hohmann Transfer:")
    print("delta v1", dv_1)
    print("delta v2", dv_2)
    print("total delta v", total_dv)
    print("propellant mass", delta_mass)

    return None  # curtis_ex6_1()


def curtis_ex6_2():
    """
    Curtis p.326, example 6.2
    TODO clean up this example description.
    A spacecraft returning from a lunar mission approaches earth on a hyperbolic
    trajectory. At its closest approach A it is at an altitude of 5000 km,
    traveling at 10 km/s. At A retrorockets are fired to lower the spacecraft
    into a 500 km altitude circular orbit, where it is to rendezvous with a
    space station. Find the location of the space station
    at retrofire so that rendezvous will occur at B.

    Given:

    Find:


    Notes:
    ----------

        References: see list at file beginning.
    """
    r_ea = 6378  # earth radius [km]
    mu_e = 3.986e5  # earth mu [km^3/s^2]
    r_hyp = 5000 + r_ea  # [km]
    v_hyp = 10  # [km/s]

    ra_o2 = 5000 + r_ea
    rp_o2 = 500 + r_ea

    a_o2 = (ra_o2 + rp_o2) / 2

    T_o2 = ((2 * np.pi) / np.sqrt(mu_e)) * a_o2**1.5

    time_taken = T_o2 / 2

    T_o3 = ((2 * np.pi) / np.sqrt(mu_e)) * rp_o2**1.5

    orbital_portion = time_taken / T_o3
    orbital_angle = orbital_portion * 360
    print("Hyperbolic approach, Hohmann transfer:")
    print("given; hyperbolic closest:", r_hyp, "[km]")
    print("given; hyperbolic velocity @ r_hyp:", v_hyp, "[km/s]")
    print("given; inner orbit altitude:", ra_o2, "[deg]")
    print("\norbit transfer time:", time_taken, "[s]")
    print("randevous phasing:", orbital_angle, "[deg]")
    return None  # curtis_ex6_2()


def curtis_ex6_3():
    """
    Curtis, p.
    TODO clean up this example description.
    Find the total delta-v requirement for a bi-elliptical Hohmann
    transfer from a geocentric circular orbit of 7000 km radius to
    one of 105 000 km radius. Let the apogee of the first ellipse
    be 210 000 km. Compare the delta-v schedule and total flight time
    with that for an ordinary single Hohmann transfer ellipse.

    Given:

    Find:


    Notes:
    ----------

        References: see list at file beginning.
    """
    r_o1 = 7000
    r_o2 = 210000
    r_o3 = 105000
    mu = 398600

    # Compare delta v

    dv_hohmann = deltaV_Hohmann_circular(r_o1, r_o3, mu)
    dv_biell = deltaV_Bielliptic_circular(r_o1, r_o2, r_o3, mu)

    if dv_biell < dv_hohmann:
        print(
            "Bi-elliptic transfer more efficient by "
            + str(round(dv_hohmann - dv_biell, 4))
            + " km/s"
        )

    # Compare flight times
    # Hohmann:
    dt_hohmann = t_ellipse(r_o1, r_o3, mu) / 2

    # Bi-elliptic:
    dt_biell_1 = t_ellipse(r_o1, r_o2, mu) / 2
    dt_biell_2 = t_ellipse(r_o2, r_o3, mu) / 2
    dt_biell = dt_biell_1 + dt_biell_2

    print(
        "Bi-elliptic transfer takes "
        + str(round((dt_biell - dt_hohmann) / 3600, 4))
        + " hours longer"
    )

    return None  # curtis_ex6_3()


def curtis_ex6_4():
    """
    Curtis, p.
    TODO clean up this example description.
    Spacecraft at A and B are in the same orbit (1).
    At the instant shown, the chaser vehicle at A executes a phasing
    maneuver so as to catch the target spacecraft back at A after just
    one revolution of the chaser’s phasing orbit (2).
    What is the required total delta-v?

    Initial orbit is an ellipse given by A and C.
    Phasing orbit (of A) reduces the apogee to D
    Inital difference in true anomaly is 90 deg

    Given:

    Find:


    Notes:
    ----------

        References: see list at file beginning.
    """
    r_a = 6800
    r_c = 13600
    mu_e = 398600
    d_theta = 90 * (np.pi / 180)

    # Part 1: what is the time difference (for anomaly of 90deg)

    e_o1 = (r_c - r_a) / (r_c + r_a)

    E_B = E_from_theta(e_o1, d_theta)
    Me_B = E_B - e_o1 * np.sin(E_B)
    h_A = orbit_equation_h(r_a, mu_e, e_o1, 0)
    T_o1 = t_ellipse(r_a, r_c, mu_e)

    dt = t_from_Me(Me_B, mu_e, h_A, e_o1)

    T_phase = T_o1 - dt

    a_phase = (T_phase * np.sqrt(mu_e) / (2 * np.pi)) ** (2 / 3)

    ra_phase = 2 * a_phase - r_a

    v_o1 = v_ellipse_peri(r_a, r_c, mu_e)
    v_phase = v_ellipse_peri(r_a, ra_phase, mu_e)
    delta_v = v_o1 - v_phase

    # Total maneuver is double
    total_delta_v = 2 * delta_v

    return None  # curtis_ex6_4()


def test_curtis_ex6_1():
    print(f"\nTest Curtis example 6.1, ... :")
    # function does not need input parameters.
    curtis_ex6_1()
    return None


def test_curtis_ex6_2():
    print(f"\nTest Curtis example 6.2, ... :")
    # function does not need input parameters.
    curtis_ex6_2()
    return None


def test_curtis_ex6_3():
    print(f"\nTest Curtis example 6.3, ... :")
    # function does not need input parameters.
    curtis_ex6_3()
    return None


def test_curtis_ex6_4():
    print(f"\nTest Curtis example 6.4, ... :")
    # function does not need input parameters.
    curtis_ex6_4()
    return None


# use the following to test/examine functions
if __name__ == "__main__":

    # test_curtis_ex6_1()  # test curtis example 6.1
    # test_curtis_ex6_2()  # test curtis example 6.2
    test_curtis_ex6_3()  # test curtis example 6.3
    # test_curtis_ex6_4()  # test curtis example 6.4
