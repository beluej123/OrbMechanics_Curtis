"""
Curtis chapter 6, orbital maneuvers examples collection.

Notes:
----------
    This file is organized with each example as a function, i.e.
        def curtis_ex6_1():
References:
----------
    See references.py for references list.
"""

import math

import numpy as np

from constants_1 import DEG2RAD, GM_EARTH_KM, RADI_EARTH
from func_gen import (
    delta_mass,
    ecc_conic_rv,
    hohmann_transfer,
    v_circle,
    v_conic,
    v_ellipse_apo,
    v_ellipse_peri,
)


# TODO remove redundant functions, below, collected in the various sections.
def delta_v_r1v1r2v2(r1_vec, v1_vec, r2_vec, v2_vec):
    """
    Find delta v, given r1_vec, v1_vec, r2_vec, v2_vec.
    """
    delta_v_vec = v2_vec - v1_vec
    delta_v_mag = np.linalg.norm(delta_v_vec)
    return delta_v_mag


def delta_v_bielliptic_circular(r_a, r_b, r_c, mu):
    """
    Curtis [9] pp296, example 6.3. Hohmann transfer.
        Inner/outer circular radius, co-planar
    Input Parameters:
    ----------
        r_a: Radius of the initial circular orbit.
        r_b: Radius of the final circular orbit.
        mu: central body gravitational parameter
    """
    # rb is transfer ellipse
    a = r_c / r_a
    b = r_b / r_a
    A = math.sqrt((2 * (a + b)) / (a * b))
    B = -1 * ((1 + math.sqrt(a)) / math.sqrt(a))
    C = -1 * math.sqrt(2 / (b * (1 + b))) * (1 - b)
    D = math.sqrt(mu / r_a)
    total_delta_v = (A + B + C) * D
    return total_delta_v


def ea_from_theta(ecc, theta):
    """eccentric anomaly; Curtis [9] p.146, eqn3.13b, note example 6.4"""
    a_ = math.sqrt((1 - ecc) / (1 + ecc))
    b_ = np.tan(theta / 2)
    return 2 * np.arctan(a_ * b_)


def orbit_equation_h(r, mu, ecc, theta):
    """inspired by example 6.4"""
    a_ = r * mu
    b_ = 1 + ecc * np.cos(theta)
    return math.sqrt(a_ * b_)


def t_from_Me(Me, mu, h, ecc):
    """2*pi/T ; Curtis [9] p.143 near page bottom; note example 6.4"""
    a_ = Me
    b_ = (mu**2) / (h**3)
    c_ = (1 - ecc**2) ** 1.5
    return a_ / (b_ * c_)


def t_ellipse(r_p, r_a, mu):
    """Ellipse period."""
    sma = (r_a + r_p) / 2
    return ((2 * np.pi) / math.sqrt(mu)) * sma**1.5


def curtis_ex6_1():
    """
    Calculate Hohmann delta_v's.
    Curtis [3], pp323, Curtis [9], pp290, example 6.1.
        Started to fool with units-aware.  Also see Orbit_from_r0v0.py
        Initial spacecraft Earth orbit, 480km by 800km.
        (a) find spacecraft velocity(perigee) for a 480km by 16,000km transfer orbit.
        (b) find v (apogee kick) required at B of the transfer orbit to
            establish a circular orbit of 16000km altitude (orbit 3)
        (c) total propellant if specific impulse is 300s
    Given:
        earth altitude's r1 & r2 (not vectors), dt, and delta true anomaly;
    Find:
        periapsis altitude, time to periapsis

    References:
    ----------
        See references.py for references list.
    """
    print("Hohmann transfer delta_v's & delta_mass:")
    mu_e = GM_EARTH_KM.magnitude  # earth mu [km^3/s^2]
    r_ea = RADI_EARTH.magnitude  # earth radius [km]
    mass_sc = 2000  # [kg]
    isp = 300  # [s] specific impulse

    orbit1_peri = 480 + r_ea  # [km]
    orbit1_apo = 800 + r_ea

    orbit2_peri = 480 + r_ea
    orbit2_apo = 16000 + r_ea

    orbit3_apo = 16000 + r_ea # circular

    # part a
    v_peri_o1 = v_ellipse_peri(orbit1_peri, orbit1_apo, mu_e)
    v_peri_o2 = v_ellipse_peri(orbit2_peri, orbit2_apo, mu_e)
    dv_1 = v_peri_o2 - v_peri_o1
    print(f"pre-burn, v_peri_o1 = {v_peri_o1:0.5f} [km/s]")
    print(f"post-burn, v_peri_o2 = {v_peri_o2:0.5f} [km/s]")
    print("first delta_v to transfer ellipse (orbit 2):")
    print(f"  dv_1 = {dv_1:0.5f} [km/s]")

    # part b
    v_apo_o2 = v_ellipse_apo(orbit2_peri, orbit2_apo, mu_e)
    # v_conic velocity at position r for any conic section
    #   v_cir_03 means sma = apoapsis (or periapsis)
    v_cir_o3 = v_conic(r=orbit3_apo, sma=orbit3_apo, mu=mu_e)
    dv_2 = v_cir_o3 - v_apo_o2
    total_dv = dv_1 + dv_2  # [km/s]
    print(f"pre-burn is ellipse, v_apo_o2 = {v_apo_o2:0.5f} [km/s]")
    print(f"post-burn is circle, v_cir_o3 = {v_cir_o3:0.5f} [km/s]")
    print("2nd delta_v to transfer circle (orbit 3):")
    print(f"  dv_2 = {dv_2:0.5f} [km/s]")
    print(f"total delta v = {total_dv:0.5f} [km/s]")

    # part c;
    # convert specific impulse defined for 9.807 [m/s^2] not [km/s^2]
    d_mass = mass_sc * delta_mass(dv_km=total_dv, isp=isp)
    print(f"propellant mass = {d_mass:0.4f} [kg]")


def curtis_ex6_2():
    """
    Lunar return to earth to rendezvous with space station.
    Curtis [3] p.326, Curtis [9] pp294, example 6.2
    A lunar return mission approaches earth on a hyperbolic trajectory. The
        closest Earth approach altitude is 5000 km, at 10 km/s.  Then lower
        the spacecraft to a 500 km circular orbit to rendezvous with a
        space station.
        Find the location of the space station at retrofire to rendezvous.
    Given:
    ----------
        Earth entry: altitude 500km, velocity 10km/s; entry ellipse apoapsis
            for circular final orbit.
    Find:
    ----------
        Entry transfer ellipse parameters. Time to circularization and delta_v.

    References:
    ----------
        See references.py for references list.
    """
    print("Hyperbolic Earth approach, Hohmann transfer to circular orbit:")
    mu_e = GM_EARTH_KM.magnitude  # earth mu [km^3/s^2]
    r_ea = RADI_EARTH.magnitude  # earth radius [km]
    r_hyp = 5000 + r_ea  # [km] approach hyperbola, orbit1
    v_hyp = 10  # [km/s]
    isp = 300  # [s] propellant specific impulse
    # ecc_conic_rv()  may use r(periapsis) or r_vec; and v(periapsis) or v_vec
    ecc_hyp = ecc_conic_rv(r_peri=r_hyp, v_peri=v_hyp, mu=mu_e)
    print(f"Orbit1 hyperbolic eccentricity = {ecc_hyp}")

    ra_o2 = 5000 + r_ea  # transfer ellipse = orbit2; apoapsis
    rp_o2 = 500 + r_ea  # transfer ellipse = orbit2; periapsis

    sma_o2 = (ra_o2 + rp_o2) / 2  # transfer ellipse, i.e. orbit2

    t_o2 = ((2 * np.pi) / math.sqrt(mu_e)) * sma_o2**1.5

    time_taken = t_o2 / 2

    t_o3 = ((2 * np.pi) / math.sqrt(mu_e)) * rp_o2**1.5

    orbital_portion = time_taken / t_o3
    orbital_angle = orbital_portion * 360

    print(f"given; hyperbolic closest: {r_hyp:0.5f} [km]")
    print(f"given; hyperbolic velocity @ r_hyp: {v_hyp:0.5f} [km/s]")
    print(f"given; inner orbit altitude: {ra_o2:0.5f} [km]")

    print(f"\norbit transfer time: {time_taken:0.5f} [sec]")
    print(f"\norbit transfer time: {time_taken/60:0.5f} [min]")
    print(f"rendezvous phasing: {orbital_angle:0.5f} [deg]")

    v_apo_o2 = v_ellipse_apo(rp_o2, ra_o2, mu_e)
    v_peri_o2 = v_ellipse_peri(rp_o2, ra_o2, mu_e)
    v_cir_o3 = v_circle(rp_o2, mu_e)
    dv_1 = abs(v_apo_o2 - v_hyp)
    dv_2 = abs(v_cir_o3 - v_peri_o2)
    total_dv = dv_1 + dv_2  # [km/s]
    # compute % of s/c that needs to be propellant
    d_mass = delta_mass(dv_km=total_dv, isp=isp)

    print(f"\nv_hyp = {v_hyp:0.5f} [km/s]")
    print(f"v_apo_o2 = {v_apo_o2:0.5f} [km/s]")
    print(f"v_peri_o2 = {v_peri_o2:0.5f} [km/s]")
    print(f"v_cir_o3 = {v_cir_o3:0.5f} [km/s]")
    print(f"dv_1 = {dv_1:0.5f} [km/s]")
    print(f"dv_2 = {dv_2:0.5f} [km/s]")
    print(f"total_dv: {total_dv:0.5f} [km/s]")
    print(f"% of spacrcraft mass, propellant: {d_mass:0.5f} [%]")


def curtis_ex6_3(ra=None, rb=None, rc=None):
    """
    Explore transfer delta_v's. hohmann bielliptic.
    Curtis [9], pp296, example 6.3.

    Input Parameters:
    ----------
        ra
        rb
        rc

    Find:
    ----------
    Total delta-v requirement for a bi-elliptical Hohmann transfer
        from a geocentric circular orbit of 7000 km radius to one of 105 000 km
        radius. Let the apogee of the first ellipse be 210,000 km.
        Compare the delta-v schedule and total flight time with that for an
        ordinary single Hohmann transfer ellipse.
    References:
    ----------
        See references.py for references list.
    """
    if ra is not None:
        r_o1 = ra
    else:
        r_o1 = 7000  # [km]
    if rb is not None:
        r_o2 = rb
    else:
        r_o2 = 210000  # [km]
    if rc is not None:
        r_o3 = rc
    else:
        r_o3 = 105000  # [km]

    mu_e = GM_EARTH_KM.magnitude  # earth mu [km^3/s^2]

    # Compare delta v
    # dv_hohmann = delta_v_hohmann_circular(r_o1, r_o2, mu_e)
    trans_time, delta_v1, delta_v2, trans_ecc = hohmann_transfer(r_o1, r_o3, mu_e)
    dv_hohmann = delta_v1 + delta_v2
    print(f"initial circular orbit, delta_v1: {delta_v1}")

    dv_biell = delta_v_bielliptic_circular(r_o1, r_o2, r_o3, mu_e)

    if dv_biell < dv_hohmann:
        print(
            "Bi-elliptic transfer more efficient by "
            + str(round(dv_hohmann - dv_biell, 4))
            + " km/s"
        )

    # Compare flight times
    # Hohmann:
    dt_hohmann = t_ellipse(r_o1, r_o3, mu_e) / 2

    # Bi-elliptic:
    dt_biell_1 = t_ellipse(r_o1, r_o2, mu_e) / 2
    dt_biell_2 = t_ellipse(r_o2, r_o3, mu_e) / 2
    dt_biell = dt_biell_1 + dt_biell_2

    print(
        "Bi-elliptic transfer takes "
        + str(round((dt_biell - dt_hohmann) / 3600, 4))
        + " hours longer"
    )


def curtis_ex6_4():
    """
    Spacecraft phasing maneuvers. Curtis [9] pp299.
    TODO clean up this example description.
    Spacecraft at A and B are in the same orbit (1).
        A chaser vehicle at A executes a phasing maneuver to catch the target
        back at A after just one revolution of the chaser's phasing orbit (2).
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
    r_a = 6800  # [km]
    r_c = 13600  # [km]
    mu_e = GM_EARTH_KM.magnitude  # earth mu [km^3/s^2]
    d_theta = 90 * DEG2RAD

    # Part 1: what is the time difference (for anomaly of 90deg)
    ecc_o1 = (r_c - r_a) / (r_c + r_a)

    E_B = ea_from_theta(ecc_o1, d_theta)
    Me_B = E_B - ecc_o1 * np.sin(E_B)
    h_A = orbit_equation_h(r_a, mu_e, ecc_o1, 0)
    T_o1 = t_ellipse(r_a, r_c, mu_e)

    # Curtis [9] p.300
    dt = t_from_Me(Me_B, mu_e, h_A, ecc_o1)

    t_phase = T_o1 - dt

    a_phase = (t_phase * math.sqrt(mu_e) / (2 * np.pi)) ** (2 / 3)

    ra_phase = 2 * a_phase - r_a

    v_o1 = v_ellipse_peri(r_a, r_c, mu_e)
    v_phase = v_ellipse_peri(r_a, ra_phase, mu_e)
    delta_v = v_o1 - v_phase

    # Total maneuver is double
    total_delta_v = 2 * delta_v
    print(f"total_delta_v: {total_delta_v} [km/s]")


def test_curtis_ex6_1():
    """Curtis [9] pp290, example 6.1."""
    print("\nTest Curtis example 6.1, ... :")
    curtis_ex6_1()


def test_curtis_ex6_2():
    """Curtis [9] pp294, example 6.2."""
    print("\nTest Curtis example 6.2, ... :")
    curtis_ex6_2()


def test_curtis_ex6_3():
    """
    Compare delta_v hohmann vs. bielliptic.
    See function for input parameters. if none chosen, function uses book
        values.
    """
    print("\nTest Curtis example 6.3. Compare delta_v hohmann vs. bielliptic")
    curtis_ex6_3()


def test_curtis_ex6_4():
    """
    Phasing maneuvers.
    """
    print("\nTest Curtis example 6.4, Phasing Maneuvers:")
    curtis_ex6_4()


def test_delta_v_r1v1r2v2():
    """
    Test function
    """
    r1_vec = np.array([])
    v1_vec = np.array([])
    r2_vec = np.array([])
    v2_vec = np.array([])
    # h1_vec = np.cross(r1_vec, v1_vec)
    # h2_vec = np.cross(r2_vec, v2_vec)
    delta_v_mag = delta_v_r1v1r2v2(
        r1_vec=r1_vec, v1_vec=v1_vec, r2_vec=r2_vec, v2_vec=v2_vec
    )
    print(f"delta_v_mag: {delta_v_mag} [km/s]")


# use the following to test/examine functions
if __name__ == "__main__":

    test_curtis_ex6_1()  # Hohmann transfer delta_v's
    # test_curtis_ex6_2()  # Hyperbolic transfer to Hohmann Earth
    # test_curtis_ex6_3()  # bi-elliptic maneuvers
    # test_curtis_ex6_4()  # phasing maneuvers
    # test_delta_v_r1v1r2v2()
