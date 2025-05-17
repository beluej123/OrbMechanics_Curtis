"""
Curtis chapter 6, orbital maneuvers examples collection.
Notes:
----------
    This file is organized with each example as a function, i.e.
        def curtis_ex6_1():
    General orbit parameter variable naming convention examples:
        rp_a1 = radius periapsis, position a, orbit1
        vp_o1 = velocity periapsis, orbit1
        t_o1 = period/time, orbit1
        t_ab1 = time from a->b, orbit 1
References:
----------
    See references.py for references list.
"""

import math

import numpy as np

from constants_1 import DEG2RAD, GM_EARTH_KM, PI, RAD2DEG, RADI_EARTH, TAU
from func_gen import (
    bielliptic_circular,
    delta_mass,
    ecc_conic_rv,
    ecc_from_ra_rp,
    ecc_from_rp_sma,
    ecc_from_ta1_ta2,
    hohmann_transfer,
    r_conic,
    v_circle,
    v_conic,
    v_ellipse_apo,
    v_ellipse_peri,
)


# TODO remove redundant functions, below, collected in the various sections.
def delta_v_r1v1r2v2(r1_vec, v1_vec, r2_vec, v2_vec):
    """
    Not sure what I had in mind for this function.
    Find delta_v_mag, given r1_vec, v1_vec, r2_vec, v2_vec.
    """
    delta_v_vec = v2_vec - v1_vec
    delta_v_mag = np.linalg.norm(delta_v_vec)
    return delta_v_mag


def ea_from_theta(ecc, theta):
    """eccentric anomaly; Curtis [9] p.146, eqn3.13b, note example 6.4"""
    a_ = math.sqrt((1 - ecc) / (1 + ecc))
    b_ = np.tan(theta / 2)
    return 2 * np.arctan(a_ * b_)


def h_from_ta_r_ecc(r, mu, ecc, ta):
    """
    h (angular momentum) calculated from ta (true anomaly) et.al.
        Curtis [9] example 3.6 (from eqn 2.45, p.72) & 6.4
        ecc=eccentricity, ta=true anomaly
    note, orbit_equation_h() in example3_x.py
    """
    a_ = r * mu
    b_ = 1 + ecc * np.cos(ta)
    return math.sqrt(a_ * b_)


def h_from_rp_ra(rp, ra, mu):
    """
    h (angular momentum) calculated from ta's (true anomalies) & radius.
        Curtis [9] example 6.5
    """
    a_ = math.sqrt(2 * mu)
    b_ = math.sqrt(rp * ra / (rp + ra))
    return a_ * b_


def h_from_ta1_ta2(r_1, r_2, ta_1, ta_2, mu):
    """
    h (angular momentum) calculated from ta (true anomalies) & radius.
        Curtis [9] example 6.6, p.307
    """
    a_ = math.sqrt(mu * r_1 * r_2)
    b_ = math.sqrt(math.cos(ta_1) - math.cos(ta_2))
    c_ = 1 / math.sqrt(r_1 * math.cos(ta_1) - r_2 * math.cos(ta_2))
    return a_ * b_ * c_


def t_from_me(me, mu, h, ecc):
    """
    Time since periapsis.
    me = mean anomaly;  fraction of elliptical orbit period elapsed since
        passing periapsis. me = E - ecc * sin(E)
        Note Curtis [9] p.143 near page bottom; also note example 6.4
    """
    a_ = me
    b_ = (mu**2) / (h**3)
    c_ = (1 - ecc**2) ** 1.5
    return a_ / (b_ * c_)


def t_ellipse(r_p, r_a, mu):
    """Ellipse period."""
    sma = (r_a + r_p) / 2
    return (TAU / math.sqrt(mu)) * sma**1.5


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

    orbit3_apo = 16000 + r_ea  # circular

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
    print(f"mission total delta v = {total_dv:0.5f} [km/s]")

    # part c;
    # convert specific impulse defined for 9.807 [m/s^2] not [km/s^2]
    d_mass = mass_sc * delta_mass(dv_km=total_dv, isp=isp)
    print(f"s/c needs propellant mass = {d_mass:0.4f} [kg]")


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
    # reminder sma = semi-major axis
    sma_o2 = (ra_o2 + rp_o2) / 2  # transfer ellipse, i.e. orbit2

    t_o2 = (TAU / math.sqrt(mu_e)) * sma_o2**1.5
    time_taken = t_o2 / 2

    t_o3 = (TAU / math.sqrt(mu_e)) * rp_o2**1.5

    orbital_portion = time_taken / t_o3
    orbital_angle = orbital_portion * 360

    print(f"given; hyperbolic closest: {r_hyp:0.5f} [km]")
    print(f"given; hyperbolic velocity @ r_hyp: {v_hyp:0.5f} [km/s]")
    print(f"given; inner orbit altitude: {ra_o2:0.5f} [km]")

    print(f"\norbit transfer time: {time_taken:0.5f} [sec], {time_taken/60:0.5f} [min]")
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
    print(f"enter Earth delta v: {total_dv:0.5f} [km/s]")
    print(f"% of spacecraft mass, propellant: {d_mass:0.5f} [%]")


def curtis_ex6_3(ra=None, rb=None, rc=None):
    """
    Explore transfer delta_v's. hohmann bielliptic.
        Curtis [9], pp296, example 6.3.

    Input Args:
    ----------
        ra
        rb
        rc

    Find:
    ----------
    Total delta-v requirement for a bi-elliptical Hohmann transfer
        from a geocentric circular orbit of 7000 km radius to one of 105,000 km
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
    # initial earth orbit velocity
    v_o1 = v_circle(mu=mu_e, r=r_o1)  # initial orbit velocity

    # start with bi-elliptic calculations; 4-orbits
    #   transfer time, delta v
    tt_biell, dv_biell = bielliptic_circular(r_o1, r_o2, r_o3, mu_e)

    # next, hohmann transfer; orbit 5
    # tt_hohmann is transfer time for hohmann orbit
    tt_hohmann, delta_v1, delta_v2, ecc_trans_1 = hohmann_transfer(r_o1, r_o3, mu_e)
    dv_hohmann = delta_v1 + delta_v2

    print(f"initial circular orbit, v_o1: {v_o1:0.5f} [km/s]")
    print(f"Bi-elliptic transfer delta_v = {dv_biell:0.5f}")
    print(
        f"Bi-elliptic transfer time = {tt_biell:0.5f} [s], {tt_biell/(3600*24):0.5f} [day]"
    )

    print(f"\nHohmann transfer eccentricity, ecc_trans_1: {ecc_trans_1:0.5f}")
    print(f"Hohmann transfer delta_v = {dv_hohmann:0.5f} [km/s]")
    print(
        f"Hohmann transfer time = {tt_hohmann:0.5f} [s], {tt_hohmann/(3600*24):0.5f} [day]"
    )
    print("")
    if dv_biell < dv_hohmann:
        print(
            f"Bi-elliptic transfer more efficient by {(dv_hohmann - dv_biell)} [km/s]"
        )

    # Compare flight times
    print(
        f"Bi-elliptic transfer takes {(tt_biell - tt_hohmann) / 3600:0.5f} hours longer"
    )


def curtis_ex6_4():
    """
    Catch target phasing maneuver.
        Includes number of orbits for chase to catch target.
        Curtis [9] example 6.4, pp299.
    General orbit parameter variable naming convention:
        parameter_position orbit number. for example;
            rp_a1 = radius periapsis, position a, orbit1
    Given:
    ----------
        rp_a1  : 600 [km] radius periapsis, orbit1
        ra_c1  : 13,600 [km] radius apoapsis, orbit1
        num_o2 : number of orbits for chase to catch target, orbit2
    Find:
    ----------
    Chasecraft at A, target at B. Both are in the same orbit (1).
        Chasecraft at A executes a phasing maneuver to catch the target.
        Chasecraft may take one or more revolutions to randuvous with target; orbit (2).
        Find total delta-v.

    References:
    ----------
    See references.py for references list.
    """
    rp_a1 = 6800  # [km] radius periapsis, position a, orbit1
    ra_c1 = 13600  # [km] radius apoapsis, position c, orbit1
    mu_e = GM_EARTH_KM.magnitude  # [km^3/s^2] earth mu; strip units
    d_theta = 90 * DEG2RAD.magnitude  # strip units
    num_o2 = 1  # number of orbits for chase to catch target

    ecc_o1 = ecc_from_ra_rp(ra=ra_c1, rp=rp_a1)  # ecc orbit 1

    # calculate timing (for spacecraft at true anomaly of d_theta)
    ea_b1 = ea_from_theta(ecc_o1, d_theta)  # eccentric anomaly, position b orbit 1
    me_b1 = ea_b1 - ecc_o1 * np.sin(ea_b1)  # mean anomaly, position b, orbit 1
    h_o1 = h_from_ta_r_ecc(r=rp_a1, mu=mu_e, ecc=ecc_o1, ta=0)  # ta=true anomaly

    # period/time orbit 1; Curtis [9] p.300
    t_o1 = t_ellipse(rp_a1, ra_c1, mu_e)  # time/period of orbit 1
    # time from a->b, orbit 1; Curtis [9] p.301
    t_ab1 = t_from_me(me_b1, mu_e, h_o1, ecc_o1)

    # include number of orbits of chase to catch target
    t_o2 = t_o1 - (t_ab1 / num_o2)  # period/time phasing orbit (orbit 2)

    sma_o2 = (t_o2 * math.sqrt(mu_e) / TAU) ** (2 / 3)  # sma=semi-major axis
    ra_d2 = 2 * sma_o2 - rp_a1  # [km] radius apoapsis, position d, orbit2
    # note, for this exercise rp_a2 = rp_a1
    ecc_o2 = ecc_from_rp_sma(rp=rp_a1, sma=sma_o2)  # ecc orbit 2

    vp_o1 = v_ellipse_peri(rp_a1, ra_c1, mu_e)
    vp_o2 = v_ellipse_peri(rp_a1, ra_d2, mu_e)
    delta_v = vp_o1 - vp_o2

    # total maneuver is double
    total_delta_v = 2 * delta_v
    print(f"  number of orbits for chase: {num_o2}")  # half orbit 1 period
    print(
        f"  period orbit 1: {t_o1:0.5f} [s], {t_o1/3600:0.5f} [hr]"
    )  # half orbit 1 period
    print(f"  t a->b orbit 1: {t_ab1:0.5f} [s], {t_ab1/3600:0.5f} [hr]")
    print(f"  ecc orbit 1: {ecc_o1:0.5f} [-]")
    print(f"  ecc orbit 2: {ecc_o2:0.5f} [-]")
    print(f"  velocity periapsis orbit 1: {vp_o1:0.5f} [km/s]")
    print(f"  velocity periapsis orbit 2: {vp_o2:0.5f} [km/s]")
    print(f"  phasing delta_v: {total_delta_v:0.5f} [km/s]")


def curtis_ex6_5():
    """
    Geostationary longitude shift, 12 degrees westward.
        Includes number of orbits for shift.
        Curtis [9] example 6.5, pp301.
    Given:
    ----------
        shift_long : 12 [deg] west shift longitude
        num_o2     : number of orbits for shift, orbit2
    Find:
    ----------
        Find delta-v for geostationary longitude shift

    References:
    ----------
    See references.py for references list.
    """
    shift_long = 12 * DEG2RAD.magnitude  # strip units
    num_o2 = 3  # number of orbits to make longitude shift
    mu_e = GM_EARTH_KM.magnitude  # [km^3/s^2] earth mu; strip units

    # geostationary orbital period = sidereal day; period of orbit 1
    t_o1 = (23 * 3600) + (56 * 60) + 4.09  # [s] sidereal period
    #  geosynchronous angular rate/velocity
    geo_theta_rate = TAU / t_o1  # [rad/s]
    # GEO radius,  Curtis [9] p.78, eqn 2.68
    r_o1 = (mu_e / (geo_theta_rate**2)) ** (1 / 3)  # [km]
    v_o1 = v_circle(r=r_o1, mu=mu_e)

    print(f"  GEO angular rate: {geo_theta_rate:0.7g} [rad/s]")
    print(f"  GEO period (orbit1): {t_o1:0.7g} [s]")
    print(f"  GEO radius, r_01 (orbit1): {r_o1:0.7g} [km]")
    print(f"  GEO velocity, v_01 (orbit1): {v_o1:0.7g} [km/s]")

    # calculate 3*orbit1 + 12 degrees;
    #   remember slower (+) means drift west
    t_o2 = (num_o2 * TAU + shift_long) / (num_o2 * geo_theta_rate)
    sma_o2 = (t_o2 * math.sqrt(mu_e) / TAU) ** (2 / 3)
    ra_o2 = 2 * sma_o2 - r_o1
    ecc_o2 = ecc_from_ra_rp(ra=ra_o2, rp=r_o1)  # ecc orbit 1
    h_o2 = h_from_rp_ra(rp=r_o1, ra=ra_o2, mu=mu_e)
    vp_o2 = h_o2 / r_o1
    dv_o2 = vp_o2 - v_o1
    dv_o1 = v_o1 - vp_o2
    tdv = abs(dv_o2) + abs(dv_o1)

    print(f"  orbit shift apoapsis, ra_02 (orbit2): {ra_o2:0.7g} [km]")
    print(f"  orbit shift period (orbit2): {t_o2:0.7g} [s]")
    print(f"  orbit shift sma (orbit2): {sma_o2:0.7g} [km]")
    print(f"  orbit shift ecc (orbit2): {ecc_o2:0.7g} [km]")
    print(f"  h orbit2 : {h_o2:0.7g} [km^3/s^2]")
    print(f"  velocity, vp_02 (orbit2): {vp_o2:0.7g} [km/s]")

    print(f"\n  delta velocity 1 (o1->o2): {dv_o2:0.7g} [km/s]")
    print(f"  delta velocity 2 (o2->o2): {dv_o1:0.7g} [km/s]")
    print(f"  mission delta velocity (01-o2, o2->o2): {tdv:0.7g} [km/s]")


def curtis_ex6_6():
    """
    Non-Hohmann transfer, earth centered.
        Curtis [9] example 6.6, pp305.
    Given:
    ----------
        mu_e  : Central body gravitational parameter, Earth
        ta_a1 : 150 [deg->rad] true anomaly, position a, orbit 1
        ra_b1 : 20,000 [km] radai apoapsis, position b,  orbit 1
        rp_c1 : 10,000 [km] radai periapsis, position c,  orbit 1
        rp_d2 : 6,378 [km] radai periapsis, position d,  orbit 2
    Find:
    ----------

    References:
    ----------
    See references.py for references list.
    """
    # given
    mu_e = GM_EARTH_KM.magnitude  # [km^3/s^2] earth mu; strip units
    ta_a1 = 150 * DEG2RAD.magnitude  # [rad] strip units
    ra_b1 = 20000  # [km]
    rp_c1 = 10000  # [km]
    rp_d2 = 6378  # [km]

    # orbit 1, Curtis [9] pp306
    ecc_o1 = ecc_from_ra_rp(ra=ra_b1, rp=rp_c1)  # ecc orbit 1
    h_o1 = h_from_rp_ra(rp=rp_c1, ra=ra_b1, mu=mu_e)
    r_a1 = r_conic(h=h_o1, ecc=ecc_o1, ta=ta_a1, mu=mu_e)
    # transverse velocity, position a, orbit1
    vt_a1 = h_o1 / r_a1
    # [km/s] radial velocity, position a, orbit1
    vr_a1 = (mu_e / h_o1) * ecc_o1 * math.sin(ta_a1)
    # [km/s] velocity, position a, orbit1
    v_a1 = math.sqrt(vt_a1**2 + vr_a1**2)
    # [rad] flight path angle, position a, orbit 1
    fpa_a1 = math.atan(vr_a1 / vt_a1)

    print(f"  ecc orbit1: {ecc_o1:0.7g} [-]")
    print(f"  h orbit1 : {h_o1:0.7g} [km^3/s^2]")
    print(f"  radial, position a, orbit1 : {r_a1:0.7g} [km^3/s^2]")
    print(f"  velocity, position a, orbit1 : {v_a1:0.7g} [km^3/s^2]")
    fpa_a1_deg = fpa_a1 * (RAD2DEG.magnitude)  # convenience variable
    print(f"  flight path angle, position a, orbit1 : {fpa_a1_deg:0.7g} [deg]")

    # orbit 2, similar to orbit 1
    ecc_o2 = ecc_from_ta1_ta2(r_1=rp_d2, ta_1=0.0, r_2=r_a1, ta_2=ta_a1)
    h_o2 = h_from_ta1_ta2(r_1=rp_d2, r_2=r_a1, ta_1=0.0, ta_2=ta_a1, mu=mu_e)
    r_a2 = r_a1
    # transverse velocity, position a, orbit2
    vt_a2 = h_o2 / r_a2
    # [km/s] radial velocity, position a, orbit2
    vr_a2 = (mu_e / h_o2) * ecc_o2 * math.sin(ta_a1)
    # [km/s] velocity, position a, orbit1
    v_a2 = math.sqrt(vt_a2**2 + vr_a2**2)
    # [rad] flight path angle, position a, orbit 1
    fpa_a2 = math.atan(vr_a2 / vt_a2)

    print(f"  ecc orbit2: {ecc_o2:0.7g} [-]")
    print(f"  h orbit2 : {h_o2:0.7g} [km^3/s^2]")
    print(f"  radial, position a, orbit2 : {r_a2:0.7g} [km^3/s^2]")
    print(f"  velocity, position a, orbit2 : {v_a2:0.7g} [km^3/s^2]")
    fpa_a2_deg = fpa_a2 * (RAD2DEG.magnitude)  # convenience variable
    print(f"  flight path angle, position a, orbit1 : {fpa_a2_deg:0.7g} [deg]")

    # evaluate orbit 1 & orbit 2 delta's
    dfpa_a = fpa_a2 - fpa_a1  # delta fpa at position a, orbit1->orbit2
    dv_vec_a = math.sqrt(v_a1**2 + v_a2**2 - 2 * v_a1 * v_a2 * math.cos(dfpa_a))
    dv_mag_a = v_a2 - v_a1  # delta_v magnitude, not the same as delta_v vector

    dfpa_a_deg = dfpa_a * (RAD2DEG.magnitude)  # convenience variable
    print(f"  delta fpa_a, position a : {dfpa_a_deg:0.7g} [deg]")
    print(f"  delta_v_vec, position a, orbit1->orbit2 : {dv_vec_a:0.7g} [km/s]")
    print(f"  delta_v_mag, position a, orbit1->orbit2 : {dv_mag_a:0.7g} [km/s]")


def curtis_ex6_9():
    """
    Chase/intercept with vectors, earth centered.
        Curtis [9] example 6.9, pp314.
    Given:
    ----------
        mu_e  : Central body gravitational parameter, Earth
        ta_b1 : 45 [deg->rad] true anomaly, position b, orbit 1
        ta_c1 : 180-30 [deg->rad] true anomaly, position c, orbit 1
        ra_1  : 18,900 [km] radai apoapsis, orbit 1
        rp_1  : 8,100 [km] radai periapsis, orbit 1
    Find:
    ----------

    References:
    ----------
    See references.py for references list.
    """
    # given
    mu_e = GM_EARTH_KM.magnitude  # [km^3/s^2] earth mu; strip units
    ta_a1 = 45 * DEG2RAD.magnitude  # [rad] strip units
    ta_c1 = (180 - 30) * DEG2RAD.magnitude  # [rad] strip units
    ra_o1 = 18900  # [km]
    rp_o1 = 8100  # [km]

    # orbit 1 from given args
    ecc_o1 = ecc_from_ra_rp(ra=ra_o1, rp=rp_o1)
    h_o1 = h_from_rp_ra(rp=rp_o1, ra=ra_o1, mu=mu_e)
    # period/time orbit 1
    t_o1 = t_ellipse(r_a=ra_o1, r_p=rp_o1, mu=mu_e)
    # peri-focal vectorize r & v @ position b

    print(f"  ecc orbit1: {ecc_o1:0.7g} [-]")
    print(f"  h orbit1 : {h_o1:0.7g} [km^3/s^2]")
    print(f"  period orbit 1: {t_o1:0.5f} [s], {t_o1/3600:0.5f} [hr]")


def test_curtis_ex6_1():
    """Curtis [9] pp290, example 6.1."""
    print("\nTest Curtis example 6.1:")
    curtis_ex6_1()


def test_curtis_ex6_2():
    """Curtis [9] pp294, example 6.2."""
    print("\nTest Curtis example 6.2:")
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
    Catch target phasing maneuver for rendezvous.
    """
    print("\nTest Curtis example 6.4, Catch Target Phasing Maneuver:")
    curtis_ex6_4()


def test_curtis_ex6_5():
    """
    Phasing maneuvers.
    """
    print("\nTest Curtis example 6.5, Geostationary longitude shift:")
    curtis_ex6_5()


def test_curtis_ex6_6():
    """
    Non-hohmann transfer.
    """
    print("\nTest Curtis example 6.6, Non-Hohmann Transfer:")
    curtis_ex6_6()


def test_curtis_ex6_9():
    """
    Chase with vectors; maybe easier, we will see.
    """
    print("\nTest Curtis example 6.8, Chase with vectors:")
    curtis_ex6_9()


def test_delta_v_r1v1r2v2():
    """
    Find delta_v given r1_vec, v1_vec
        Needs descriptive work...
    """
    print("\nTest find delta_v given r1_vec, v1_vec")
    r1_vec = np.array([])
    v1_vec = np.array([])
    r2_vec = np.array([])
    v2_vec = np.array([])
    # h1_vec = np.cross(r1_vec, v1_vec)
    # h2_vec = np.cross(r2_vec, v2_vec)
    delta_v_mag = delta_v_r1v1r2v2(
        r1_vec=r1_vec, v1_vec=v1_vec, r2_vec=r2_vec, v2_vec=v2_vec
    )
    print(f"delta_v_mag: {delta_v_mag:0.5f} [km/s]")


# use the following to test/examine functions
if __name__ == "__main__":

    # test_curtis_ex6_1()  # Hohmann transfer delta_v's
    # test_curtis_ex6_2()  # Hyperbolic transfer to Hohmann Earth
    # test_curtis_ex6_3()  # bi-elliptic maneuvers
    # test_curtis_ex6_4()  # catch target phasing maneuver
    # test_curtis_ex6_5()  # orbit shift phasing maneuver
    # test_curtis_ex6_6()  # non-hohmann transfer
    test_curtis_ex6_9()  # chase with vectors
    # test_delta_v_r1v1r2v2()
