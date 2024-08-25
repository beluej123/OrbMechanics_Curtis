"""
Chapter 8, Curtis examples collection.

Notes:
----------
    This file is organized with each example as a function, and all function test
        defined/enabled at the end of this file.
References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import math

import numpy as np  # for vector math


def curtis_ex8_4():
    """
    Curtis pp.446, example 8.4.  Earth->Mars Mission.
    Given:
        Earth orbit launch, alt=300 [km] circular, parabolic launch trajectory
            thus ecc=1, and Earth GM (or mu)
        r1: periapsis altitude 500 [km];
        r2: earth-sun SOI (sphere of influence); soi calculation known

    Find:
        (a) delta-v required
        (b) departure hyperbola perigee location
        (c) propellant as a percentage of the spacecraft, before delta-v burn
            assume Isp (specific impulse) = 300 [s]
    Notes:
    ----------
        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
        References: see list at file beginning.
    """
    # constants; mostly from Vallado not Curtis
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    mu_sun = GM_sun_km  # [km^3/s^2]
    mu_earth = GM_earth_km  # [km^3/s^2]

    r_earth_orb = 149598023  # [km], Vallado p.1041, tbl.D-3
    r_mars_orb = 227939186  # [km], Vallado p.1041, tbl.D-3

    r_earth = 6378.1363  # [km], Vallado p.1041, tbl.D-3
    alt_earth = 300  # [km], given altitude above earth

    # part a
    # Curtis p.442, eqn 8.35
    v_inf = math.sqrt(mu_sun / r_earth_orb) * (
        math.sqrt(2 * r_mars_orb / (r_earth_orb + r_mars_orb)) - 1
    )
    print(f"depart v_infinity, v_inf = {v_inf:.5g} [km/s]")

    # spacecraft speed in 300km circular parking orbit; Curtis p.444, eqn 8.41
    v_c = math.sqrt(
        mu_earth / (r_earth + alt_earth)
    )  # departure from circular parking orbit
    print(f"departure parking orbit, v_c= {v_c:.5g} [km/s]")

    # Delta_v required to enter departure hyperbola; eqn 8.42, p444
    delta_v = v_c * (math.sqrt(2 + (v_inf / v_c) ** 2) - 1)
    print(f"delta_v to enter departure hyperbola = {delta_v:.5g} [km/s]")

    # part b
    # Perigee of the departure hyperbola, relative to the earth’s orbital velocity vector
    # eqn 8.43, p444
    r_p = r_earth + alt_earth  # periapsis
    beta_depart = math.acos(1 / (1 + r_p * v_inf**2 / mu_earth))
    print(f"departure hyperbola beta angle= {beta_depart*180/math.pi:.5g} [deg]")
    ecc_depart = 1 + (r_p * v_inf**2) / mu_earth
    print(f"eccentricity, departure hyperbola = {ecc_depart:.5g}")

    # part c
    # Perigee can be located on either the sun lit or darkside of the earth.
    # It is likely that the parking orbit would be a prograde orbit (west to east),
    # which would place the burnout point on the darkside.
    I_sp = 300  # [s]
    g_0 = 9.81e-3  # [km/s^2]
    delta_mRatio = 1 - math.exp(-delta_v / (I_sp * g_0))
    print(f"Propellant mass ratio = {delta_mRatio:.5g}")
    return None  # curtis_ex8_4()


def curtis_ex8_5():
    """
    Curtis pp.456, example 8.5, Hohmann Earth->Mars.
    After a Hohmann transfer from earth to mars, calculate:
        (a) the minimum delta_v required to place spacecraft in an orbit with 7hour period
        (b) periapsis radius, the aiming radius and the angle between periapse and Mars’ velocity vector.
        (c) aiming radius
        (d) angle between periapsis and Mars' velocity vector

    Given:
        TODO fix this later
    Find:
        TODO fix this later
    Notes:
    ----------
        May help development; see https://github.com/jkloser/OrbitalMechanics
        Helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
        References: see list at file beginning.
    """
    mu_sun_km = 1.327e11  # [km^3/s^2]
    mu_earth_km = 398600  # [km^3/s^2]
    mu_mars_km = 42830  # [km^3/s^2]

    r_earth_orb = 149.6e6  # earth solar orbit [km]
    r_mars_orb = 227.9e6  # mars solar orbit [km]

    r_earth = 6378  # earth radius [km]
    r_mars = 3396  # earth radius [km]
    alt_earth = 300  # altitude above earth [km]

    T_mars_orb = 7 * 60 * 60  # satellite period in mars orbit [s]

    # part a
    # from eqn 8.4
    v_inf = math.sqrt(mu_sun_km / r_mars_orb) * (
        1 - math.sqrt(2 * r_earth_orb / (r_earth_orb + r_mars_orb))
    )
    print(f"arrive v_infinity, v_inf = {v_inf:.5g} [km/s]")

    # Semi-major axis of capture orbit
    a_capture = (T_mars_orb * math.sqrt(mu_mars_km) / (2 * math.pi)) ** (2 / 3)
    print(f"arrive semi-major axis = {a_capture:.5g} [km]")

    # from eqn 8.67, not sure my print description below is correct
    ecc_mars_orb = (2 * mu_mars_km / (a_capture * v_inf**2)) - 1
    print(f"eccentricity, at mars = {ecc_mars_orb:.5g}")

    # from eqn 8.70
    delta_v = v_inf * math.sqrt((1 - ecc_mars_orb) / 2)
    print(f"delta_v enter mars = {delta_v:.5g} [km/s]")

    # part b
    # periapsis radius at mars capture, from eqn 8.67
    r_p = (2 * mu_mars_km / v_inf**2) * (1 - ecc_mars_orb) / (1 + ecc_mars_orb)
    print(f"periapsisr_p at mars = {r_p:.5g} [km]")

    # part c
    # aiming radius from eqn 8.71
    aim_radius = r_p * math.sqrt(2 / (1 - ecc_mars_orb))
    print(f"aiming radius (aka delta) at mars = {aim_radius:.5g} [km]")

    # part d
    # angle to periapsis from eqn 8.43
    beta_p = math.acos(1 / (1 + r_p * v_inf**2 / mu_mars_km))
    print(f"angle to periapsis at mars = {(beta_p*180/math.pi):.5g} [km]")
    return None  # curtis_ex8_5()


def curtis_ex8_6():
    """
    Curtis pp.462, example 8.6.  Venus Fly-by Mission.
    Spacecraft departs earth with a velocity perpendicular to the sun line.
    Encounter occurs at a true anomaly in the approach trajectory of 30◦.
    Periapse altitude 300 km.
    (a) Dark side Venus apporach, show the post-flyby orbit is as shown in Figure 8.20.
    (b) Sunlit side Venus approach, show the post-flyby orbit is as shown in Figure 8.21.

    Leading-side flyby results in a decrease in the spacecraft's heliocentric speed.
    Trailing-side flyby increases helliocentric speed;
    e1, h1, and θ1 are eccentricity, angular momentum, and true anomaly of heliocentric approach trajectory.

    TODO update variable names to be consistant with darkside & lightside calculations
    Given:
        TODO update
    Find:
        TODO update
    Notes:
    ----------
        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
        References: see list at file beginning.
    """
    np.set_printoptions(precision=4)  # set vector printing

    mu_sun = 1.327e11  # [km^3/s^2]
    mu_venus = 324900  # [km^3/s^2]
    mu_earth = 398600  # [km^3/s^2]
    mu_mars = 42830  # [km^3/s^2]

    r_venus_orb = 108.2e6  # venus orbit around sun [km]
    r_earth_orb = 149.6e6  # earth orbit [km]
    r_mars_orb = 227.9e6  # mars orbit [km]

    r_venus = 6052  # venus radius [km]
    r_earth = 6378  # earth radius [km]
    r_mars = 3396  # mars radius [km]
    alt_earth = 300  # altitude above earth [km]
    alt_venus = 300  # altitude above venus [km]

    nu_venus = -30 * math.pi / 180  # venus approach true anomaly (nu); saved as [rad]

    # part a, Pre-Flyby ellipse; p.462+
    # orbit id (1), transfer orbit eccentricity; p.464
    ecc1_venus_orb = (r_earth_orb - r_venus_orb) / (
        r_earth_orb + r_venus_orb * math.cos(nu_venus)
    )
    print(f"eccentricity, at venus, ecc1_venus_orb = {ecc1_venus_orb:.5g}")

    # orbit 1 angular momentum; p.464
    h1 = math.sqrt(mu_sun * r_earth_orb * (1 - ecc1_venus_orb))
    print(f"angular momentum, orbit1, h1 = {h1:.5g} [km^2/s]")

    # Calculate spacecraft radial and transverse components heliocentric velocity at
    # the inbound crossing of Venus’s sphere of influence.
    v1_perp = h1 / r_venus_orb  # perpendicular velocity orbit 1[km/s]
    v1_radi = (
        (mu_sun / h1) * (ecc1_venus_orb) * math.sin(nu_venus)
    )  # radial velocity orbit 1[km/s]
    print(f"velocity inbound perpendicular, v1_perp = {v1_perp:.5g} [km/s]")
    print(f"velocity inbound radial, v1_radi = {v1_radi:.5g} [km/s]")

    # flight path angle; p.464; eqn 2.51 on p.xx
    # The following negative sign is consistent with the spacecraft flying towards
    #   perihelion of the pre-flyby elliptical trajectory (orbit 1).
    gamma1 = math.atan(v1_radi / v1_perp)
    print(f"flight path angle, gamma1 = {gamma1*180/math.pi:.5g} [deg]")
    # Speed of the space vehicle at the inbound crossing
    v_in = math.sqrt(v1_perp**2 + v1_radi**2)
    print(f"velocity inbound (from SOI) = {v_in:.5g} [km/s]")

    # part a, Flyby Hyperbola; p.464+
    print("********** darkside flyby hyperbola **********")
    # velocity inbound (1) vector, planet, sun direction coordinates
    v1p_vec = np.array([v1_perp, -v1_radi])  # [km/s]
    v1p_mag = np.linalg.norm(v1p_vec)  # [km/s]
    print(f"velocity inbound vector, v1p_vec = {v1p_vec} [km/s]")
    # assume venus in circular orbit; velocity planet (venus) relative vector
    vp_vec = np.array([math.sqrt(mu_sun / r_venus_orb), 0])  # [km/s]
    print(f"velocity planet (venus) vector, vp_vec = {vp_vec} [km/s]")

    # p.465
    v1_infty_vec = v1p_vec - vp_vec
    print(f"velocity,inbound from infinity, v1_infty_vec = {v1_infty_vec} [km/s]")
    v1_infty = np.linalg.norm(v1_infty_vec)
    print(f"velocity inbound, magnitude, v1_infty = {v1_infty:.5g} [km/s]")

    # hyperbola periapsis radius; p.465
    rp_venus = r_venus + alt_venus  # [km]
    # planetcentric angular momentum & eccentricity; eqns 8.39, 8.39; p.465
    h2 = rp_venus * math.sqrt(v1_infty**2 + 2 * mu_venus / rp_venus)
    ecc1_venus = 1 + (rp_venus * v1_infty**2) / mu_venus
    print(f"angular momentum, h2 = {h2:.5g} [km^2/s]")
    print(f"eccentricity, inbound, ecc1_venus = {ecc1_venus:.5g} [km^2/s]")

    # turn angle and true anomaly of asymptote
    delta_turn1 = 2 * math.asin(1 / ecc1_venus)
    nu_asym = math.acos(-1 / ecc1_venus)
    print(f"turn angle inbound, delta_turn1 = {delta_turn1*180/math.pi:.5g} [deg]")
    print(f"true anomaly of asymptote, nu_asym = {nu_asym*180/math.pi:.5g} [deg]")

    # aiming radius; p.465; eqns. 2.50, 2.103, 2.107
    delta_aim = rp_venus * math.sqrt((ecc1_venus + 1) / (ecc1_venus - 1))
    print(f"aiming radius, delta_aim = {delta_aim:.5g} [km]")

    # angle between v1_infty and v_venus; p.465
    phi1 = math.atan(v1_infty_vec[1] / v1_infty_vec[0])
    print(f"true anomaly of asymptote, inbound, phi1 = {phi1*180/math.pi:.5g} [deg]")

    # part a, Dark Side Approach; p.466
    # There are two flyby approaches:
    # (1) Dark side approach, the turn angle is counterclockwise (+102.9◦)
    # (2) Sunlit side approach, the turn anble is clockwise (−102.9◦).

    # angle between v_infty & V_venus_vec, darkside turn; eqn 8.85; p.466
    phi2 = phi1 + delta_turn1
    print(f"darkside turn angle, phi2 = {phi2*180/math.pi:.5g} [deg]")

    # eqn 8.86; p.466
    v2_infty_vec = v1_infty * np.array([math.cos(phi2), math.sin(phi2)])  # [km/s]
    print(f"darkside velocity infinity, v2_infty_vec = {v2_infty_vec} [km/s]")

    # outbound velocity vector, planet, sun direction coordinates; p.466
    v2p_vec = vp_vec + v2_infty_vec  # [km/s]
    print(f"outbound velocity vector, v2p_vec = {v2p_vec} [km/s]")
    v2p_mag = np.linalg.norm(v2p_vec)
    print(f"outbound crossing velocity, magnitude, v2p_mag = {v2p_mag:.5g} [km/s]")
    print(f"compare darkside inbound/outbound speeds: {(v2p_mag-v1p_mag):.5g} [km/s]")

    # part a, Post Flyby Ellipse (orbit 2) for Darkside Approach; p.467
    # The heliocentric post flyby trajectory, orbit 2.
    # Angular momentum orbit 2; eqn 8.90.
    ho2 = r_venus_orb * v2p_vec[0]
    print(f"angular momentum, orbit 2, ho2 = {ho2:.5g} [km/s]")
    ecc_cos = (ho2**2 / (mu_sun * r_venus_orb)) - 1
    ecc_sin = -v2p_vec[1] * ho2 / (mu_sun)
    ecc_tan = ecc_sin / ecc_cos
    print(f"interium, ecc_cos = {ecc_cos:.5g}")
    print(f"interium, ecc_sin = {ecc_sin:.5g}")
    print(f"interium, ecc_tan = {ecc_tan:.5g}")
    theta2 = math.atan(ecc_tan)
    print(f"theta2, 1st possibility = {theta2*180/math.pi:.5g} [deg]")
    print(f"theta2, 2nd possibility = {(theta2*180/math.pi)+180:.5g} [deg]")
    # based on cos and sin quadrants select angle
    if (ecc_cos < 0 and ecc_sin < 0) or (ecc_cos > 0 and ecc_sin < 0):
        # quadrant 3 or quadrant 4
        theta2 = theta2 + math.pi
        print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")
    else:
        print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")

    print(f"perihelion of departure, theta2 = {theta2*180/math.pi:.5g} [deg]")
    ecc2_venus = ecc_cos / math.cos(theta2)
    print(f"eccentricity, orbit 2, ecc2_venus = {ecc2_venus:.5g}")

    r2_perihelion = (ho2**2 / mu_sun) * (1 / (1 + ecc2_venus))
    print(f"radius orbit2, perihelion, r2_perihelion = {r2_perihelion:.5g}")

    # part b, Sunlit side approach; p.467+
    print(
        "\n********** sunlit approach **********"
    )  # make line seperation in print list
    # angle lightside, v_infty & V_venus_vec, outbound crossing; p.467
    phi2 = phi1 - delta_turn1
    print(f"lightside turn angle, phi2 = {phi2*180/math.pi:.5g} [deg]")

    # velocity 2 lightside vector; p.468
    v2l_infty_vec = v1_infty * np.array([math.cos(phi2), math.sin(phi2)])  # [km/s]
    print(f"lightside velocity infinity, v2l_infty_vec = {v2l_infty_vec} [km/s]")

    # velocity outbound lightside vector, planet, sun direction coordinates; p.468
    v2pl_vec = vp_vec + v2l_infty_vec  # [km/s]
    print(f"outbound velocity vector lightside, v2pl_vec = {v2pl_vec} [km/s]")
    v2pl_mag = np.linalg.norm(v2pl_vec)
    print(
        f"outbound crossing velocity lightside, magnitude, v2pl_mag = {v2pl_mag:.5g} [km/s]"
    )
    print(f"compare lightside inbound/outbound speeds: {(v2pl_mag-v1p_mag):.5g} [km/s]")

    print("********** post flyby ellipse **********")
    # Angular momentum lightside orbit 2; eqn 8.90.
    h_lo2 = r_venus_orb * v2pl_vec[0]
    print(f"angular momentum, lightside orbit 2, ho2 = {h_lo2:.5g} [km/s]")
    ecc_cos = (h_lo2**2 / (mu_sun * r_venus_orb)) - 1
    ecc_sin = -v2pl_vec[1] * h_lo2 / mu_sun
    ecc_tan = ecc_sin / ecc_cos
    print(f"interium, ecc_cos = {ecc_cos:.5g}")
    print(f"interium, ecc_sin = {ecc_sin:.5g}")
    print(f"interium, ecc_tan = {ecc_tan:.5g}")
    theta2 = math.atan(ecc_tan)
    print(f"theta2, 1st possibility = {theta2*180/math.pi:.5g} [deg]")
    print(f"theta2, 2nd possibility = {(theta2*180/math.pi)+180:.5g} [deg]")
    # based on cos and sin quadrants select angle
    if (ecc_cos < 0 and ecc_sin < 0) or (ecc_cos > 0 and ecc_sin < 0):
        # quadrant 3 or quadrant 4
        theta2 = theta2 + math.pi
        print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")
    else:
        print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")

    print(f"departure perihelion, lightside, theta2 = {theta2*180/math.pi:.5g} [deg]")
    ecc2_venus = ecc_cos / math.cos(theta2)
    print(f"eccentricity, orbit 2, ecc2_venus = {ecc2_venus:.5g}")

    r2_perihelion = (h_lo2**2 / mu_sun) * (1 / (1 + ecc2_venus))
    print(f"radius orbit2, perihelion lightside, r2_perihelion = {r2_perihelion:.5g}")

    return None  # curtis_ex8_6()


def test_curtis_ex8_4():
    print(f"\nTest Curtis example 8.4, ... :")
    # function does not need input parameters.
    curtis_ex8_4()
    return None


def test_curtis_ex8_5():
    print(f"\nTest Curtis example 8.5, ... :")
    # function does not need input parameters.
    curtis_ex8_5()
    return None


def test_curtis_ex8_6():
    print(f"\nTest Curtis example 8.6, ... :")
    # function does not need input parameters.
    curtis_ex8_6()
    return None


# use the following to test/examine functions
if __name__ == "__main__":

    # test_curtis_ex8_4()  # test curtis example 8.4
    # test_curtis_ex8_5()  # test curtis example 8.5
    test_curtis_ex8_6()  # test curtis example 8.6; Venus fly-by
