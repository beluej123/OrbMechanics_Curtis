"""
Curtis chapter 8, examples collection; Interplanetary Trajectories.

Notes:
----------
    The Vallado [2], [4] interplanetary discussion is a useful addition to
        Curtis chapter 8.  Vallado [2] section 12.2, pp.944, patched conic
        trajectories, pp.948, interplanetary trajectories...
    This file is organized with each example as a function; example function name:
        def curtis_ex8_1():
    
    Supporting functions for the test functions below, may be found in other
        files, for example ..., etc. Also note, the test examples are
        collected right after this document block.  However, the example test
        functions are defined/enabled at the end of this file.  Each example
        function is designed to be stand-alone, but, if you use the function
        as stand alone you will need to copy the imports...

    Reminder to me; cannot get black formatter to work within VSCode,
        so in terminal type; black *.py.
    Reminder to me; VSCode DocString, Keyboard shortcut: ctrl+shift+2.
    
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally angles are saved in [rad], distance [km].
    
References:
----------
    See references.py for references list.
"""

import math

import numpy as np  # for vector math

import astro_data
import Braeunig
import Braeunig.braeunigFunctions
import functionCollection as funColl  # includes planetary tables
import Stumpff_1
from Algorithm8_x import rv_from_date
from astro_time import g_date2jd, julian_date


def curtis_ex8_3_soi():
    """
    Find soi (sphere of influence).
        Curtis [3] p.441, example 8.3.
    Given:
        m1  : mass of smaller body; i.e. planet
        m2  : mass of smaller body; i.e. sun
        R   : distance between mass's; for earth->sun this is semi-major axis
    Find:
        sphere of influence
    Notes:
    ----------
        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    # constants; mostly from Vallado [2 or 4] not Curtis
    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mass_sun = 1.989e30  # [kg] Curtis [3] p.689, appendix, table A.1
    mass_earth = 5.974e24  # [kg] Curtis [3] p.689, appendix, table A.1

    soi = funColl.sphere_of_influence(mass1=mass_earth, mass2=mass_sun, R=au)
    print(f"soi earth-sun, {soi:.6g}")

    return


def curtis_ex8_4_depart():
    """
    Earth->Mars, depart Earth.  Curtis [3] pp.446, example 8.4.
    Given:
        Earth orbit launch, from alt=300 [km] circular, hyperbolic launch trajectory;
            thus ecc=1, and Earth GM (or mu)
        r1: periapsis altitude 500 [km];
        r2: earth-sun SOI (sphere of influence)

    Find:
        (a) delta-v required
        (b) departure hyperbola perigee location
        (c) propellant as a percentage of the spacecraft, before delta-v burn
            assume Isp (specific impulse) = 300 [s]
    Notes:
    ----------
        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
        Solar system parameters/constants; dataclass's organized; orbit & body.
    """
    # constants from astro_data.py; mostly from Vallado [4] not Curtis [3]
    mu_sun = astro_data.sun_prms.mu  # [km^3/s^2]
    # earth body & orbit constants
    mu_earth = astro_data.earth_b_prms.mu  # [km^3/s^2]
    r_earth = astro_data.earth_b_prms.eq_radius_km  # [km]
    r_earth_orb = astro_data.earth_o_prms.sma  # [km] ref in astro_data.py
    alt_earth = 300  # [km], given altitude above earth
    # mars body & orbit constants
    r_mars_orb = astro_data.mars_o_prms.sma  # [km]

    # part a
    # Curtis [3] p.442, eqn 8.35
    v_inf = math.sqrt(mu_sun / r_earth_orb) * (
        math.sqrt(2 * r_mars_orb / (r_earth_orb + r_mars_orb)) - 1
    )
    print(f"depart v_infinity, v_inf = {v_inf:.6g} [km/s]")

    # spacecraft speed in 300km circular parking orbit; Curtis p.444, eqn 8.41
    # departure from circular parking orbit
    v_c = math.sqrt(mu_earth / (r_earth + alt_earth))
    print(f"departure parking orbit, v_c= {v_c:.6g} [km/s]")

    # Delta_v required to enter departure hyperbola; eqn 8.42, p444
    delta_v = v_c * (math.sqrt(2 + (v_inf / v_c) ** 2) - 1)
    print(f"delta_v to enter departure hyperbola = {delta_v:.6g} [km/s]")

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


def curtis_ex8_5_arrive():
    """
    Earth->Mars, arrival at Mars.  Curtis pp.456, example 8.5.
    After a Hohmann transfer from earth to mars, calculate:
        (a) the minimum delta_v required to place spacecraft in an orbit with 7hour period
        (b) periapsis radius, the aiming radius and the angle between periapse and Mars velocity vector.
        (c) aiming radius
        (d) angle between periapsis and Mars' velocity vector

    Given:
        depart; mu_earth, r_orb_earth
        arrive; mu_mars, r_orb_mars, r_p_mars
    Find:
        TODO fix this later
    Notes:
    ----------
        May help development; see https://github.com/jkloser/OrbitalMechanics
        Helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    # constants from astro_data.py; mostly from Vallado [4] not Curtis [3]
    # earth body & orbit constants
    r_earth_orb = astro_data.earth_o_prms.sma  # [km]
    # mars body & orbit constants
    r_mars_orb = astro_data.mars_o_prms.sma  # [km]
    mu_mars_km = astro_data.mars_b_prms.mu  # [km^3/s^2]
    # sun constant
    mu_sun_km = astro_data.sun_prms.mu  # [km^3/s^2]
    # problem given data
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

    # from eqn 8.67
    ecc_mars_orb = (2 * mu_mars_km / (a_capture * v_inf**2)) - 1
    print(f"eccentricity, at mars = {ecc_mars_orb:.5g}")

    # from eqn 8.70
    delta_v = v_inf * math.sqrt((1 - ecc_mars_orb) / 2)
    print(f"delta_v enter mars = {delta_v:.5g} [km/s]")

    # part b
    # periapsis radius at mars capture, from eqn 8.67
    r_p = (2 * mu_mars_km / v_inf**2) * ((1 - ecc_mars_orb) / (1 + ecc_mars_orb))
    print(f"periapsis at mars, r_p = {r_p:.5g} [km]")

    # part c
    # aiming radius from eqn 8.71
    aim_radius = r_p * math.sqrt(2 / (1 - ecc_mars_orb))
    print(f"aiming radius (aka delta) at mars = {aim_radius:.5g} [km]")

    # part d
    # angle to periapsis from eqn 8.43
    beta_p = math.acos(1 / (1 + r_p * v_inf**2 / mu_mars_km))
    print(f"angle to periapsis at mars = {(beta_p*180/math.pi):.5g} [km]")
    return None  # curtis_ex8_5()


def curtis_ex8_6_flyby():
    """
    Earth->Venus fly-by.  Curtis [3] pp.462, example 8.6.
        Spacecraft departs earth with a velocity perpendicular to the sun line.
        Encounter occurs at a true anomaly in the approach trajectory of 30[deg].
        Periapse altitude 300 km.
    (a) Dark side Venus apporach.
            Post fly-by orbit shown in Figure 8.20.
    (b) Sunlit side Venus approach.
            Post fly-by orbit shown in Figure 8.21.

    Leading-side flyby results in a decrease in the spacecraft's heliocentric speed.
    Trailing-side flyby increases helliocentric speed;
        e1, h1, and θ1 are eccentricity, angular momentum,
        and true anomaly of heliocentric approach trajectory.

    TODO update variable names to be consistant with darkside & lightside calculations
    Input Parameters:
    ----------
        TODO update
    Return:
    ----------
        TODO update
    Notes:
    ----------
        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    # get constants from astro_data.py; mostly from Vallado [4] not Curtis [3]
    mu_venus_km = 3.257e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5

    r_earth_orb = 149598023  # [km], Vallado [2] p.1041, tbl.D-3, sma
    r_venus_orb = 108208601  # [km], Vallado [2] p.1041, tbl.D-3, sma

    r_venus = 6052.0  # [km], Vallado [2] p.1041, tbl.D-3

    alt_venus = 300  # altitude above venus [km]
    nu_venus = -30 * math.pi / 180  # venus approach true anomaly (nu); saved as [rad]

    np.set_printoptions(precision=4)  # numpy, set vector printing size
    # part a, Pre-Flyby ellipse; p.462+
    # orbit id (1), transfer orbit eccentricity; p.464
    ecc1_venus_orb = (r_earth_orb - r_venus_orb) / (
        r_earth_orb + r_venus_orb * math.cos(nu_venus)
    )
    print(f"eccentricity, at venus, ecc1_venus_orb = {ecc1_venus_orb:.5g}")

    # orbit 1 angular momentum; p.464
    h1 = math.sqrt(mu_sun_km * r_earth_orb * (1 - ecc1_venus_orb))
    print(f"angular momentum, orbit1, h1 = {h1:.5g} [km^2/s]")

    # Calculate spacecraft radial and transverse components heliocentric velocity at
    # the inbound crossing of Venus’s sphere of influence.
    v1_perp = h1 / r_venus_orb  # perpendicular velocity orbit 1[km/s]
    v1_radi = (
        (mu_sun_km / h1) * (ecc1_venus_orb) * math.sin(nu_venus)
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
    vp_vec = np.array([math.sqrt(mu_sun_km / r_venus_orb), 0])  # [km/s]
    print(f"velocity planet (venus) vector, vp_vec = {vp_vec} [km/s]")

    # p.465
    v1_infty_vec = v1p_vec - vp_vec
    print(f"velocity,inbound from infinity, v1_infty_vec = {v1_infty_vec} [km/s]")
    v1_infty = np.linalg.norm(v1_infty_vec)
    print(f"velocity inbound, magnitude, v1_infty = {v1_infty:.5g} [km/s]")

    # hyperbola periapsis radius; p.465
    rp_venus = r_venus + alt_venus  # [km]
    # planetcentric angular momentum & eccentricity; eqns 8.39, 8.39; p.465
    h2 = rp_venus * math.sqrt(v1_infty**2 + 2 * mu_venus_km / rp_venus)
    ecc1_venus = 1 + (rp_venus * v1_infty**2) / mu_venus_km
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
    ecc_cos = (ho2**2 / (mu_sun_km * r_venus_orb)) - 1
    ecc_sin = -v2p_vec[1] * ho2 / (mu_sun_km)
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

    r2_perihelion = (ho2**2 / mu_sun_km) * (1 / (1 + ecc2_venus))
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
    ecc_cos = (h_lo2**2 / (mu_sun_km * r_venus_orb)) - 1
    ecc_sin = -v2pl_vec[1] * h_lo2 / mu_sun_km
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

    r2_perihelion = (h_lo2**2 / mu_sun_km) * (1 / (1 + ecc2_venus))
    print(f"radius orbit2, perihelion lightside, r2_perihelion = {r2_perihelion:.5g}")

    return None  # curtis_ex8_6()


def curtis_ex8_7_earth_mars():
    """
    Planetary Ephemeris; distance Earth->Mars.
    Curtis [3] section 8.10, pp.470; example 8.7, pp.473.

    Given:
        t0, 2003-03-27 12:00 UT
        Earth position, Mars position
    Find:
        From date/time find r1_vec(Mars)-r0_vec(Earth)
    Notes:
    ----------
        Uses Curtis [3] pp.471, algorithm 8.1; Julian day p.277, example 5.4.
        Note curtis_ex4_7().
        Also see Vallado [2] functions: pp. 296, planetRV() (algotithm 33),
            cov2rv() (algorithm 11), et.al
        Orbital elements tables kept in functionCollection.py
        For my code, generally angles are saved in [rad].

        Orbital elements in this function:
            sma   = [km] semi-major axis (aka a)
            ecc   = [-] eccentricity
            incl  = [deg] inclination angle; to the ecliptic
            RAAN  = [deg] right ascension of ascending node (aka capital W)
            w_bar = [deg] longitude of periapsis (NOT arguement of periapsis, w)
                    Note, w_bar = w + RAAN
            L     = [deg] mean longitude (NOT mean anomaly, M)
                    Note, L = w_bar + M

        Helpful for interplanetary flight http://www.braeunig.us/space/interpl.htm
        References: see list at file beginning.
    """
    np.set_printoptions(precision=6)  # numpy, set vector printing size
    deg2rad = math.pi / 180  # saves calculations
    rad2deg = 1 / deg2rad  # saves calculations

    # get constants from astro_data.py; mostly from Vallado [4] not Curtis [3]
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5

    # given date/time for t0, find Julian date
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT
    date_UT = [2003, 8, 27, 12, 0, 0]  # [UT] date/time python list
    yr, mo, d, hr, minute, sec = date_UT

    # ********** Earth part **********
    # Earth: steps 1, 2, 3, of Curtis p.471-472, part of algorithm 8.1.
    planet_id = 3  # earth
    coe_t0, jd_t0 = funColl.coe_from_date(planet_id, date_UT)
    # coe element names: ["sma[km]", "ecc", "incl[rad]", "RAAN[rad]", "w_hat[rad]", "L_[rad]"]
    sma, ecc, incl_rad, RAAN_rad, w_hat_rad, L_rad = coe_t0
    incl_deg = incl_rad * rad2deg
    RAAN_deg = RAAN_rad * rad2deg
    w_hat_deg = w_hat_rad * rad2deg
    L_deg = L_rad * rad2deg

    print(f"t0, given date/time, {yr}-{mo}-{d} {hr}:{minute}:{sec:.4g} UT")
    print(f"Julian date, jd_t0= {jd_t0}")
    print(f"\n** Earth Calculations: **")
    print(
        f"sma earth, {sma:.8g} [km]; "
        f"ecc earth, {ecc:.8g}; "
        f"\nincl earth, {incl_deg:.8g} [deg]; "
        f"RAAN earth, {RAAN_deg:.6g} [deg]; "
        f"w_hat earth, {w_hat_deg:.6g} [deg]; "
        f"L_ earth, {L_deg:.6g} [deg]"
    )
    # Earth: Curtis, p.473, step 4
    h_mag = math.sqrt(mu_sun_km * sma * (1 - ecc**2))

    # Earth: Curtis, p.473, step 5
    w_deg = (w_hat_deg - RAAN_deg) % (360)  # [rad] limit value 0->2*pi
    M_deg = (L_deg - w_hat_deg) % (360)  # [rad] limit value 0->2*pi
    w_rad = w_deg * deg2rad
    M_rad = M_deg * deg2rad
    print(f"Earth: w_deg= {w_deg:.8g} [deg], M_deg= {M_deg:.8g} [deg]")

    # Earth: Curtis, p.474, step 6; find eccentric angle/anomaly
    E_rad = funColl.solve_for_E(Me=M_rad, ecc=ecc)  # [rad]
    E_deg = E_rad * rad2deg
    print(f"Earth: E_deg= {E_deg:.8g}")

    # Earth: Curtis, p.474, step 7; find true angle/anomaly
    TA_rad = 2 * math.atan(
        math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E_rad / 2)
    )  # [rad]
    TA1_rad = TA_rad % (2 * math.pi)  # [rad] limit value 0->2*pi
    TA1_deg = TA1_rad * rad2deg
    print(f"Earth: TA1_deg= {TA1_deg:.8g}")

    # Earth: Curtis, p.474, step 8; find r_vec, v_vec
    #   note Curtis, pp.232, example 4.7 & p.231, algorithm 4.5
    r_vec_earth, v_vec_earth = funColl.sv_from_coe(
        h=h_mag,
        ecc=ecc,
        RA_rad=RAAN_rad,
        incl_rad=incl_rad,
        w_rad=w_rad,
        TA_rad=TA1_rad,
        mu=mu_sun_km,
    )

    # ********** Mars part **********
    # Mars: steps 1, 2, 3, of Curtis p.471-472, part of algorithm 8.1.
    planet_id = 4  # mars
    coe_t0, jd_t0 = funColl.coe_from_date(planet_id, date_UT)
    # coe element names: ["sma[km]", "ecc", "incl[rad]", "RAAN[rad]", "w_hat[rad]", "L_[rad]"]
    sma, ecc, incl_rad, RAAN_rad, w_hat_rad, L_rad = coe_t0
    incl_deg = incl_rad * rad2deg
    RAAN_deg = RAAN_rad * rad2deg
    w_hat_deg = w_hat_rad * rad2deg
    L_deg = L_rad * rad2deg
    print(f"\n** Mars Calculations: **")
    print(
        f"sma mars, {sma:.8g} [km]; "
        f"ecc mars, {ecc:.8g}; "
        f"\nincl mars, {incl_deg:.8g} [deg]; "
        f"RAAN mars, {RAAN_deg:.6g} [deg]; "
        f"w_hat mars, {w_hat_deg:.6g} [deg]; "
        f"L_ mars, {L_deg:.6g} [deg]"
    )
    # Mars: Curtis, p.473, step 4
    h_mag = math.sqrt(mu_sun_km * sma * (1 - ecc**2))

    # Mars: Curtis, p.473, step 5
    w_deg = (w_hat_deg - RAAN_deg) % (360)  # [rad] limit value 0->2*pi
    M_deg = (L_deg - w_hat_deg) % (360)  # [rad] limit value 0->2*pi
    w_rad = w_deg * deg2rad
    M_rad = M_deg * deg2rad
    print(f"Mars: w_deg= {w_deg:.8g} [deg], M_deg= {M_deg:.8g} [deg]")

    # Mars: Curtis, p.474, step 6; find eccentric angle/anomaly
    E_rad = funColl.solve_for_E(Me=M_rad, ecc=ecc)  # [rad]
    E_deg = E_rad * rad2deg
    print(f"Mars: E_deg= {E_deg:.8g}")

    # Mars: Curtis, p.474, step 7; find true angle/anomaly
    TA_rad = 2 * math.atan(
        math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E_rad / 2)
    )  # [rad]
    TA1_rad = TA_rad % (2 * math.pi)  # [rad] limit value 0->2*pi
    TA1_deg = TA1_rad * rad2deg
    print(f"Mars: TA1_deg= {TA1_deg:.8g}")

    # Mars: Curtis, p.474, step 8; find r_vec, v_vec
    #   note Curtis, pp.232, example 4.7 & p.231, algorithm 4.5
    r_vec_mars, v_vec_mars = funColl.sv_from_coe(
        h=h_mag,
        ecc=ecc,
        RA_rad=RAAN_rad,
        incl_rad=incl_rad,
        w_rad=w_rad,
        TA_rad=TA1_rad,
        mu=mu_sun_km,
    )
    print(f"\nr_vec_earth= {r_vec_earth} [km]")
    print(f"v_vec_earth= {v_vec_earth} [km/s]")
    print(f"r_vec_mars= {r_vec_mars} [km]")
    print(f"v_vec_mars= {v_vec_mars} [km/s]")

    # ********** Earth->Mars Distance **********
    dist_earth_mars = np.linalg.norm(r_vec_mars - r_vec_earth)
    print(f"\nDistance Earth->Mars, dist_earth_mars= {dist_earth_mars:.8g} [km]")

    # ********** above finished with the book example ***************
    # test rf_from_date() function; should be the same as calculated above
    planet_id = 3
    r_vec_earth, v_vec_earth, coe_earth, jd_t0 = rv_from_date(
        planet_id=planet_id, date_UT=date_UT, mu=mu_sun_km
    )
    planet_id = 4
    r_vec_mars, v_vec_mars, coe_mars, jd_t0 = rv_from_date(
        planet_id=planet_id, date_UT=date_UT, mu=mu_sun_km
    )
    # coe element names= ["sma[km]", "ecc", "incl[rad]", "RAAN[rad]", "w_hat[rad]", "L_[rad]"]
    print(f"coe id's:      sma,        ecc,        incl,       RAAN,     w_hat,    L_")
    print(f"coe_earth", [f"{num:.6g}" for num in coe_earth])
    print(f"coe_mars", [f"{num:.6g}" for num in coe_mars])
    print(f"\nr_vec_earth= {r_vec_earth} [km]")
    print(f"v_vec_earth= {v_vec_earth} [km/s]")
    print(f"r_vec_mars= {r_vec_mars} [km]")
    print(f"v_vec_mars= {v_vec_mars} [km/s]")

    return None  # curtis_ex8_7()


def curtis_ex8_7_astropy():
    """
    Use astropy to corroborate ephemeris in Curtis [3] example 8-7.
    Corroborate ephemeris with JPL Horizons
        https://ssd.jpl.nasa.gov/horizons/app.html#/
    Notes:
    ----------
        Very good match with JPL horizons.
    """
    from astropy import units as u
    from astropy.coordinates import (
        get_body_barycentric,
        get_body_barycentric_posvel,
        solar_system_ephemeris,
    )
    from astropy.time import Time

    print(f"(\nAssociated with Curtis example 8.7, planetary ephemeris:")
    au = 149597870.7  # [km/au] Vallado [2] p.1042, tbl.D-5

    np.set_printoptions(precision=8)  # numpy, set vector printing size
    # tdb runs at uniform rate of one SI second per second; independent of Earth rotation irregularities.
    ts0 = Time("2003-08-27 12:0", scale="tdb")
    print(f"date ts0 = {ts0}, Julian date: {ts0.jd}")

    # JPL ephemeris files: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    #   List of files, https://ssd.jpl.nasa.gov/planets/eph_export.html
    #   DE421 valid dates 1899-2053, small file, 17MB, circa 2008
    #   DE422 valid dates -3000 to 3000, medium file, 623MB, circa 2008
    #   DE440 valid dates 1549 to 2650,  file 10.6MB, circa 2020
    #   DE440s valid dates 1849 to 2150, smallish file 32MB, circa 2020
    #   DE441 valid dates -13200 to 17191, huge file 3.1GB, circa 2020

    # astropy gets locally saved file from local user cached file
    with solar_system_ephemeris.set("de440s"):  # times between years 1550 to 2650
        # earthBc = get_body_barycentric("earth", ts1, ephemeris='builtin')
        earthBc = get_body_barycentric("earth", ts0)  # equatorial (not ecliptic)
        marsBc = get_body_barycentric("mars", ts0)
        # astropy provides equatorial (not ecliptic)
        earthBc_pv = get_body_barycentric_posvel("earth", ts0)  # position & velocity
        marsBc_pv = get_body_barycentric_posvel("mars", ts0)

    # np.set_printoptions(formatter={"float": "{: 0.7f}".format})
    print(f"\nearth(ts0), astropy equatorial, {earthBc_pv[0].xyz.to(u.km)}")  # [km]
    print(f"earth(ts0), astropy equatorial, {earthBc_pv[0].xyz.to(u.au)}")  # [au]
    print(f"earth(ts0), astropy equatorial, {earthBc_pv[1].xyz.to(u.km / u.s)}")
    print()
    print(f"mars(ts0), astropy equatorial, {marsBc_pv[0].xyz.to(u.km)}")
    print(f"mars(ts0), astropy equatorial, {marsBc_pv[0].xyz.to(u.au)}")
    print(f"mars(ts0), astropy equatorial, {marsBc_pv[1].xyz.to(u.km / u.s)}")

    return None  # curtis_ex8_7_astropy


def curtis_ex8_8():
    """
    3D Earth->Mars, planetary transfer parameters calculations.
    Curtis [3] section 8.10, pp.470; example 8.8, pp.476.  Used to develop
        algorithm 8.2.

    Given:
        t0, 1996-11-07 0:0 UT, depart Earth
        t1, 1997-09-12 0:0 UT, arrive Mars
        Earth position, Mars position
    Find:

    Notes:
    ----------
        Uses Curtis [3] pp.471, algorithm 8.1.  Note Curtis p.277, example 5.4, Sideral time.
        Also see Vallado [2] functions: pp. 296, planetRV() (algotithm 33),
            cov2rv() (algorithm 11), et.al
        Orbital elements tables kept in functionCollection.py

        Orbital elements identifiers:
            sma   = semi-major axis (aka a) [km]
            ecc   = eccentricity
            incl  = inclination angle; to the ecliptic [deg]
            RAAN  = right ascension of ascending node (aka capital W) [deg]
                    longitude node
            w_bar = longitude of periapsis [deg]
            L     = mean longitude [deg]

        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    References:
    ----------
        See references.py for references list.
    """
    deg2rad = 180 / math.pi  # save multiple calculations of same value
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5

    # mu_mars_km = 4.305e4  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    au_km = 149597870  # [km/au], Vallado [4] p.1057, tbl.d-3
    mass_sun_kg = 1.9891e30  # [kg], Vallado [4] p.1059, tbl.d-5
    mass_earth_kg = 5.9742e24  # [kg], Vallado [4] p.1057, tbl.d-3
    mass_mars_kg = 6.4191e23  # [kg], Vallado [4] p.1057, tbl.d-3

    # step 1
    # given date/time for t0, find Julian date
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT
    t0_date_UT = [1996, 11, 7, 0, 0, 0]  # [UT] date/time python list
    planet0 = 3  # earth Id
    r0_vec_earth, v0_vec_earth, coe_earth, jd_t0 = rv_from_date(
        planet_id=planet0, date_UT=t0_date_UT, mu=mu_sun_km
    )

    t1_date_UT = [1997, 9, 12, 0, 0, 0]  # [UT] date/time python list
    planet1 = 4  # mars
    r0_vec_mars, v0_vec_mars, coe_mars, jd_t1 = rv_from_date(
        planet_id=planet1, date_UT=t1_date_UT, mu=mu_sun_km
    )
    yr, mo, d, hr, minute, sec = t0_date_UT
    print(f"t0, departure date/time, {yr}-{mo}-{d} {hr}:{minute}:{sec:.4g} UT")
    print(f"    Departure Julian date, jd_t0= {jd_t0:.8g}")

    yr, mo, d, hr, minute, sec = t1_date_UT
    print(f"t1, arrival date/time, {yr}-{mo}-{d} {hr}:{minute}:{sec:.4g} UT")
    print(f"    Arrival Julian date, jd_t1= {jd_t1:.8g}")

    print(f"r0_vec_earth(t0)= {r0_vec_earth}")  # depart
    print(f"v0_vec_earth(t0)= {v0_vec_earth}")
    print(f"\nr0_vec_mars(t1)= {r0_vec_mars}")  # arrive
    print(f"v0_vec_mars(t1)= {v0_vec_mars}")

    # step 2: assign vector's for SOI (sphere of influence); earth and mars
    #   Note, Curtis [3] assigns SOI vectors as sun->planet position, which
    #       will be off by the SOI; but will be close due to distances involved.
    #   TODO calculate vectors for SOI
    r0_mag_earth = np.linalg.norm(r0_vec_earth)
    r0_mag_mars = np.linalg.norm(r0_vec_mars)
    SOI_earth = funColl.sphere_of_influence(
        R=au_km, mass1=mass_earth_kg, mass2=mass_sun_kg
    )
    SOI_mars = funColl.sphere_of_influence(
        R=au_km, mass1=mass_mars_kg, mass2=mass_sun_kg
    )
    # find SOI_vec_earth; remember r0_vec is sun->earth vector
    r1_vec_earth = r0_vec_earth - (r0_vec_earth / r0_mag_earth) * SOI_earth
    r1_vec_mars = r0_vec_mars - (r0_vec_mars / r0_mag_mars) * SOI_mars
    print(f"***** Below, SOI calculations not in Curtis [3] example8-8 *****")
    print(f"SOI mag, earth, {SOI_earth:.8g} [km]")
    print(f"SOI vec, earth, {r1_vec_earth} [km]")  # for a better approximation...
    print(f"SOI mag, mars, {SOI_mars:.8g} [km]")
    print(f"SOI vec, mars, {r1_vec_mars} [km]")  # for a better approximation...
    print(f"***** Above, SOI calculations not in Curtis [3] example8-8 *****")

    # use the Curtis [3] assignments; to verify calcuations
    #   comment out next 2 commands to for "more accurate" calculations...
    r1_vec_earth = r0_vec_earth  # soi earth depart, Curtis [3]; sun relative
    r1_vec_mars = r0_vec_mars  # soi mars arrival, Curtis [3]; sun relative

    tof_jd = jd_t1 - jd_t0  # [julian days]
    print(f"\nTime-of-flight, tof_1= {tof_jd:.8g} [days]")

    tof = tof_jd * 24 * 3600  # days->sec
    v1_vec_D, v2_vec_A = funColl.Lambert_v1v2_solver(
        r1_v=r1_vec_earth, r2_v=r1_vec_mars, dt=tof, mu=mu_sun_km
    )
    print(f"v1_vec_D= {v1_vec_D} [km/s]")  # depart Earth
    print(f"v2_vec_A= {v2_vec_A} [km/s]")  # arrive Mars

    # Vallado [2] position/velocity->coe; function includes all Kepler types
    # elements = np.array([sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_])
    # return o_type, elements

    o_type, elements = funColl.val_rv2coe(
        r_vec=r1_vec_earth, v_vec=v1_vec_D, mu=mu_sun_km
    )
    funColl.print_coe(o_type=o_type, elements=elements)  # replaces the following

    print(f"\nEarth departure hyperbolic excess velocity:")
    v_vec_inf_D = v1_vec_D - v0_vec_earth
    v_mag_inf_D = np.linalg.norm(v_vec_inf_D)

    # calculate parabolic earth escape; not part of Curtis example8-8
    #   find time to soi (sphere of influence)
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    r_earth = 6378.1363  # [km] Vallado [4] p.1057, tbl.d-3; earth radius
    r0_alt = 180  # [km] parking orbit altitude, at t0; relative to earth
    v_esc_D = math.sqrt(2 * mu_earth_km / (r_earth + r0_alt))

    print(f"v_vec_inf_D= {v_vec_inf_D} [km/s]")
    print(f"v_mag_inf_D= {v_mag_inf_D:.6g} [km/s]")
    print(f"v_esc_D= {v_esc_D:.6g} [km/s]")

    print(f"\nMars arrival hyperbolic excess velocity:")
    v_vec_inf_A = v2_vec_A - v0_vec_mars
    v_mag_inf_A = np.linalg.norm(v_vec_inf_A)
    print(f"v_vec_inf_A= {v_vec_inf_A} [km/s]")
    print(f"v_mag_inf_A= {v_mag_inf_A:.6g} [km/s]")

    # ********** review Vallado's Lambert from lamberthub **********
    # 2024-09-x I still have not figured out importing sub-directory modules/functions.
    # from Braeunig import vallado_1

    # v1_vec, v2_vec, tof_new, numiter, tpi = vallado_1.vallado2013(
    #     mu=mu_sun_km,
    #     r1=r1_vec_earth,
    #     r2=r1_vec_mars,
    #     tof=tof,
    #     M=0,
    #     prograde=True,
    #     low_path=True,
    #     maxiter=100,
    #     atol=1e-5,
    #     rtol=1e-7,
    #     full_output=True,
    # )
    # # v1, v2, numiter, tpi if full_output is True else only v1, v2.
    # v1_mag, v2_mag = [np.linalg.norm(v) for v in [v1_vec, v2_vec]]
    # np.set_printoptions(precision=5)  # numpy has spectial print provisions
    # print(f"v1_vec= {v1_vec} [km/s]")  # note conversion au->km
    # print(f"v2_vec= {v2_vec} [km/s]")  # note conversion au->km
    # print(f"# of iterations {numiter}, time per iteration, tpi= {tpi:.6g} [s]")

    return None


def curtis_ex8_9_10():
    """
    3D Earth->Mars, delta-v for planetary transfer.
        Curtis [3] section 8.10, pp.470; example 8.9, pp.479;
        & example 8.10, pp.480.
    Some input from Curtis [3] example 8.8.

    Given, time from ex8_8:
        t0, 1996-11-07 0:0 UT, depart Earth from 180[km] orbit
        t1, 1997-09-12 0:0 UT, arrive Mars with 48 [hr] peroid orbit
    Find:
        Delta-v for planetary transfer.
    Notes:
    ----------

        Helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    References:
    ----------
        See references.py for references list.
    """
    deg2rad = 180 / math.pi  # save multiple calculations of same value
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3

    au_km = 149597870  # [km/au], Vallado [4] p.1057, tbl.d-3
    mass_sun_kg = 1.9891e30  # [kg], Vallado [4] p.1059, tbl.d-5
    mass_earth_kg = 5.9742e24  # [kg], Vallado [4] p.1057, tbl.d-3
    mass_mars_kg = 6.4191e23  # [kg], Vallado [4] p.1057, tbl.d-3
    r_earth = 6378.1363  # [km] Vallado [4] p.1057, tbl.d-3; earth radius
    r0_alt = 180  # [km] parking orbit altitude, at t0; relative to earth
    rp0_earth = r_earth + r0_alt  # parking orbit altitude, at t0; relative to earth

    # from ex8-8
    v_mag_inf_D = 3.16435  # [km/s] earth departure relative to sun

    # ********** Below, Earth depart, Curtis [3] ex.8-9 ********
    # find velocity assume periapsis departure hyperbola
    vp0_earth = math.sqrt((v_mag_inf_D**2) + (2 * mu_earth_km) / rp0_earth)

    # calculate earth relative orbit; assume vehicle/satellite is in circular orbit
    v_v_cir_earth = math.sqrt(mu_earth_km / rp0_earth)
    print(f"periapsis, vp0_earth= {vp0_earth:.6g} [km/s]")
    print(f"circular orbit, v_v_cir_earth= {v_v_cir_earth:.6g} [km/s]")
    # departure delta-v
    delta_v_depart = vp0_earth - v_v_cir_earth
    print(f"delta_v_depart= {delta_v_depart:.6g} [km/s]")

    # eccentricity
    ecc_depart = 1 + ((rp0_earth * v_mag_inf_D**2) / mu_earth_km)
    print(f"ecc_depart= {ecc_depart:.6g}")

    # ********** Below, Mars arrival, Curtis [3] ex.8-10 *******
    print(f"\n*** Arrive at Mars, example8-10: ***")
    mu_mars_km = 4.305e4  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    mu_mars_km = 42830  # [km^3/s^2], Curtis [3]
    r_mars = 3397.2  # [km] Vallado [4] p.1057, tbl.d-3; earth radius
    r1_alt = 300  # [km] parking orbit altitude, at t0; relative to mars
    rp1_mars = r_mars + r1_alt  # parking orbit altitude, at t0; relative to earth
    T_ellipse_mars = 48 * 3600  # [s] mars elliptical period; convert hours->sec

    # from ex8-8
    v_mag_inf_A = 2.88517  # [km/s] mars arrival relative to sun

    # approach hyperbola eccentricity; relative to mars
    ecc_Arrival = 1 + ((rp1_mars * v_mag_inf_A**2) / mu_mars_km)
    print(f"approach hyperbola eccentricity, ecc_Arrival= {ecc_Arrival:.6g}")
    # velocity, arrival hyperbolic periapsis
    vp1_mars = math.sqrt((v_mag_inf_A**2) + (2 * mu_mars_km) / rp1_mars)
    print(f"periapsis, vp1_mars= {vp1_mars:.6g} [km/s]")

    # find speed needed for arrival ellipse; NOTE Curtis [3] p.480 eqn error,
    #   calculating sma (semi-major axis) sma=()^2/3 not sma=()^3/2
    print(f"parameters: {T_ellipse_mars} [s], {mu_mars_km}")
    sma_mars = ((T_ellipse_mars * math.sqrt(mu_mars_km)) / (2 * math.pi)) ** (2 / 3)
    print(f"sma_mars=, {sma_mars:.6g} [km]")
    # eccentricity
    ecc_arrive = 1 - (rp1_mars / sma_mars)
    print(f"ecc_arrive= {ecc_arrive:.6g}")
    # velocity, vehicle, ellipse at mars arrival; periapsis velocity; eqn.8.59
    v_v_elli_mars = math.sqrt((mu_mars_km / rp1_mars) * (1 + ecc_arrive))
    print(f"velocity, vehicle, ellipse @mars, {v_v_elli_mars:.6g} [km/s]")
    # arrival delta-v
    delta_v_arrive = vp1_mars - v_v_elli_mars
    print(f"mars, delta_v_arrive= {delta_v_arrive:.6g} [km/s]")

    return None


def test_curtis_ex8_3_soi():
    print(f"\nTest Curtis example 8.3, ... :")
    # function does not need input parameters.
    curtis_ex8_3_soi()
    return None


def test_curtis_ex8_4_depart():
    print(f"\nTest Curtis example 8.4, Earth->Mars, depart:")
    # function does not need input parameters.
    curtis_ex8_4_depart()
    return None


def test_depart_a():
    """
    Earth->Mars, depart Earth.
    Based on Curtis [3] pp.446, example 8.4.
    Assign:
        Planet departure parameters:
        Planet arrival parameters:
        Central body parameters:
    Find:
        (a) delta-v required
        (b) departure hyperbola perigee location
        (c) propellant as a percentage of the spacecraft, before delta-v burn
            assume Isp (specific impulse) = 300 [s]
    """
    print(f"\nTest patched conic depart_a(), Earth->Mars:")

    # earth = r1 & rp1
    r1 = 149598023  # [km], planet1 orbit, Vallado [4] p.1057, tbl.D-3
    rp1 = 6378.1363  # [km], planet1 radius, Vallado [4] p.1057, tbl.D-3
    rp1_alt = 300  # [km], given altitude above rp1
    rp1_mu = 3.986004415e5  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    # mars = r2 & rp2
    r2 = 227939186  # [km], planet2 orbit, Vallado [4] p.1057, tbl.D-3
    rp2 = 3397.2  # [km], planet2 radius, Vallado [4] p.1057, tbl.D-3
    rp2_alt = 300  # [km], given altitude above rp2
    rp2_mu = 4.305e4  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    # sun (central body)
    cb_mu = 1.32712428e11  # [km^3/s^2], Vallado [4] p.1059, tbl.D-5

    # departure planet 1 parameter list for function call
    depart = [r1, rp1, rp1_alt, rp1_mu]
    # arrival planet 2 parameter list for function call
    arrival = [r2, rp2, rp2_alt, rp2_mu]

    #
    v_inf, v_c, delta_v, beta_depart, ecc_depart = funColl.depart_a(
        depart=depart, arrival=arrival, cb_mu=cb_mu
    )
    # mass ratio calculation from Curtis [3] pp.446, example 8.4.
    I_sp = 300  # [s]
    g_0 = 9.81e-3  # [km/s^2] specific gravity (earth surface gravity)
    delta_mRatio = 1 - math.exp(-delta_v / (I_sp * g_0))

    print(f"depart v_infinity, v_inf = {v_inf:.6g} [km/s]")
    print(f"departure parking orbit, v_c= {v_c:.6g} [km/s]")
    print(f"delta_v to enter departure hyperbola = {delta_v:.6g} [km/s]")
    print(f"departure hyperbola beta angle= {beta_depart*180/math.pi:.6g} [deg]")
    print(f"eccentricity, departure hyperbola = {ecc_depart:.6g}")
    print(f"Propellant mass ratio = {delta_mRatio:.6g}")
    return None


def test_curtis_ex8_5_arrive():
    print(f"\nTest Curtis example 8.5, Earth->Mars, arrive:")
    # function does not need input parameters.
    curtis_ex8_5_arrive()
    return None


def test_arrive_b():
    """
    body1 (Earth_a) -> body2 (Mars_b), arrive at Mars.
        Related to Curtis [3] pp.456, example 8.5.
    After Hohmann transfer calculate arrival parameters, assuming satellite orbit period

    Input Parameters:
    ----------
        None
    Notes:
    ----------
        May help development; see https://github.com/jkloser/OrbitalMechanics
        Helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    print(f"\nTest patched conic arrive_b(), Earth_a->Mars_b:")

    # earth = r1 & rp1
    r1 = 149598023  # [km], planet1 orbit, Vallado [4] p.1057, tbl.D-3
    rp1 = 6378.1363  # [km], planet1 radius, Vallado [4] p.1057, tbl.D-3
    rp1_alt = 300  # [km], given altitude above rp1
    rp1_mu = 3.986004415e5  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    # mars = r2 & rp2
    r2 = 227939186  # [km], planet2 orbit, Vallado [4] p.1057, tbl.D-3
    rp2 = 3397.2  # [km], planet2 radius, Vallado [4] p.1057, tbl.D-3
    rp2_alt = 300  # [km], given altitude above rp2
    rp2_mu = 4.305e4  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    p2_sat_T = 7 * 60 * 60  # planet 2 satellite period [s]
    # sun (central body)
    cb_mu = 1.32712428e11  # [km^3/s^2], Vallado [4] p.1059, tbl.D-5

    # departure planet 1 parameter list for function call
    depart = [r1, rp1, rp1_alt, rp1_mu]
    # arrival planet 2 parameter list for function call
    arrive = [r2, rp2, rp2_alt, rp2_mu, p2_sat_T]

    v_inf, a_capture, rp2_ecc, delta_v, r_p, aim_radius, beta_p = funColl.arrive_b(
        depart=depart, arrive=arrive, cb_mu=cb_mu, p2_sat_T=p2_sat_T
    )
    print(f"arrive v_infinity, v_inf = {v_inf:.5g} [km/s]")
    print(f"arrive semi-major axis = {a_capture:.5g} [km]")
    print(f"eccentricity, mars orbit = {rp2_ecc:.5g}")
    print(f"delta_v enter mars = {delta_v:.5g} [km/s]")
    print(f"periapsis at mars, r_p = {r_p:.5g} [km]")
    print(f"aiming radius (aka delta) at mars = {aim_radius:.5g} [km]")
    print(f"angle to periapsis at mars = {(beta_p*180/math.pi):.5g} [deg]")

    return None


def test_curtis_ex8_6_flyby():
    print(f"\nTest Curtis example 8.6, Earth->Venus, fly-by:")
    # function does not need input parameters.
    curtis_ex8_6_flyby()
    return None


def test_flyby():
    """
    Earth->Venus fly-by.  Curtis [3] pp.462, example 8.6.
        Spacecraft departs earth with a velocity perpendicular to the sun line.
        Encounter occurs at a true anomaly in the approach trajectory of 30[deg].
        Periapse altitude 300 km.
    (a) Dark side Venus apporach.
            Post fly-by orbit shown in Figure 8.20.
    (b) Sunlit side Venus approach.
            Post fly-by orbit shown in Figure 8.21.

    Leading-side flyby results in a decrease in the spacecraft's heliocentric speed.
    Trailing-side flyby increases helliocentric speed;
        e1, h1, and θ1 are eccentricity, angular momentum,
        and true anomaly of heliocentric approach trajectory.

    Returns:
    ----------
    """
    print(f"\nTest fly-by function:")
    deg2rad = math.pi / 180

    # earth = r1 & rp1 (orbit parameters & body parameters)
    r1 = 149598023  # [km], planet1 orbit, Vallado [4] p.1057, tbl.D-3
    rp1_alt = 300  # [km], given altitude above rp1
    rp1 = 6378.1363  # [km], planet1 radius, Vallado [4] p.1057, tbl.D-3
    rp1_mu = 3.986004415e5  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3

    # venus = r2 & rp2 (orbit parameters & body parameters)
    r2 = 108208601  # [km], planet2 orbit, Vallado [4] p.1057, tbl.D-3
    rp2_alt = 300  # [km], given altitude above rp2
    rp2 = 6052  # [km], planet2 radius, Vallado [4] p.1057, tbl.D-3
    rp2_mu = 3.257e5  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    TA_t = -30 * deg2rad  # true anomaly/angle of transfer, given

    # sun (central body)
    cb_mu = 1.32712428e11  # [km^3/s^2], Vallado [4] p.1059, tbl.D-5

    # departure planet 1 parameter list for function call
    depart = [r1, rp1, rp1_alt, rp1_mu]
    # arrival planet 2 parameter list for function call
    arrive = [r2, rp2, rp2_alt, rp2_mu, TA_t]

    funColl.flyby(depart, arrive, cb_mu)
    return None


def test_curtis_ex8_7():
    print(f"\nTest Curtis example 8.7, planetary ephemeris:")
    print(f"(Associated with algorithm 8.1.)")
    # function does not need input parameters.
    curtis_ex8_7_earth_mars()
    return None


def test_curtis_ex8_8():
    print(f"\nTest Curtis example 8.8, planetary transfer parameters:")
    print(f"(associated with appendix 8.2)")
    # function does not need input parameters.
    curtis_ex8_8()
    return None


def test_curtis_ex8_9_10():
    print(f"\nTest Curtis example 8.9 & 8.10; transfer delta-t's:")
    # Need some parameters from curtis_ex8_8.
    curtis_ex8_9_10()
    return None


def test_imports():
    """
    Python's import is confusing.  I've spent hours and hours to figure it out.
    """
    coords, angle = np.array([1, 2, 3]), 30
    a = Braeunig.braeunigFunctions.rotate_coordinates(coords=coords, angle_deg=angle)
    print(f"import test, {a}")
    return


def main():
    # placeholder at the end of the file; helps my editor navigation
    return None


# use the following to test/examine functions
if __name__ == "__main__":
    # test naming convension,
    # test_curtis_ex8_3_soi()  # example 8.3; Earth->Sun soi
    # test_curtis_ex8_4_depart()  # example 8.4; Earth->Mars, depart
    # test_depart_a()  # Earth->Mars, depart function
    test_curtis_ex8_5_arrive()  # example 8.5; Earth->Mars, arrive
    # test_arrive_b()  # Earth->Mars, arrive function
    # test_curtis_ex8_6_flyby()  # example 8.6; Earth->Venus fly-by
    # test_flyby()  # Earth->Venus fly-by
    # test_curtis_ex8_7_earth_mars()  # Ephemeris, earth->mars
    # curtis_ex8_7_astropy()  # compare curtis ex8_7 planet positions
    # test_curtis_ex8_8()  # example 8.8; planetary transfer
    # test_curtis_ex8_9_10()  # depart & arrival, transfer delta-t's

    # test_imports() # python importing is a real problem, 2024-11-19
