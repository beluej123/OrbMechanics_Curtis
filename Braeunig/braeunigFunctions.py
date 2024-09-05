"""
Braeunig functions & function tests.
Some functions copied/edited from lamberthub ~2024 August.
Not sure how long the hyperlink below will work, but the inspiration came from
Braeunig's sections; Orbital Mechanics, Interplanetary Flight, and Example Problems.
The example problems were key to my appreciation, but since Braeunig example
problems variable names were not all ways consistant or clear I wrote this code:-)
http://braeunig.us/space/index.htm and http://www.braeunig.us/space/orbmech.htm

References
    ----------
    [1] Braeuning http://www.braeunig.us/space/interpl.htm
        Braeuning http://www.braeunig.us/space/orbmech.htm
    [2] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [3] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [4] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import math
from math import gamma

import numpy as np
import vallado_1
from numpy import dot
from numpy.linalg import norm
from validations_1 import assert_parameters_are_valid


def rotate_coordinates(coords, angle_deg):
    """rotate equatorial to ecliptic; rotate about X-axis
    https://community.openastronomy.org/t/trouble-rotating-coordinate-system/801

    Parameters
    ----------
    coords : numpy array
        xyz input
    angle_deg : _type_
        angle to rotate [deg]

    Returns
    -------
    _type_
        _description_
    """

    # Convert Degrees to Radians
    angle_rad = np.radians(angle_deg)
    # Rotation matrix around the X-axis
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    # Apply the Rotation
    return np.dot(rotation_matrix, coords)


def get_transfer_angle(r1, r2, prograde=True):
    """
    Find transfer angle, given r1, r2, sense of rotation (long or short)
    2024-08-13 Copied from LambertHub angles.py

    Parameters
    ----------
    r1: np.array
        Initial position vector.
    r2: np.array
        Final position vector.
    prograde: bool
        If True, prograde motion, otherwise retrograde motion.

    Return
    -------
    dtheta: float, angle between vectors [rad]
    """
    # Check if both position vectors are collinear. If so, check if the transfer
    # angle is 0 or pi.
    if np.all(np.cross(r1, r2) == 0):
        return 0 if np.all(np.sign(r1) == np.sign(r2)) else np.pi

    # Solve for a unitary vector normal to the vector plane. Its direction and
    # sense the one given by the cross product (right-hand) from r1 to r2.
    h = np.cross(r1, r2) / norm(np.cross(r1, r2))

    # Compute the projection of the normal vector onto the reference plane.
    alpha = dot(np.array([0, 0, 1]), h)

    # Get the minimum angle (0 <= dtheta <= pi) between r1 and r2.
    r1_norm, r2_norm = [norm(vec) for vec in [r1, r2]]
    theta0 = np.arccos(dot(r1, r2) / (r1_norm * r2_norm))

    # Fix the value of theta if necessary
    if prograde is True:
        dtheta = theta0 if alpha > 0 else 2 * np.pi - theta0
    else:
        dtheta = theta0 if alpha < 0 else 2 * np.pi - theta0

    return dtheta


def angle_between(a, b):
    """
    Angle between vectors.

    Parameters
    ----------
    a : np.array
        _description_
    b : np.array
        _description_

    Returns
    -------
    angle : float, angle between vectors [deg]

    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        print("Zero magnitude vector!")
    else:
        # take care of rounding errors
        arccosInput = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
        arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
        arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
        angle = math.acos(arccosInput)
        angle = np.degrees(angle)
        return angle
    return 0


def dot_product_angle(v1, v2):
    """angle between two vectors
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    Parameters
    ----------
    v1 : _type_
        _description_
    v2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(
            vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
        angle = np.degrees(arccos)
        return angle
    return 0


# *********************************************************
# copied stumff functions from lamberthub; for stumff() also in Curtis ex5.2
def c2(psi):
    r"""Second Stumpff function.
    For positive arguments:
    .. math::

        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}

    """
    eps = 1.0
    if psi > eps:
        res = (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -eps:
        res = (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
    else:
        res = 1.0 / 2.0
        delta = (-psi) / gamma(2 + 2 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 2 + 1)

    return res


def c3(psi):
    r"""Third Stumpff function.
    For positive arguments:
    .. math::
        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}

    """
    eps = 1.0
    if psi > eps:
        res = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    elif psi < -eps:
        res = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))
    else:
        res = 1.0 / 6.0
        delta = (-psi) / gamma(2 + 3 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 3 + 1)

    return res


# *********************************************************


def test_planets_ecliptic():

    from datetime import datetime, timedelta

    from astropy import units as u
    from astropy.coordinates import (
        get_body_barycentric,
        get_body_barycentric_posvel,
        solar_system_ephemeris,
    )
    from astropy.time import Time

    # Dates for Braeunig problem 5.3
    # tdb runs at uniform rate of one SI second per second; independent of Earth rotation irregularities.
    ts0 = Time("2020-07-20 0:0", scale="tdb")
    ts1 = ts0 + 207 * u.day  # t2 is 207 days later than t1
    print(f"date ts0 = {ts0}, Julian date: {ts0.jd}")
    print(f"date ts1 = {ts1}, Julian date: {ts1.jd}")

    with solar_system_ephemeris.set("de430"):  # times between years 1550 to 2650
        # with solar_system_ephemeris.set('de432s'):  # times between 1950 and 2050
        # earthBc = get_body_barycentric("earth", ts1, ephemeris='builtin')
        earthBc = get_body_barycentric("earth", ts0)  # equatorial (not ecliptic)
        marsBc = get_body_barycentric("mars", ts1)
        # astropy provides equatorial (not ecliptic)
        earthBc_pv = get_body_barycentric_posvel("earth", ts0)  # position & velocity
        marsBc_pv = get_body_barycentric_posvel("mars", ts1)

    # Rotate equatorial to ecliptic (earth tilt, X-axis)
    earth_xyz_ecl = rotate_coordinates(earthBc_pv[0].xyz.to(u.au), -23.4393)
    mars_xyz_ecl = rotate_coordinates(marsBc_pv[0].xyz.to(u.au), -23.4393)
    earth_vel_ecl = rotate_coordinates(earthBc_pv[1].xyz.to(u.km / u.s), -23.4393)
    mars_vel_ecl = rotate_coordinates(marsBc_pv[1].xyz.to(u.km / u.s), -23.4393)

    np.set_printoptions(formatter={"float": "{: 0.7f}".format})
    print(f"earth pos(ts0), astropy equatorial, {earthBc_pv[0].xyz.to(u.km)}")  # [km]
    print(f"earth pos(ts0), astropy equatorial, {earthBc_pv[0].xyz.to(u.au)}")  # [au]
    print(f"earth vel(ts0), astropy equatorial, {earthBc_pv[1].xyz.to(u.km / u.s)}")
    print()
    print(f"earth pos(ts0), astropy ecliptic, {earth_xyz_ecl}")
    print(f"earth orbit radius(ts0), {np.linalg.norm(earthBc_pv[0].xyz.to(u.au))}")
    print(f"earth orbit velocity(ts0), {np.linalg.norm(earth_vel_ecl)}")
    print()
    print(f"mars pos(ts1), astropy equatorial, {marsBc_pv[0].xyz.to(u.km)}")
    print(f"mars pos(ts1, astropy equatorial, {marsBc_pv[0].xyz.to(u.au)}")
    print(f"mars vel(ts1), astropy equatorial, {marsBc_pv[1].xyz.to(u.km / u.s)}")
    print()
    print(f"mars pos(ts1), astropy ecliptic {mars_xyz_ecl}")
    print(f"mars orbit radius(ts1), {np.linalg.norm(mars_xyz_ecl.to(u.au))}")
    print(f"mars orbit velocity(ts1), {np.linalg.norm(mars_vel_ecl)}")

    print()
    vectorAngle = dot_product_angle(earth_xyz_ecl, mars_xyz_ecl)
    print(f"earth-mars phase angle, dot_product_angle = {vectorAngle} [deg]")
    return None


def b_gauss(r1, r2, delta_nu: float, tof: float, GM: float):
    """
    Braeunig's Gauss Orbit Solution. P-iteration method.
    Taken from Braeunig text and problems 5.3 & 5.4.
    2024-August
    NOTE updated p-iteration, includes initial p values estimation.
        I tested resulst against several other routines;
        vallado_1.py (edited from lamberthub) agrees with values tested here.

    2024-08-09, TODO remains; check 4 valid inputs; check ellipse, parabols, hyperbola...
    For verifying inputs etc. checkout code in LambertHub
    The commented out print statements, below, can be manually uncommented for debugging.

    Input Parameters:
    ----------
        r1 : float
            distance from center to departure point
        r2 : float
            distance from center to arrival point
        delta_nu : _type_
            Change in true anomaly [deg]
        tof : float
            time of flight [sec]
        GM : float
            central body gravitional constant [au^3/s^2]

    Returns:
    -------
        p, sma_2, tof_2, f, g, f_dot, g_dot

    Notes:
    -----
        The Algorithm maybe singular for transfer angles of 180 degrees.
        TODO test for performance, and really small angles.

        References: see list at file beginning.
    """

    # convert input degrees to radians for trig calculations
    delta_nu1 = delta_nu * (math.pi / 180)
    k = r1 * r2 * (1 - math.cos(delta_nu1))  # BMWS, p.204, eqn 5-42
    l = r1 + r2
    m = abs(r1) * abs(r2) * (1 + math.cos(delta_nu1))
    # print(f"k={k:.8g}, l={l:.8g}, m={m:.8g}")

    # bracket p values for ellipse, sp_i & sp_ii; BMWS [2], p.205
    # values > p_ii will be hyperbolic trajectories
    # minimum sp for ellipse
    sp_i = k / (l + math.sqrt(2 * m))  # BMWS [2], p.208, eqn 5-52
    # maximum sp for ellipse; calculated value is actually a parabola
    sp_ii = k / (l - math.sqrt(2 * m))  # BMWS [2], p.208, eqn 5-53
    # p = 1.2  # [au] initial p assignment from Braeuning problem solution
    p = sp_i + (sp_ii - sp_i) / 2  # updated initial p assignment

    # initial 1, p value *******************
    p_1, sma_1, tof_1, f, g, f_dot, g_dot = gauss_cal_mid(
        r1=r1, r2=r2, GM=GM, delta_nu1=delta_nu1, m=m, k=k, p=p, l=l
    )
    # print(f"p_1={p_1:.8g}, sma_1={sma_1:.8g}, tof_1={(tof_1/(24*3600)):.8g}")

    # p = 1.3 # [au] 2nd p assignment from Braeuning problem solution
    p = p * 1.05

    # initital 2, p value *******************
    p_2, sma_2, tof_2, f, g, f_dot, g_dot = gauss_cal_mid(
        r1=r1, r2=r2, GM=GM, delta_nu1=delta_nu1, m=m, k=k, p=p, l=l
    )
    # print(f"p_2={p_2:.8g}, sma_2={sma_2:.8g}, tof_2={(tof_2/(24*3600)):.8g}")
    # initial 2 *******************

    maxiter = 10  # maximum iterations
    atol = 0.0001 * (24 * 3600)  # tof tolerance [day]
    for numiter in range(1, maxiter + 1):
        # print()
        # Compute new p estimate
        p_3 = p_2 + (tof - tof_2) * ((p_2 - p_1) / (tof_2 - tof_1))
        # print(f"p_3={p_3:.8g}")
        p = p_3

        # Compute new p, sma, tof
        p_2, sma_2, tof_2, f, g, f_dot, g_dot = gauss_cal_mid(
            r1=r1, r2=r2, GM=GM, delta_nu1=delta_nu1, m=m, k=k, p=p, l=l
        )
        # print(f"p_2={p_2:.8g}, sma_2={sma_2:.8g}, tof_2={(tof_2/(24*3600)):.8g}")

        # Check the convergence of the method
        # print(
        #     f"loop {numiter}, abs(tof-tof_2)= {(abs((tof-tof_2))/(24*3600)):.8g} [day]"
        # )
        if abs(tof - tof_2) <= atol:
            break
        else:
            # The new initial guess is the previously computed y value
            p = p_2
    else:  # exceeded loop value
        print(f"*** exceeded iteration max, {maxiter} ***")

    # print(f"finals:")
    # print(f"p= {p:.8g}, sma_2={sma_2:.8g}, tof_2={(tof_2/(24*3600)):.8g}")
    return (p, sma_2, tof_2, f, g, f_dot, g_dot)


def gauss_cal_mid(
    r1: float,
    r2: float,
    GM: float,
    delta_nu1: float,
    m: float,
    k: float,
    p: float,
    l: float,
):
    # semi-major axis; ref[2], BMWS p.204, eqn.5-46
    sma = (m * k * p) / ((2 * m - l**2) * p**2 + (2 * k * l * p - k**2))
    # print(f"sma={sma:.8g} [au]") # for debug

    # compute f, g, f_dot, g_dot for future solving v1 and v2
    f = 1 - (r2 / p) * (1 - math.cos(delta_nu1))
    g = (r1 * r2 * math.sin(delta_nu1)) / math.sqrt(GM * p)
    f_dot = (
        (math.sqrt(GM / p))
        * (math.tan((delta_nu1 / 2)))
        * (((1 - math.cos(delta_nu1)) / p) - (1 / r1) - (1 / r2))
    )  # f_dot
    g_dot = 1 - (r1 / p) * (1 - math.cos(delta_nu1))
    dE_a = math.acos(1 - (r1 / sma) * (1 - f))  # delta Eeecntric anomaly
    tof = g + (math.sqrt(sma**3 / GM)) * (dE_a - math.sin(dE_a))
    # print(f"f= {f:.8g}, g= {g:.8g}, f_dot= {f_dot:.8g}, g_dot= {g_dot:.8g}") # for debug
    # print(f"time of flight-1, tof={tof:.8g} [s]") # for debug
    # print(f"time of flight-1, tof={tof/(24*3600):.8g} [day]") # for debug

    return (p, sma, tof, f, g, f_dot, g_dot)


def test_b_p4_28(plot_sp=False):
    """
    Find time of flight (tof) of hyperbolic trajectory.
    Braeunig problem 4.28.
    Given:
        Earth-centric hyperbolic trajectory launch
        sma (semi-major axis) = -36,000 [km] # negative value = hyperbolic
        ecc (eccentricity) = 1.1823 [km]
        TA0 (true angle/anomaly) = 15 [deg]
        TA1 (true angle/anomaly) = 120 [deg]
    Find:
        tof to move from TA0 to TA1; remember earth-centric

    Returns
    -------
        None

    Notes
    -------
        Example problems http://braeunig.us/space/problem.htm#4.28
        Detailed explanations http://braeunig.us/space/
        References: see list at file beginning.
    """

    print(f"\nTest Braeunig problem 4.28:")
    # constants
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3
    # given parameters
    sma = -36000  # [km]
    ecc = 1.1823
    TA0 = 15 * math.pi / 180  # [rad] conversion
    TA1 = 120 * math.pi / 180  # [rad] conversion

    cosh_F0 = (ecc + math.cos(TA0)) / (1 + ecc * math.cos(TA0))
    F0 = math.acosh(cosh_F0)

    cosh_F1 = (ecc + math.cos(TA1)) / (1 + ecc * math.cos(TA1))
    F1 = math.acosh(cosh_F1)

    tof = math.sqrt(((-(sma**3))) / mu_earth_km) * (
        (ecc * math.sinh(F1) - F1) - (ecc * math.sinh(F0) - F0)
    )

    # tof (time of flight)
    print(f"time of flight, tof= {tof:.6g} [s], {tof/(3600):.6g} [day]")

    # extra calculations; explore sp vs. sma plot
    if plot_sp == True:
        sp = sma * (1 - ecc**2)
        r0_mag = sp / (1 + ecc * math.cos(TA0))
        r1_mag = sp / (1 + ecc * math.cos(TA1))
        delta_nu = TA1 - TA0  # [rad]
        # plot_sp=True, to see possible range of orbit parameters plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker.
        # note, since sma may be near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(
            r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=delta_nu, sp=sp, clip1=True
        )

    return None  # test_b_p4_28()


def test_b_p4_29():
    """
    Find hyperbolic excess velocity.
    Braeunig problem 4.29; related to problem (prob) 4.25.
    Remember, keeping track of which central body influence can be tricky.
    Given:
        r0_alt : [km] earth altitude 200
        v0_bo  : [km/s] burn-out velocity = 11.5; satellite orbital velocity
        v0_esc : [km/s] escape velocity = 11.009, from prob 4.25
    Find:
        v0_inf : [km/s] hyperbolic excess velocity; at infinity

    Returns:
    -------
        none

    Notes:
    -------
        Example problems http://braeunig.us/space/problem.htm#4.29
        Detailed explanations http://www.braeunig.us/space/orbmech.htm
        References: see list at file beginning.
    """

    print(f"\nTest Braeunig problem 4.29:")
    # constants
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    # calculations from prob 4.25, below
    r0_alt = 200  # [km]
    r0_mag = r_earth + r0_alt  # [km]
    # earth escape velocity starting from r0
    v0_esc = math.sqrt(2 * mu_earth_km / r0_mag)  # [km/s]
    print(f"Earth(r0_alt) escape; v0_esc= {v0_esc:.8g} [km/s]")
    # calculations from prob 4.25, above

    # calculations for this problem (4.29)
    v0_bo = 11.5  # [km/s] given; spacecraft orbit velocity, relative to Earth
    v0_inf = math.sqrt(v0_bo**2 - v0_esc**2)
    print(f"v0_inf= {v0_inf:.8g} [km/s]")

    return None  # test_b_p4_29()


def test_b_gauss_p5_1(plot_sp=False):
    """
    Earth->Mars mission, one tangent burn.
    Test Braeunig problem 5.1.
    Given:
        One-tangent burn, Earth to Mars
        r0_mag : [au] Earth departure radius vector,
        r1_mag : [au] Mars arrival radius vector, 1.524,
        sma_tx : [au] Transfer orbit semi-major axis, 1.3.
    Find:
        tof    : [s] Transfer tof (time-of-flight); Earth->Mars,
        nu_tx  : Change in true angle/anomaly

    Notes:
    ----------
        Note, sp=semi-parameter; aka p.
        Enabling plot allows visibility to range of orbits based on sma vs. sp.
        Also see Vallado [3] section 6.3.3, p.331+
        Example problems http://braeunig.us/space/problem.htm#5.1
        Detailed explanations http://www.braeunig.us/space/interpl.htm

        Use ecliptic coordinates.
        References: see list at file beginning.

    Return:
    -------
        None
    """
    print(f"\nTest Braeunig problem 5.1; one-tangent burn, given sma:")
    # constants from Vallado, not Braeuning
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    # Vector magnitude, initial and final position
    r0_mag = 1.0  # [au] given, Earth orbit radius
    r1_mag = 1.524  # [au] given, Mars orbit radius
    # note, given the sma_tx orbit is slightly larger then Mars orbit
    sma_tx = 1.3  # [au] given, semi-major axis

    # calculate transfer eccentricity
    ecc_tx = 1 - (r0_mag / sma_tx)
    print(f"transfer eccentricity, ecc_tx= {ecc_tx:.6g}")
    nu_tx = math.acos(((sma_tx * (1 - ecc_tx**2) / r1_mag) - 1) / ecc_tx)
    print(
        f"true anomaly/angle, nu_tx= {nu_tx:.6g} [rad], {nu_tx*180/math.pi:.6g} [deg]"
    )

    # eccentric anomaly/angle related to transfer ellipse
    E = math.acos((ecc_tx + math.cos(nu_tx)) / (1 + ecc_tx * math.cos(nu_tx)))
    print(f"eccentric anomaly/angle, E= {E:.6g} [rad], {E*180/math.pi:.6g} [deg]")

    # transfer time of flight, tof; remember units for sma_tx
    tof_tx = (E - ecc_tx * math.sin(E)) * math.sqrt((sma_tx * au) ** 3 / mu_sun_km)
    print(f"time of flight, tof_tx= {tof_tx:.6g} [s], {tof_tx/(24*3600):.6g} [day]")

    # extra calculations to plot range of possible orbits
    if plot_sp == True:
        sp = sma_tx * (1 - ecc_tx**2)  # semi-parameter, aka p
        # to see range pf possible orbits, plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker; if sp not defined.
        # note, since sma may go near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=nu_tx, sp=sp, clip1=True)
    return None


def test_b_gauss_p5_2(plot_sp=False):
    """
    Earth->Mars mission, one tangent burn.
    Test Braeunig problem 5.2. Related to problem 5.1.
    Given (some givens (*) come from problem 5.1):
        * One-tangent burn, Earth to Mars
        * r0_mag   : [au] Earth departure radius magnitude,
        * r1_mag   : [au] Mars arrival radius magnitude,
        * sma_tx   : [au] Transfer orbit semi-major axis,
        omega_mars : [deg/day] Mars orbit angular rotation rate.
    Find:
        * tof      : [s] Transfer tof (time-of-flight); Earth->Mars,
        * nu_tx    : Change in true angle/anomaly,
        gamma_d    : phase angle Earth->Mars at launch (departure from Earth).

    Notes:
    ----------
        Note, sp=semi-parameter; aka p.
        Enabling plot allows visibility to range of orbits based on sma vs. sp.
        Also see Vallado [3] section 6.3.3, p.331+
        Example problems http://braeunig.us/space/problem.htm#5.2
        Detailed explanations http://www.braeunig.us/space/interpl.htm

        Use ecliptic coordinates.
        References: see list at file beginning.

    Return:
    -------
        None
    """
    print(f"\nTest Braeunig problem 5.2; one-tangent burn:")

    # parameters and calculations from problem 5.1, below
    # constants from Vallado, not Braeuning
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado p.1042, tbl.D-5
    r_earth = 6378.1363  # [km] earth radius; Vallado p.1041, tbl.D-3

    # Vector magnitude, initial and final position
    r0_mag = 1.0  # [au] given, Earth orbit radius
    r1_mag = 1.524  # [au] given, Mars orbit radius
    sma_tx = 1.3  # [au] given, semi-major axis

    # calculate transfer eccentricity
    ecc_tx = 1 - (r0_mag / sma_tx)
    print(f"transfer eccentricity, ecc_tx= {ecc_tx:.6g}")
    nu_tx = math.acos(((sma_tx * (1 - ecc_tx**2) / r1_mag) - 1) / ecc_tx)
    print(
        f"true anomaly/angle, nu_tx= {nu_tx:.6g} [rad], {nu_tx*180/math.pi:.6g} [deg]"
    )

    # eccentric anomaly/angle related to transfer ellipse
    E = math.acos((ecc_tx + math.cos(nu_tx)) / (1 + ecc_tx * math.cos(nu_tx)))
    print(f"eccentric anomaly/angle, E= {E:.6g} [rad], {E*180/math.pi:.6g} [deg]")

    # transfer time of flight, tof; remember units for sma_tx
    tof_tx = (E - ecc_tx * math.sin(E)) * math.sqrt(
        (sma_tx * au) ** 3 / mu_sun_km
    )  # [s]
    print(f"time of flight, tof_tx= {tof_tx:.6g} [s], {tof_tx/(24*3600):.6g} [day]")
    # parameters and calculations from problem 5.1, above

    # convert given [degrees/day] to [rad/s]
    omega_mars = 0.524 * (math.pi / 180) / (24 * 3600)  # [rad/s] convert

    # departure phase between earth-mars, gamma_d
    gamma_d = nu_tx - omega_mars * tof_tx
    print(
        f"departure phase between earth-mars, gamma_d= {gamma_d:.6g} [rad], {gamma_d*180/math.pi:.6g} [deg]"
    )
    if plot_sp == True:
        sp = sma_tx * (1 - ecc_tx**2)  # semi-parameter, aka p
        # to see range pf possible orbits, plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker; if sp not defined.
        # note, since sma may go near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=nu_tx, sp=sp, clip1=True)
    return None


def test_b_gauss_p5_3(plot_sp=False):
    """
    Calculate semi-parameter and semi-major axis Earth->Mars transfer orbit.
    Test Braeunig problem 5.3.
    Earth->Mars mission launch 2020-7-20, 0:00 UT, planned time of flight 207 days.
    TODO, check Earth, Mars orbit positions with Skyfield or astropy.
    Earth's at departure is 0.473265X - 0.899215Y AU.
    Mars' at intercept is 0.066842X + 1.561256Y + 0.030948Z AU.
    Note: updated (2024-Aug) b_gauss() function includes calculated initial p-values.

     Given:
        Earth to Mars mission
        tof      : [day] Transfer tof (time-of-flight); Earth->Mars,
        r0_vec   : [au] Earth departure radius vector,
        r1_vec   : [au] Mars arrival radius vector.
    Find:
        sp_tx    : transfer semi-parameter, aka p,
        sma_tx   : transfer semi-major axis.

    Notes:
    ----------
        Note, sp=semi-parameter; aka p.
        Enabling plot allows visibility to range of orbits based on sma vs. sp.
        Also see Vallado [3] section 7.6, pp.467; Lambert's problem.
        Example problems http://braeunig.us/space/problem.htm#5.3
        Detailed explanations http://www.braeunig.us/space/interpl.htm

        Use ecliptic coordinates.
        References: see list at file beginning.

    Return:
    -------
        None
    """
    print(f"\nTest Braeunig problem 5.3, Earth->Mars, given tof:")
    # Earth(t0) & Mars(t1) vectors; initial and final position
    r0_vec = np.array([0.473265, -0.899215, 0.0])  # [au] earth(t0)
    r1_vec = np.array([0.066842, 1.561256, 0.030948])  # [au] mars(t1)
    r0_mag, r1_mag = [np.linalg.norm(r) for r in [r0_vec, r1_vec]]
    print(f"magnitudes: r0= {r0_mag:.8g} [au], r1= {r1_mag:.8g}")

    # constants from Vallado, not Braeuning
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    GM_sun_au = GM_sun_km / (au**3)  # unit conversion

    # compute angle between vectors; note, problem statement gives angle
    delta_nu = get_transfer_angle(r0_vec, r1_vec, prograde=True)  # [rad]
    delta_nu_deg = delta_nu * 180 / math.pi  # [deg] convert
    print(f"delta_nu= {delta_nu_deg:.6g} [deg]")
    tof = 207 * 24 * 60 * 60  # [s]

    sp_tx, sma_tx, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r0_mag, r2=r1_mag, delta_nu=delta_nu_deg, tof=tof, GM=GM_sun_au
    )
    print(
        f"sp= {sp_tx:.8g} [au], sma= {sma_tx:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]"
    )
    ecc = math.sqrt(1 - (sp_tx / sma_tx))
    print(f"eccentricity, ecc= {ecc:.6g}")

    if plot_sp == True:
        # to see range pf possible orbits, plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker; if sp not defined.
        # note, since sma may go near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(
            r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=delta_nu, sp=sp_tx, clip1=True
        )

    return None  # test_b_gauss_p5_3()


def test_b_gauss_p5_4(plot_sp=False):
    """
    test Braeunig problem 5.4 (includes 5.3 calculation results).
    For Earth->Mars of problem 5.3, calculate departure and intecept velocity vectors.
    Note: updated (2024-Aug) b_gauss() function includes calculated initial p-values.

     Given (some givens (*) come from problem 5.3):
        Earth to Mars mission
        * tof      : [day] Transfer tof (time-of-flight); Earth->Mars,
        * r0_vec   : [au] Earth departure radius vector,
        * r1_vec   : [au] Mars arrival radius vector.
    Find (some finds (*) come from problem 5.3):
        * sp_tx    : transfer semi-parameter, aka p,
        * sma_tx   : transfer semi-major axis,
        v0_vec     : Earth departure velocity
        v1_vec     : Mars intercept velocity

    Notes:
    ----------
        Note, sp=semi-parameter; aka p.
        Enabling plot allows visibility to range of orbits based on sma vs. sp.
        Also see Vallado [3] section 6.3.3, pp.331.
        Example problems http://braeunig.us/space/problem.htm#5.4
        Detailed explanations http://www.braeunig.us/space/interpl.htm

        Use ecliptic coordinates.
        References: see list at file beginning.

    Return:
    -------
        None
    """
    print(f"\nTest Braeunig problem 5.4, depart/intercept velocity vectors:")
    # below, calculations from problem 5.3
    # Earth(t0) & Mars(t1) vectors; initial and final position
    r0_vec = np.array([0.473265, -0.899215, 0.0])  # [au] earth(t0)
    r1_vec = np.array([0.066842, 1.561256, 0.030948])  # [au] mars(t1)
    r0_mag, r1_mag = [np.linalg.norm(r) for r in [r0_vec, r1_vec]]
    print(f"magnitudes: r0= {r0_mag:.8g} [au], r1= {r1_mag:.8g}")

    # constants from Vallado, not Braeuning
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    GM_sun_au = GM_sun_km / (au**3)  # unit conversion

    # compute angle between vectors; note, problem statement gives angle
    delta_nu = get_transfer_angle(r0_vec, r1_vec, prograde=True)  # [rad]
    delta_nu_deg = delta_nu * 180 / math.pi  # [deg] convert
    print(f"delta_nu= {delta_nu_deg:.6g} [deg]")
    tof = 207 * 24 * 60 * 60  # [s]

    sp_tx, sma_tx, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r0_mag, r2=r1_mag, delta_nu=delta_nu_deg, tof=tof, GM=GM_sun_au
    )
    print(
        f"sp= {sp_tx:.8g} [au], sma= {sma_tx:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]"
    )
    ecc = math.sqrt(1 - (sp_tx / sma_tx))
    print(f"eccentricity, ecc= {ecc:.6g}")
    # above, calculations from problem 5.3

    v0_vec = (r1_vec - f * r0_vec) / g  # [au/s]
    print(f"v0_vec= {v0_vec*au} [km/s]")  # convert [au] to [km]
    v1_vec = f_dot * r0_vec + g_dot * v0_vec
    print(f"v2_vec= {v1_vec*au} [km/s]")  # convert [au] to [km]

    if plot_sp == True:
        # to see range pf possible orbits, plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker; if sp not defined.
        # note, since sma may go near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(
            r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=delta_nu, sp=sp_tx, clip1=True
        )
    return None  # test_b_gauss_p5_4()


def test_b_gauss_p5_5(plot_sp=False):
    # test Braeunig problem 5.5 (includes 5.3, 5.4).
    # For Earth->Mars mission of problem 5.3,  calculate transfer orbit orbital elements.
    # For this problem use r1_vec & v1_vec; can also use r2_vec & v2_vec.
    # Example problems http://braeunig.us/space/problem.htm#5.5

    # NOTE updated b_gauss() function includes calculated initial p-values.

    # Use ecliptic coordinates.
    print(f"\ntest Braeunig problem 5.5:")
    # Vector magnitude, initial and final position
    r1_vec = np.array([0.473265, -0.899215, 0.0])  # earth(t0) position [AU]
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # mars(t1) position [AU]
    r1, r2 = [np.linalg.norm(r) for r in [r1_vec, r2_vec]]
    # print(f"magnitudes: r1= {r1:.8g} [au], r2= {r2:.8g}")

    GM_sun = 3.964016e-14  # [au^3/s^2] sun
    km_au = 149.597870e6  # [km/au]; convert: [au] to [km]
    delta_nu = 149.770967  # [deg]
    tof = 207 * 24 * 60 * 60  # [s]

    p, sma, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r1, r2=r2, delta_nu=delta_nu, tof=tof, GM=GM_sun
    )
    print(f"p= {p:.8g} [au], sma= {sma:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]")
    print(f"f= {f:.8g}, g= {g:.8g}, f_dot= {f_dot:.8g}, g_dot= {g_dot:.8g}")

    v1_vec = (r2_vec - f * r1_vec) / g  # [au/s]
    print(f"v1_vec= {v1_vec*km_au} [km/s]")  # convert [au] to [km]
    v2_vec = f_dot * r1_vec + g_dot * v1_vec
    print(f"v2_vec= {v2_vec*km_au} [km/s]")  # convert [au] to [km]

    # ****************************************************
    # be careful of units !! au vs. km
    GM = 132712.4e6  # [km^3/s^2] sun, new value for GM from above calculations !!!
    r1_vec = r1_vec * km_au  # convert to km from au
    v1_vec = v1_vec * km_au  # convert to km from au

    r1_mag = np.linalg.norm(r1_vec)
    v1_mag = np.linalg.norm(v1_vec)  # v1 magnitude
    print(f"v1_mag= {v1_mag:.8g} [km/s], r1_mag= {r1_mag:.8g} [km]")
    # angular momentum vector, h
    h1_vec = np.cross(r1_vec, v1_vec)
    h1_mag = np.linalg.norm(h1_vec)
    print(f"h1_vec= {h1_vec}, h1_mag= {h1_mag:.8g}")

    n1_vec = np.cross([0.0, 0.0, 1.0], h1_vec)  # line-of-nodes; z cross h_vec
    n1_mag = np.linalg.norm(n1_vec)
    print(f"normal to h1, n1_vec= {n1_vec}")

    # calculate eccentricity; ecc_vec; points along apsis, periapsis
    ecc_vec = (
        (v1_mag**2 - GM / r1_mag) * r1_vec - np.dot(r1_vec, v1_vec) * v1_vec
    ) / GM
    ecc_mag = np.linalg.norm(ecc_vec)
    print(f"ecc_vec= {ecc_vec}, ecc_mag= {ecc_mag:.8g}")

    # calculate semi-major axis, sma; in [km] this time
    sma_km = 1 / ((2 / r1_mag) - (v1_mag**2 / GM))
    print(f"sma_km= {sma_km:.8g} [km]")

    # calculate inclination
    inc_rad = math.acos(h1_vec[2] / h1_mag)
    print(f"inclination, {inc_rad:.8g} [rad], {inc_rad*180/np.pi} [deg]")

    # calculate longitude of ascending node (Omega)
    long_asc_node = math.acos(n1_vec[0] / n1_mag)
    if n1_vec[1] < 0:  # check y-component
        long_asc_node = 2 * math.pi - long_asc_node
    print(
        f"longitude of ascending node, long_asc_node= {long_asc_node:.8g} [rad], {long_asc_node*180/np.pi:.8g} [deg]"
    )

    # calculate angle/argument of periapsis (omega)
    angle_peri = math.acos(np.dot(n1_vec, ecc_vec) / (n1_mag * ecc_mag))
    print(
        f"angle of periapsis, angle_peri_rad= {angle_peri:.8g} [rad], {angle_peri*180/np.pi:.8g} [deg]"
    )

    # calculate true angle/anomaly
    nu_o = math.acos(np.dot(r1_vec, ecc_vec) / (r1_mag * ecc_mag))
    print(f"true anomaly, nu_o= {nu_o:.8g} [rad], {nu_o*180/np.pi:.8g} [deg]")

    # calculate u_o, not sure what this is called
    u_o = math.acos(np.dot(n1_vec, r1_vec) / (n1_mag * r1_mag))
    print(f"u_o= {u_o:.8g} [rad], {u_o*180/np.pi:.8g} [deg]")

    # calculate longitude of periapsis (often called pie, sometimes in place of )
    long_peri = long_asc_node + angle_peri
    print(
        f"longitude of periapsis, long_peri= {long_peri:.8g} [rad], {long_peri*180/np.pi:.8g} [deg]"
    )

    # calculate true longitude
    long_o = long_asc_node + angle_peri + nu_o
    print(f"true longitude, long_o= {long_o:.8g} [rad], {long_o*180/np.pi:.8g} [deg]")

    return None


def test_b_gauss_p5_6():
    # test Braeunig problem 5.6 (includes 5.3, 5.4).
    # For Earth->Mars mission of problem 5.3,  calculate the hyperbolic excess
    # velocity at departure, the injection deltaV, and the zenith angle of the departure
    # asymptote.  Injection occurs from earth 200 km parking orbit.  Earth's velocity
    # vector at departure is 25876.6X + 13759.5Y m/s.
    # Example problems http://braeunig.us/space/problem.htm#5.6

    # NOTE updated b_gauss() function includes calculated initial p-values.

    # Use ecliptic coordinates.
    print(f"\ntest Braeunig problem 5.6:")
    # Vector magnitude, initial and final position
    r1_vec = np.array([0.473265, -0.899215, 0.0])  # earth(t0) position [AU]
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # mars(t1) position [AU]
    r1, r2 = [np.linalg.norm(r) for r in [r1_vec, r2_vec]]
    r_eo = 6378.2 + 200  # [km] radius to earth orbit
    # print(f"magnitudes: r1= {r1:.8g} [au], r2= {r2:.8g}")

    GM_sun = 3.964016e-14  # [au^3/s^2] sun
    km_au = 149.597870e6  # [km/au]; convert: [au] to [km]
    delta_nu = 149.770967  # [deg]
    tof = 207 * 24 * 60 * 60  # [s]

    p, sma, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r1, r2=r2, delta_nu=delta_nu, tof=tof, GM=GM_sun
    )
    print(f"p= {p:.8g} [au], sma= {sma:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]")
    print(f"f= {f:.8g}, g= {g:.8g}, f_dot= {f_dot:.8g}, g_dot= {g_dot:.8g}")

    v1_vec = (r2_vec - f * r1_vec) / g  # [au/s] at earth
    # print(f"v1_vec= {v1_vec*km_au} [km/s]")  # convert [au] to [km]
    v2_vec = f_dot * r1_vec + g_dot * v1_vec  # [au/s] at mars
    # print(f"v2_vec= {v2_vec*km_au} [km/s]")  # convert [au] to [km]

    # ****************************************************

    # be careful of units !! au vs. km
    vp_vec = np.array(
        [25.8766, 13.7595, 0.0]
    )  # [km/s], close to astropy ephemeris value
    print(f"earth velocity vector, vp_vec= {vp_vec} [km/s]")

    GM_sun = 132712.4e6  # [km^3/s^2] sun
    GM_earth = 398600.5  # [km^3/s^2] earth
    r1_vec = r1_vec * km_au  # convert to km from au
    r1_mag = np.linalg.norm(r1_vec)  # [km]
    v1_vec = v1_vec * km_au  # convert to km from au; from 5.4
    vs_vec = v1_vec  # velocity of satellite
    print(f"earth departure satellite, vs_vec= {vs_vec} [km/s]")

    vsp_vec = vs_vec - vp_vec
    vsp_mag = np.linalg.norm(vsp_vec)
    print(f"vel vector satellite-planet, vsp_vec= {vsp_vec} [km/s]")
    print(f"vel magnitude satellite-planet, vsp_mag= {vsp_mag:.8g} [km/s]")

    # earth launch/injection conditions, eqn.5.35; assume v_inf = vsp
    v_it = math.sqrt(
        vsp_mag**2 + (2 * GM_earth / r_eo)
    )  # velocity injection to transfer
    print(f"vel, injection to transfer, v_it= {v_it:.8g} [km/s]")

    # earth launch/injection deltaV_i, eqn.5.36
    dv_it = v_it - math.sqrt(GM_earth / r_eo)  # velocity injection to transfer
    print(f"delta vel, injection to transfer, dv_it= {dv_it:.8g} [km/s]")

    # calculate departure asymptote, gamma, g_it
    print(f"earth to sun, r1_mag= {r1_mag:.8g} [km]")
    g_it = math.acos(np.dot(r1_vec, vsp_vec) / (r1_mag * vsp_mag))
    print(
        f"departure asymptote, gamma, g_it= {g_it:.8g} [rad], {g_it*180/math.pi:.8g} [deg]"
    )
    return None


def test_b_gauss_p5_7():
    # test Braeunig problem 5.7 (includes 5.3, 5.4).
    # For Earth->Mars mission of problem 5.3. Given mars arrival miss distance
    # +18,500 km, calculate hyperbolic excess velocity, impact parameter,
    # semi-major axis, and eccentricity of the hyperbolic approach trajectory.
    # Mars' velocity vector at SOI intercept is -23307.8X + 3112.0Y + 41.8Z m/s.
    # SOI=sphere of influence.
    # Example problems http://braeunig.us/space/problem.htm#5.7
    # Detailed explanations http://braeunig.us/space/

    # NOTE updated b_gauss() function includes calculated initial p-values.

    # Use ecliptic coordinates.
    print(f"\ntest Braeunig problem 5.7:")
    # Vector magnitude, initial and final position
    r1_vec = np.array([0.473265, -0.899215, 0.0])  # earth(t0) position [AU]
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # mars(t1) position [AU]
    r1, r2 = [np.linalg.norm(r) for r in [r1_vec, r2_vec]]

    # print(f"magnitudes: r1= {r1:.8g} [au], r2= {r2:.8g}")

    GM_sun = 3.964016e-14  # [au^3/s^2] sun
    km_au = 149.597870e6  # [km/au]; convert: [au] to [km]
    delta_nu = 149.770967  # [deg]
    tof = 207 * 24 * 60 * 60  # [s]

    p, sma, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r1, r2=r2, delta_nu=delta_nu, tof=tof, GM=GM_sun
    )
    # print(f"p= {p:.8g} [au], sma= {sma:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]")
    # print(f"f= {f:.8g}, g= {g:.8g}, f_dot= {f_dot:.8g}, g_dot= {g_dot:.8g}")

    v1_vec = (r2_vec - f * r1_vec) / g  # [au/s] at earth
    print(f"vel(t0,earth), v1_vec= {v1_vec*km_au} [km/s]")  # convert [au] to [km]
    v2_vec = f_dot * r1_vec + g_dot * v1_vec  # [au/s] at mars
    print(f"vel(t1,mars), v2_vec= {v2_vec*km_au} [km/s]")  # convert [au] to [km]

    # ****************************************************
    # be careful of units !! au vs. km
    mars_mis_dist = 18500 / km_au  # [au] mars miss distance

    # mars/planet intercept vector; given; estimate at some point
    vp_vec = np.array([-23.3078, 3.112, 0.0418])  # [km/s]
    print(f"mars SOI intercept, vp_vec= {vp_vec} [km/s]")

    GM_sun = 132712.4e6  # [km^3/s^2] sun
    GM_earth = 398600.5  # [km^3/s^2] earth
    GM_mars = 42828.31  # [km^3/s^2] mars
    r2_vec = r2_vec * km_au  # convert to km from au
    r2_mag = np.linalg.norm(r2_vec)  # [km]
    v2_vec = v2_vec * km_au  # convert to km from au; from 5.4
    vs_vec = v2_vec  # velocity of satellite
    print(f"mars arrival satellite, vs_vec= {vs_vec} [km/s]")

    vsp_vec = vs_vec - vp_vec
    vsp_mag = np.linalg.norm(vsp_vec)
    print(f"vsp_vec= {vsp_vec} [km/s]")
    print(f"vsp_mag= {vsp_mag:.8g} [km/s]")

    # desire to miss the arrival planet by, mars_mis_dist
    print(f"r2_x, {r2_vec[0]/km_au}, r2_y, {r2_vec[1]/km_au}, r2_z, {r2_vec[2]}")
    d_x = (-mars_mis_dist * r2_vec[1]) / (math.sqrt(r2_vec[0] ** 2 + r2_vec[1] ** 2))
    d_y = mars_mis_dist * r2_vec[0] / math.sqrt(r2_vec[0] ** 2 + r2_vec[1] ** 2)
    print(f"d_x= {d_x:.9f}, d_y= {d_y:.9f}")

    theta = math.acos((d_x * vsp_vec[0] + d_y * vsp_vec[1]) / (mars_mis_dist * vsp_mag))
    print(f"theta= {theta:.8g} [rad], {theta*180/math.pi:.8g} [deg]")

    # impact parameter, b_impact
    b_impact = (mars_mis_dist * km_au) * math.sin(theta)
    print(f"impact parameter, b_impact= {b_impact:.8g} [km]")

    # mars arrival hyperbola sma (semi-major axis)
    sma_arival = -GM_mars / vsp_mag**2
    print(f"arrival semi-major axis, sma_arrival= {sma_arival:.8g} [km]")

    # mars arrival hyperbola eccentricity, ecc_arrival
    ecc_arrival = math.sqrt(1 + (b_impact**2 / sma_arival**2))
    print(f"mars arrival eccentricity, ecc_arrival= {ecc_arrival:.8g}")
    return None


def test_b_gauss_p5_8():
    # test Braeunig problem 5.8.
    # Jupiter fly-by mission. Spacecraft Jupiter approach with a velocity of 9,470 m/s,
    # a flight path angle of 39.2 degrees, and a targeted miss distance of -2,500,000 km.
    # At Jupiter intercept, Jupiter's velocity is 12,740 m/s with a flight path angle of 2.40
    # degrees.  Calculate the spacecraft's velocity and flight path angle following
    # its swing-by of Jupiter.
    # Example problems http://braeunig.us/space/problem.htm#5.8
    # Detailed explanations http://braeunig.us/space/

    # Use ecliptic coordinates.
    print(f"\ntest Braeunig problem 5.8:")

    vp_mag = 12.74  # [km/s]
    phi_p = 2.4 * math.pi / 180  # [rad] flight path angle, planet
    vsi_mag = 9.47  # [km/s] spacecraft initial velocity
    phi_si = 39.2 * math.pi / 180  # [rad] flight path angle, satellite initial
    jup_mis_dist = -2500000  # [km] mars miss distance

    GM_sun = 132712.4e6  # [km^3/s^2] sun
    GM_earth = 398600.5  # [km^3/s^2] earth
    GM_mars = 42828.31  # [km^3/s^2] mars
    GM_jup = 1.26686e8  # [km^3/s^2] jupiter

    # Planet vector initial position
    vp_vec = np.array([vp_mag * math.cos(phi_p), vp_mag * math.sin(phi_p), 0.0])
    print(f"vp_vec=, {vp_vec} [km/s]")
    # Satellite vector initial position
    vsi_vec = np.array([vsi_mag * math.cos(phi_si), vsi_mag * math.sin(phi_si), 0.0])
    print(f"vsi_vec=, {vsi_vec} [km/s]")

    # Relative satellite-planet velocities
    vspi_vec = vsi_vec - vp_vec
    vspi_mag = np.linalg.norm(vspi_vec)
    print(f"sat-planet relative, vspi_vec=, {vspi_vec} [km/s]")
    print(f"sat-planet vel mag, vspi_mag=, {vspi_mag:.8g} [km/s]")

    # remember v_infinity ~= vsp; atan2() does the quadrant test
    theta_i = math.atan2(vspi_vec[1], vspi_vec[0])  # y / x
    print(f"theta_i= {theta_i:.8g} [rad], {theta_i*180/math.pi:.8g} [deg]")

    # impact parameter, b_impact
    b_impact = (jup_mis_dist) * math.sin(theta_i)
    print(f"impact parameter, b_impact= {b_impact:.8g} [km]")

    # jupiter arrival hyperbola sma (semi-major axis)
    sma_arival = -GM_jup / vspi_mag**2
    print(f"arrival semi-major axis, sma_arrival= {sma_arival:.8g} [km]")

    # jupiter arrival hyperbola eccentricity, ecc_arrival
    ecc_arrival = math.sqrt(1 + (b_impact**2 / sma_arival**2))
    print(f"mars arrival eccentricity, ecc_arrival= {ecc_arrival:.8g}")

    # turning angle, delt
    if jup_mis_dist >= 0:
        delt = 2 * math.asin(1 / ecc_arrival)
    else:
        delt = -2 * math.asin(1 / ecc_arrival)
    print(f"truning angle, delt= {delt:.7g} [rad], {delt*180/math.pi:.7g} [deg]")

    # review theta_f geometry
    theta_f = theta_i + delt
    print(f"theta final, theta_f= {theta_f:.7g} [rad], {theta_f*180/math.pi:.7g} [deg]")

    # Satellite Jupiter relative velocity
    vspf_vec = np.array([vspi_mag * math.cos(theta_f), vspi_mag * math.sin(theta_f), 0])
    print(f"vspf_vec= {vspf_vec} [km/s]")

    # Spacecraft ecliptic velocity
    vsf_vec = vspf_vec + vp_vec
    vsf_mag = np.linalg.norm(vsf_vec)
    print(f"vsf_vec= {vsf_vec} [km/s], vsf_mag= {vsf_mag:.7g} [km/s]")

    # Spacecraft final flight path angle
    phi_sf = math.atan2(vsf_vec[1], vsf_vec[0])
    print(
        f"final satellite flight path angle, phi_sf= {phi_sf:.7g} [rad], {phi_sf*180/math.pi:.7g} [deg]"
    )
    return None


import time


def test_vallado_1(plot_sp=False):
    from vallado_1 import vallado2013

    print(f"\nTest the vallado_1() LambertSolver, with Braeunig parameters:")
    # Solar system constants
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    GM_sun_au = GM_sun_km / (au**3)

    mu = GM_sun_au
    tof = 207 * 24 * 60 * 60  # [s] given, time of flight
    # Ecliptic coordinates
    r1_vec = np.array([0.473265, -0.899215, 0])  # [au]
    r1_mag = np.linalg.norm(r1_vec)
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # [au]
    r2_mag = np.linalg.norm(r2_vec)

    v1_vec, v2_vec, tof_new, numiter, tpi = vallado2013(
        mu,
        r1_vec,
        r2_vec,
        tof,
        M=0,
        prograde=True,
        low_path=True,
        maxiter=100,
        atol=1e-5,
        rtol=1e-7,
        full_output=True,
    )
    # v1, v2, numiter, tpi if full_output is True else only v1, v2.
    v1_mag, v2_mag = [np.linalg.norm(v) for v in [v1_vec, v2_vec]]

    np.set_printoptions(precision=5)  # numpy has spectial print provisions
    print(f"v1_vec= {v1_vec*au} [km/s]")  # note conversion au->km
    print(f"v2_vec= {v2_vec*au} [km/s]")  # note conversion au->km
    print(f"# of iterations {numiter}, time per iteration, tpi= {tpi:.6g} [s]")

    orbit_energy = ((v1_mag**2) / 2) - (mu / r1_mag)
    sma = -mu / (2 * orbit_energy)
    print(f"transfer semimajor axis, sma= {sma:.8g} [au]")

    h_vec = np.cross(r1_vec, v1_vec)
    h_mag = np.linalg.norm(h_vec)
    # print(f"h_vec= {h_vec} [au^2/s], h_mag= {h_mag:.6g} [au^2/s]")

    p = (h_mag**2) / mu
    print(
        f"p= {p:.6g} [au], sma= {sma:.6g} [au], new_tof= {tof_new/(24*3600):.6g} [day]"
    )

    ecc_vec = ((np.cross(v1_vec, h_vec)) / mu) - (r1_vec / r1_mag)
    ecc_mag = np.linalg.norm(ecc_vec)
    print(f"ecc_mag= {ecc_mag:.6g}")

    # compute angle between rx vectors
    delta_nu = get_transfer_angle(r1_vec, r2_vec, prograde=True)  # [rad]
    if plot_sp == True:
        # to see range pf possible orbits, plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker; if sp not defined.
        # note, since sma may go near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(
            r0_mag=r1_mag, r1_mag=r2_mag, delta_nu=delta_nu, sp=p, clip1=True
        )

    return None  # test_vallado_1()


def sma_as_sp(r0_mag, r1_mag, delta_nu):
    """
    Find sma (semi-major axis) as a function of sp (semi-parameter)
    Given:
        positions; r0, r1; angle between positions
    Find:
        sma (semi-major axis)
    Returns:
        sma, ecc
    """
    k = r0_mag * r1_mag * (1 - math.cos(delta_nu))  # BMWS, p.204, eqn 5-42
    l = r0_mag + r1_mag
    m = abs(r0_mag) * abs(r1_mag) * (1 + math.cos(delta_nu))

    # bracket p values for ellipse, p_i & p_ii; BMWS, p.205
    # values > p_ii will be hyperbolic trajectories
    p_i = k / (l + math.sqrt(2 * m))  # BMWS, p.208, eqn 5-52
    p_ii = k / (l - math.sqrt(2 * m))  # BMWS, p.208, eqn 5-53
    print(f"p_i={p_i:.8g}, p_ii={p_ii:.8g}, p_ii-p_i= {(p_ii-p_i):.8g}")
    sp = p_i + (p_ii - p_i) / 2  # choose value near sma minimum
    sp = p_ii * 1.00001  # look for parabola
    print(f"choose semi-parameter, sp= {sp:.8g} [au]")

    # semi-major axis; BMWS, p.204, eqn.5-46
    sma = (m * k * sp) / ((2 * m - l**2) * sp**2 + (2 * k * l * sp - k**2))
    print(f"semi-major axis, sma= {sma} [au]")

    ecc = math.sqrt(1 - (sp / sma))
    print(f"eccentricity, ecc= {ecc:.6g}")
    return sma, ecc


def explore_sp(plot_sp=False):
    """
    Investigate sp (semi-parameter, aka p) values; related to conic sections.

    Began with Braeunig text, gauss p-iteration function, and problems 5.3 & 5.4.
    Note BMWS [2] fig 5-3, p.205.

    Function Input
    ----------
        plot_sp : boolean, True=enable the plot function (sp vs. sma).

    Calculation Parameters
    ----------
        r0=distance from center to departure point
        r1=distance from center to arrival point
        delta_nu=Change in true anomaly [deg]

    Returns
    -------
        none
    References [...] given at file beginning
    """
    import numpy as np

    print(f"\nExplore Semi-Parameter Values:")
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [3] p.1041, tbl.D-3
    au = 149597870.7  # [km/au] Vallado [3] p.1042, tbl.D-5
    r_earth_km = 6378.1363  # [km] earth radius; Vallado [3] p.1041, tbl.D-3

    # Vector magnitude, initial and final position
    r0_vec = np.array([0.473265, -0.899215, 0.0])  # [au] earth(t0) position
    r1_vec = np.array([0.066842, 1.561256, 0.030948])  # [au] mars(t1) position
    r0_mag, r1_mag = [np.linalg.norm(r) for r in [r0_vec, r1_vec]]  # [au]
    print(f"magnitudes: r0_mag= {r0_mag:.8g} [au], r1_mag= {r1_mag:.8g} [au]")
    delta_nu = get_transfer_angle(r0_vec, r1_vec)  # [rad]
    print(
        f"angle between vectors, delta_nu= {delta_nu:.8g} [rad], {delta_nu*180/math.pi} [deg]"
    )
    # parameters to calculate sp limits
    k = r0_mag * r1_mag * (1 - math.cos(delta_nu))  # BMWS [2], p.204, eqn 5-42
    l = r0_mag + r1_mag
    m = abs(r0_mag) * abs(r1_mag) * (1 + math.cos(delta_nu))

    # bracket p values for ellipse, p_i & p_ii; BMWS [2], p.205
    # values > p_ii will be hyperbolic trajectories
    p_i = k / (l + math.sqrt(2 * m))  # BMWS [2], p.208, eqn 5-52
    p_ii = k / (l - math.sqrt(2 * m))  # BMWS [2], p.208, eqn 5-53
    print(f"p_i={p_i:.8g}, p_ii={p_ii:.8g}, p_ii-p_i= {(p_ii-p_i):.8g}")

    sp = p_i + (p_ii - p_i) / 2  # choose value near sp minimum
    print(f"choose semi-parameter, sp= {sp:.8g} [au]")

    # semi-major axis; BMWS [2], p.204, eqn.5-46
    sma = (m * k * sp) / ((2 * m - l**2) * sp**2 + (2 * k * l * sp - k**2))
    print(f"semi-major axis, sma= {sma} [au]")

    ecc = math.sqrt(1 - (sp / sma))
    print(f"eccentricity, ecc= {ecc:.6g}")

    if plot_sp == True:
        # plot_sp=True, to see possible range of orbit parameters plot sp vs. sma
        # note, plot marker at sp is optional; sp=1.0 turns off sp marker.
        # note, since sma may be near-infinate, optional clipping should always be thurned on.
        plot_sp_vs_sma(
            r0_mag=r0_mag, r1_mag=r1_mag, delta_nu=delta_nu, sp=sp, clip1=True
        )

    return None  # explore_sp()


def plot_sp_vs_sma(r0_mag, r1_mag, delta_nu, sp=1.0, clip1=True) -> None:
    """
    Plot sp (semi-parameter) vs. sma (semi-major axis); note BMWS, p.205, fig.5-3.
    Plot range recognizes difference between ellipse and hyperbolic trajectories.
    User may choose to clip calculated sma, since sma may calculate to infinite.
    Plotting may take some fooling with to get the outcome u want.
    Feel free to improve plot lables, etc.

    Input Parameters
    ----------
        r0_mag   : float, focus to initial radius
        r1_mag   : float, focus to 2nd radius
        delta_nu : float, angular distance between r0, r1; focus is central point
        sp       : float, optional, default 1.0 turns off sp marker in plot
        clip1    : boolean, optional, default true; NOTE sma maybe near infinite

    Returns
    -------
        Plot sp vs. sma.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # constants for given r0, r1, angle
    k = r0_mag * r1_mag * (1 - np.cos(delta_nu))  # BMWS [1], p.204, eqn 5-42
    l = r0_mag + r1_mag
    m = abs(r0_mag) * abs(r1_mag) * (1 + np.cos(delta_nu))
    sma = k_l_m_sp(k, l, m, sp)  # semi-major axis

    # bracket p values for ellipse, p_i & p_ii; BMWS [1], p.205
    # values > p_ii, hyperbolic trajectories
    # value at p_ii, parabolic trajectory
    # minimum sp for ellipse; calculated value maybe degenerate...
    sp_i = k / (l + np.sqrt(2 * m))  # BMWS [1], p.208, eqn 5-52
    sp_i_min = sp_i * 1.001  # practical minimum for ellipse
    # maximum sp for ellipse; calculated value is actually a parabola
    sp_ii = k / (l - np.sqrt(2 * m))  # BMWS [1], p.208, eqn 5-53
    sp_ii_max = sp_ii * 1.01  # will show part of hyperbola
    sp_i_mid = (sp_ii - sp_i) / 2

    if sma > 0 and sp > 1.0:  # ellipse
        x = np.linspace(
            sp_i_min, sp_ii_max, 100
        )  # between sp_min & sp_max plot 100 points
    elif sma <= 0 and sp > 1.0:  # parabolic or hyperbolic
        x = np.linspace(sp_ii_max, sp * 10)  # between sp_min & sp_max plot 100 points
    y = k_l_m_sp(k, l, m, x)

    # if clipping enabled, limit y value excursions
    if clip1 == True and sma > 0:
        y = np.clip(y, -sp_i_mid * 200, sp_i_mid * 500)
    else:  # parabolic or hyperbolic plot range near extremes...
        y = np.clip(y, -sp * 5, sp * 5)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    # if optional sp in calling routine, add marker
    if sp != 1.0:  # optional input parameter sp defined
        ecc = math.sqrt(1 - sp / sma)  # BMWS [2] p.21, eqn.1-44
        ax.plot(sp, sma, marker="h", ms=10, mfc="red")
        plt.text(
            sp,
            sma,
            f"sp={sp:.6g}\nsma={sma:.6g}\n ecc={ecc:.6g}",  # move ecc text away from marker
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    text1 = "Ellipse= between positive peaks.\nHyperbola= -sma"
    text2 = (
        f"r0={r0_mag:8g}\nr1={r1_mag:.8g}\ndelta angle={delta_nu*180/np.pi:.6g} [deg]"
    )
    plt.text(
        0.5,
        0.9,
        text2,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.4,
        0.75,
        text1,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.xlabel("sp (semi-parameter, aka p)")
    plt.ylabel("sma (semi-major axis, aka a)")
    plt.grid(True)
    plt.title("SMA vs. SP")
    # not so sure about a graphic fill, below, to graphically denote ellipse
    # ax.fill_between(x, np.max(y), where=y > 0, facecolor="green", alpha=0.5)
    plt.show()
    return None


def k_l_m_sp(k, l, m, sp):
    """
    Calculate sma from input parameters
    """
    # BMWS [2], p.204, eqn 5-42
    sma = (m * k * sp) / ((2 * m - l**2) * sp**2 + (2 * k * l * sp - k**2))
    return sma


# Guides tests & functions.
# Some functions have plot functions that may be enabled
if __name__ == "__main__":
    # test_planets_ecliptic() # verify Braeunig planet positions

    # test_b_p4_28(plot_sp=True)  # Braeunig problem 4.28
    # test_b_p4_29()  # Braeunig problem 4.29
    test_b_gauss_p5_1(plot_sp=True) # Braeunig problem 5.1; one-tangent burn
    test_b_gauss_p5_2(plot_sp=True) # Braeunig problem 5.2; one-tangent burn
    # test_b_gauss_p5_3(plot_sp=True)  # Braeunig problem 5.3; find sp, sma
    # test_b_gauss_p5_4(plot_sp=True) # Braeunig problem 5.4; find v1, v2
    # test_b_gauss_p5_5(plot_sp=True) # Braeunig problem 5.5
    # test_b_gauss_p5_6() # Braeunig problem 5.6
    # test_b_gauss_p5_7() # Braeunig problem 5.7
    # test_b_gauss_p5_8() # Braeunig problem 5.8

    # test_vallado_1(plot_sp=True)  # verified against Braeuning problems 5.3, 5.4...
    # explore_sp(plot_sp=True)  # explore semi-parameter values (aka p)
