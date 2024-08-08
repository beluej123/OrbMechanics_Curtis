# Braeunig Functions
import math

import numpy as np


def Julian_date(D, M, Y, UT):
    """
    # Initially needed for problem 5.3 for planet positions, given a date
    Code from
    Also note http://www.braeunig.us/space/index.htm
    convert day, month, year, and universal time into Julian date
    args: D - day
          M - month
          Y - year
          UT - universal time

    returns: Julian date
    """
    if M <= 2:
        y = Y - 1
        m = M + 12
    else:
        y = Y
        m = M

    if Y < 1582:
        B = -2
    elif Y == 1582:
        if M < 10:
            B = -2
        elif M == 10:
            if D <= 4:
                B = -2
            else:
                B = math.floor(y / 400) - math.floor(y / 100)
        else:
            B = math.floor(y / 400) - math.floor(y / 100)
    else:
        B = math.floor(y / 400) - math.floor(y / 100)

    return (
        math.floor(365.25 * y)
        + math.floor(30.6001 * (m + 1))
        + B
        + 1720996.5
        + D
        + UT / 24
    )


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


import math

from numpy import dot
from numpy.linalg import norm


def angle_between(a, b):
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


def b_gauss(r1: float, r2: float, delta_nu: float, tof: float, GM: float):
    """Braeunig's Gauss Orbit Solution. P-iteration method.
    Related to problem 5.3 & 5.4.

    2024-08-04 function not finished.
    Commented out print statements are for debugging.

    Parameters
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

    Returns
    -------
    _type_
        _description_

    Notes
    -----
    The Algorithm maybe singular for transfer angles of 180 degrees.
    Not tested performance for really small angles.
    From http://www.braeunig.us/space/interpl.htm

    References
    ----------
    [1] Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W.
    Fundamentals of Astrodynamics. Dover Publications Inc.
    copyright 2020, 2nd ed.
    """

    # convert input degrees to radians for trig calculations
    delta_nu1 = delta_nu * (math.pi / 180)
    k = r1 * r2 * (1 - math.cos(delta_nu1))
    l = r1 + r2
    m = abs(r1) * abs(r2) * (1 + math.cos(delta_nu1))
    # print(f"k={k:.8g}, l={l:.8g}, m={m:.8g}")

    # BMWS, p.205, p_i & p_ii are bracketing values for p
    p_i = k / (l + math.sqrt(2 * m))  # BMWS, eqn 5-47
    p_ii = k / (l - math.sqrt(2 * m))  # BMWS, eqn 5-48
    # print(f"p_i={p_i:.8g}, p_ii={p_ii:.8g}, p_i-p_ii= {(p_i-p_ii):.8g}")

    # TODO figure out how to select value for p
    p = 1.2  # [au] initial p assignment

    # initial 1, p value *******************
    p_1, sma_1, tof_1, f, g, f_dot, g_dot = gauss_cal_mid(
        r1=r1, r2=r2, GM=GM, delta_nu1=delta_nu1, m=m, k=k, p=p, l=l
    )
    # print(f"p_1={p_1:.8g}, sma_1={sma_1:.8g}, tof_1={(tof_1/(24*3600)):.8g}")

    p = 1.3

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
    sma = (m * k * p) / (
        (2 * m - l**2) * p**2 + (2 * k * l * p - k**2)
    )  # semi-major axis
    # print(f"sma={sma:.8g} [au]")

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
    # print(f"f= {f:.8g}, g= {g:.8g}, f_dot= {f_dot:.8g}, g_dot= {g_dot:.8g}")
    # print(f"time of flight-1, tof={tof:.8g} [s]")
    # print(f"time of flight-1, tof={tof/(24*3600):.8g} [day]")

    return (p, sma, tof, f, g, f_dot, g_dot)


def test_b_gauss_p5_3():
    # test Braeunig problem 5.3
    # Earth->Mars mission launch 2020-7-20, 0:00 UT, planned time of flight 207 days.
    # Earth's at departure is 0.473265X - 0.899215Y AU.
    # Mars' at intercept is 0.066842X + 1.561256Y + 0.030948Z AU.
    # Calculate the parameter and semi-major axis of the transfer orbit.

    # Use ecliptic coordinates.
    print(f"test Braeunig problem 5.3")
    # Vector magnitude, initial and final position
    # r1_vec = np.array([0.4633103, -0.8948005,  0.0001092])  # earth(t0) position [AU]
    # r2_vec = np.array([0.0677917,  1.5666497,  0.0309914])  # mars(t1) position [AU]
    r1_vec = np.array([0.473265, -0.899215, 0.0])  # earth(t0) position [AU]
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # mars(t1) position [AU]
    r1, r2 = [np.linalg.norm(r) for r in [r1_vec, r2_vec]]
    print(f"magnitudes: r1= {r1:.8g} [au], r2= {r2:.8g}")

    GM = 3.964016e-14  # [au^3/s^2]
    delta_nu = 149.770967  # [deg]
    tof = 207 * 24 * 60 * 60  # [s]

    p, sma, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r1, r2=r2, delta_nu=delta_nu, tof=tof, GM=GM
    )
    print(f"p= {p:.8g} [au], sma= {sma:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]")

    return None


def test_b_gauss_p5_4():
    # test Braeunig problem 5.4 (include 5.3).
    # For Earth->Mars mission of problem 3.3,  calculate departure and intecept velocity vectors.

    # Use ecliptic coordinates.
    print(f"test Braeunig problem 5.4:")
    # Vector magnitude, initial and final position
    # r1_vec = np.array([0.4633103, -0.8948005,  0.0001092])  # earth(t0) position [AU]
    # r2_vec = np.array([0.0677917,  1.5666497,  0.0309914])  # mars(t1) position [AU]
    r1_vec = np.array([0.473265, -0.899215, 0.0])  # earth(t0) position [AU]
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # mars(t1) position [AU]
    r1, r2 = [np.linalg.norm(r) for r in [r1_vec, r2_vec]]
    # print(f"magnitudes: r1= {r1:.8g} [au], r2= {r2:.8g}")

    GM = 3.964016e-14  # [au^3/s^2]
    delta_nu = 149.770967  # [deg]
    tof = 207 * 24 * 60 * 60  # [s]

    p, sma, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r1, r2=r2, delta_nu=delta_nu, tof=tof, GM=GM
    )
    print(f"p= {p:.8g} [au], sma= {sma:.8g} [au], tof= {(tof/(24*3600)):.8g} [day]")
    print(f"f= {f:.8g}, g= {g:.8g}, f_dot= {f_dot:.8g}, g_dot= {g_dot:.8g}")

    au_ = 149.597870e6  # convert: [au] to [km]
    v1_vec = (r2_vec - f * r1_vec) / g  # [au/s]
    print(f"v1_vec= {v1_vec*au_} [km/s]")  # convert [au] to [km]
    v2_vec = f_dot * r1_vec + g_dot * v1_vec
    print(f"v2_vec= {v2_vec*au_} [km/s]")  # convert [au] to [km]

    return None


def test_b_gauss_p5_5():
    # test Braeunig problem 5.5 (includes 5.3, 5.4).
    # For Earth->Mars mission of problem 3.3,  calculate transfer orbit orbital elements.
    # For this problem use r1_vec & v1_vec; can also use r2_vec & v2_vec.

    # Use ecliptic coordinates.
    print(f"test Braeunig problem 5.5:")
    # Vector magnitude, initial and final position
    # r1_vec = np.array([0.4633103, -0.8948005,  0.0001092])  # earth(t0) position [AU]
    # r2_vec = np.array([0.0677917,  1.5666497,  0.0309914])  # mars(t1) position [AU]
    r1_vec = np.array([0.473265, -0.899215, 0.0])  # earth(t0) position [AU]
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # mars(t1) position [AU]
    r1, r2 = [np.linalg.norm(r) for r in [r1_vec, r2_vec]]
    # print(f"magnitudes: r1= {r1:.8g} [au], r2= {r2:.8g}")

    GM = 3.964016e-14  # [au^3/s^2] sun
    km_au = 149.597870e6  # [km/au]; convert: [au] to [km]
    delta_nu = 149.770967  # [deg]
    tof = 207 * 24 * 60 * 60  # [s]

    p, sma, tof, f, g, f_dot, g_dot = b_gauss(
        r1=r1, r2=r2, delta_nu=delta_nu, tof=tof, GM=GM
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

    n1 = np.cross([0.0, 0.0, 1.0], h1_vec)  # z cross h_vec
    print(f"normal to h1, n1= {n1}")

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

    return None


import time

from gauss_1 import (
    _gauss_first_equation,
    _gauss_second_equation,
    _get_s,
    _get_w,
    assert_parameters_are_valid,
    gauss1809,
    get_transfer_angle,
)


def test_gauss_1():
    # Ecliptic coordinates
    GM = 3.964016e-14  # [au^3/s^2]
    r1 = np.array([0.473265, -0.899215, 0])
    r2 = np.array([0.066842, 1.561256, 0.030948])
    print(f"r1 mag, {np.linalg.norm(r1)}")

    tof = 207 * 24 * 60 * 60  # [s]
    # **********************
    M = 0
    prograde = True
    low_path = True
    maxiter = 250
    atol = 1e-5
    rtol = 1e-7
    full_output = False
    # **********************

    print(assert_parameters_are_valid(r1=r1, r2=r2, tof=tof, M=0, mu=GM))
    mu = GM
    r1_norm, r2_norm = [norm(r) for r in [r1, r2]]
    prograde = True
    dtheta = get_transfer_angle(r1, r2, prograde)
    # Compute the s and w constants
    s = _get_s(r1_norm, r2_norm, dtheta)
    w = _get_w(mu, tof, r1_norm, r2_norm, dtheta)

    # Initial guess formulation is of the arbitrary type
    y0 = 1.00

    # The iterative procedure can start now
    tic = time.perf_counter()
    for numiter in range(1, maxiter + 1):
        # Compute the value of the free-parameter
        x = _gauss_first_equation(y0, s, w)

        # Evaluate the new value of the dependent variable
        y = _gauss_second_equation(x, s)

        # Check the convergence of the method
        if np.abs(y - y0) <= atol:
            tac = time.perf_counter()
            tpi = (tac - tic) / numiter
            break
        else:
            # The new initial guess is the previously computed y value
            y0 = y
    else:
        raise ValueError("Exceeded maximum number of iterations.")

    # v1, v2=gauss1809(r1=r1, r2=r2, tof=tof, mu=GM, M=0, prograde=True, low_path=True, maxiter=250, atol=1e-5, rtol=1e-7, full_output=False)
    # print(f"{v1}, {v2}")
    return None


def main() -> None:
    pass  # placeholder


# Main code. Functions and class methods are called from main.
if __name__ == "__main__":
    # test_planets_ecliptic()
    # test_b_gauss_p5_3() # Braeunig problem 5.3
    # test_b_gauss_p5_4()  # Braeunig problem 5.4
    test_b_gauss_p5_5()  # Braeunig problem 5.5
    # test_gauss_1()

    main()  # placeholder function
