"""
Test file for general functions file
"""

import math

import numpy as np
import pint

import func_gen as fg
from constants_1 import AU_, CENT, GM_SUN_AU, RAD

ureg = pint.UnitRegistry()


def test_coe_from_date():
    """
    Code began with Curtis example 8.7.
    Note, coe_from_date() returns in units [km] & [rad]
    """
    planet_id = 3  # earth
    date_UT = [2003, 8, 27, 12, 0, 0]  # [UT]
    coe_t0, jd_t0 = fg.coe_from_date(planet_id, date_UT)
    # coe_elements_names= ["sma", "ecc", "incl", "RAAN", "w_hat", "L"]
    sma, ecc, incl, RAAN, w_hat, L = coe_t0
    incl_deg = incl * 180 / math.pi
    RAAN_deg = RAAN * 180 / math.pi
    w_hat_deg = w_hat * 180 / math.pi
    L_deg = L * 180 / math.pi

    print(f"Julian date, jd_t0= {jd_t0}")
    print(
        f"sma= {sma:.8g} [km], "
        f"ecc= {ecc:.8g}, "
        f"\nincl_deg= {incl_deg:.8g} [deg], "
        f"RAAN_deg= {RAAN_deg:.6g} [deg], "
        f"w_hat_deg= {w_hat_deg:.6g} [deg], "
        f"L_deg= {L_deg:.6g} [deg]"
    )


def test_sv_from_coe():
    """
    Curtis example 4.7.
    h, ecc, RA, incl, w, TA
    """
    print("\nTest Curtis function, sv_from_coe():")
    deg2rad = math.pi / 180
    mu_earth_km = 398600  # [km^3/s^2]
    h = 80000  # [km^2/s]
    ecc = 1.4

    RA_rad, incl_rad, w_rad, TA_rad = [
        40 * deg2rad,
        30 * deg2rad,
        60 * deg2rad,
        30 * deg2rad,
    ]  # [rad]
    r1_vec, v1_vec = fg.sv_from_coe(
        h=h,
        ecc=ecc,
        RA_rad=RA_rad,
        incl_rad=incl_rad,
        w_rad=w_rad,
        TA_rad=TA_rad,
        mu=mu_earth_km,
    )
    print(f"position, r1= {r1_vec}")
    print(f"velocity, v1= {v1_vec}")


def test_solve4E():
    """
    Useing Curtis [3] solve_for_E() to cross-check Vallado [4], example 5-5, pp.304.
    """
    rad2deg = 180 / math.pi
    Me = -150.443142 * math.pi / 180
    ecc = 0.048486
    E_rad = fg.solve_for_E(Me=Me, ecc=ecc)
    E_deg = E_rad * rad2deg
    print(f"E_, = {E_rad} [rad], {E_deg} [deg]")

    # below eliminates numerical problems near +- pi
    beta = ecc / (1 + np.sqrt(1 - ecc**2))  # quadrant checks automatically
    TA_rad = E_rad + 2 * np.arctan((beta * np.sin(E_rad)) / (1 - beta * np.cos(E_rad)))
    TA_deg = TA_rad * rad2deg
    print(f"TA, = {TA_rad} [rad], {TA_deg} [deg]")
    return None


def test_planetary_elements():
    """
    Compare data sets; Curtis [3] tbl 8.1 with JPL Horizons tbl 1.
        JPL Horizons tbl 1, https://ssd.jpl.nasa.gov/planets/approx_pos.html
        NOTE the Horizons table lists element IN A DIFFERENT ORDER THAN Curtis [3] !!
        NOTE Curtis table equatorial, Horizons table is ecliptic !!
    Conclusion; reasonable correlation between data sets. But note difference
        in raan values for earth.
    """
    np.set_printoptions(precision=4)  # numpy, set vector printing size
    # earth: Curtis and JPL Horizons data sets
    planet_id = 3  # earth
    # orbital elements tables kept in functions.py
    # d_set=1 means Curtis [3] table 8.1; d_set=0 means JPL Horizons Table 1
    e_c_coe_equ, e_c_rates_equ = fg.planetary_elements(planet_id, d_set=1)
    # JPL data used ecliptic referenced orbital elements; convert to match Curtis
    e_j_coe_ecl, e_j_rates_ecl = fg.planetary_elements(planet_id, d_set=0)
    # convert JPL coe and rates from ecliptic to equatorial
    # e_j_coe_equ = ecliptic_to_equatorial(ecl_elements=e_j_coe_ecl)
    # e_j_rates_equ = ecliptic_to_equatorial(ecl_elements=e_j_rates_ecl)

    print("\nCurtis Orbital Elements (sma, ecc, i, Ω, ω, M):")
    for val in e_c_coe_equ:
        print(f"{val:0.6g~}, ", end="")
    # convert degrees to radians
    for cnt, val in enumerate(e_j_coe_ecl):
        if cnt >= 2:
            e_c_coe_equ[cnt] = val.to(RAD)
    print("")
    for val in e_c_coe_equ:
        print(f"{val:0.6g~}, ", end="")

    print("\nCurtis Elements Rate Changes")
    for val in e_c_rates_equ:
        print(f"{val:0.6g~}, ", end="")
    # convert degrees/cy to radians/cy
    for cnt, val in enumerate(e_j_rates_ecl):
        if cnt >= 2:
            e_c_rates_equ[cnt] = val.to(RAD / CENT)
    print("")
    for val in e_c_rates_equ:
        print(f"{val:0.6g~}, ", end="")

    print("\n\nJPL Orbital Elements (sma, ecc, i, Ω, ω, M):")
    for val in e_j_coe_ecl:
        print(f"{val:0.6g~}, ", end="")
    # convert degrees to radians
    for cnt, val in enumerate(e_j_coe_ecl):
        if cnt >= 2:
            e_j_coe_ecl[cnt] = val.to(RAD)
    print("")
    for val in e_j_coe_ecl:
        print(f"{val:0.6g~}, ", end="")

    print("\nJPL Elements Rate Changes")
    for val in e_j_rates_ecl:
        print(f"{val:0.6g~}, ", end="")
    # convert degrees/cy to radians/cy
    for cnt, val in enumerate(e_j_rates_ecl):
        if cnt >= 2:
            e_j_rates_ecl[cnt] = val.to(RAD / CENT)
    print("")
    for val in e_j_rates_ecl:
        print(f"{val:0.6g~}, ", end="")

    # get julian date for planetary elements
    # yr, mo, day, hr, min, sec = 2003, 8, 27, 12, 0, 0  # [UT]
    # t0_jd = g_date2jd(yr=yr, mo=mo, d=day, hr=hr, minute=min, sec=sec)
    # t0_jd_cent = (t0_jd - 2451545.0) / 36525  # julian centuries since j2000
    # t0_jd_cent *= CENT # set units for julian date

    # Curtis [3], p.473, step 3
    # apply century rates of change to earth coe rates
    #   python list multiply; for Curtis data set
    # t0_c_rates = [e_c_rates[x] * t0_jd_cent for x in range(len(e_c_rates))]
    # # python list add; for Curtis data set
    # t0_c_coe = [e_c_coe[x] + t0_c_rates[x] for x in range(len(e_c_coe))]
    # #   python list multiply; for JPL data set
    # t0_j_rates = [e_j_rates[x] * t0_jd_cent for x in range(len(e_j_rates))]
    # # python list add; for JPL data set
    # t0_j_coe = [e_j_coe[x] + t0_j_rates[x] for x in range(len(e_j_coe))]

    # # coe elements= ["sma[km]", "ecc", "incl[deg]", "raan[deg]", "w_hat[deg]", "L_[deg]"]
    # # inclination [deg] values need to be between +- 180[deg]
    # # must be a better method for below
    # # Curtis data set
    # t0_c_coe[2] = (t0_c_coe[2] + 180) % 360 - 180  # -180 < incl < 180
    # t0_c_coe[3] = t0_c_coe[3] % 360  # note modulo arithmetic, %
    # t0_c_coe[4] = t0_c_coe[4] % 360  # note modulo arithmetic, %
    # t0_c_coe[5] = t0_c_coe[5] % 360  # note modulo arithmetic, %
    # # JPL data set
    # t0_j_coe[2] = (t0_j_coe[2] + 180) % 360 - 180  # -180 < incl < 180
    # t0_j_coe[3] = t0_j_coe[3] % 360  # note modulo arithmetic, %
    # t0_j_coe[4] = t0_j_coe[4] % 360  # note modulo arithmetic, %
    # t0_j_coe[5] = t0_j_coe[5] % 360  # note modulo arithmetic, %

    # print("\nEarth orbital elements at t0 (rounded 5-places):")
    # print("Below compare Curtis [3] and JPL Horizons data sets:")
    # print(f"t0= {yr}-{mo}-{day} {hr}:{min}:{sec}")
    # coe_elements = ["sma[km]", "ecc", "incl[deg]", "raan[deg]", "w_hat[deg]", "L_[deg]"]
    # print(f"coe order:{coe_elements}")
    # print(f"e_c_coe= {[round(elem,5) for elem in t0_c_coe]} [km] & [deg]")
    # print(f"e_j_coe= {[round(elem,5) for elem in t0_j_coe]} [km] & [deg]")

    # for cnt, coe_val in enumerate(t0_c_coe):
    #     print(f"{cnt}, e_c_coe= {coe_val:0.5f~}")


def test_hohmann_table():
    """Define planet orbital radii (semi-major axis) in AU"""
    print("Generate Hohmann transfer table:")
    # assume circular orbit; planet radii semi-major axis [AU]
    planets = {
        "Mercury": 0.387,
        "Venus": 0.723,
        "Earth": 1.000,
        "Mars": 1.524,
        "Jupiter": 5.203,
        "Saturn": 9.537,
        "Uranus": 19.191,
        "Neptune": 30.069,
    }
    # Sun gravitational parameter [AU^3/day^2]
    # mu_sun = 2.959e-4
    print(f"GM_SUN AU^3/s^2 = {GM_SUN_AU}")  # from constants_1.py
    mu_sun = GM_SUN_AU.magnitude  # strip off units-aware

    hohmann_table = fg.hohmann_table(planets, mu_sun)
    print(hohmann_table)


def test_hohmann_transfer():
    """Generate Hohmann transfer parameters."""
    print("Calculate Hohmann transfer parameters:")
    mu = GM_SUN_AU.magnitude  # strip off units-aware
    # earth -> venus
    hohmann = fg.hohmann_transfer(r1=1.0, r2=0.723, mu=mu)
    print(hohmann)


def main():
    """just a placeholder to help with editor navigation:--)"""
    return


# use the following to test/examine functions
if __name__ == "__main__":
    test_hohmann_table()  # from google search
    # test_hohmann_transfer()  # from google search
    # test_planetary_elements()  # compare Curtis [3] tbl 8.1 & JPL Horizons
    # test_coe_from_date()  # part of Curtis, algorithm 8.1
    # test_sv_from_coe()  # coe2rv
    # test_solve4E()  # solve_for_E
    # sun_rise_set1()  # calculate sunrise sunset, given location
    main()  # do nothing :--)
