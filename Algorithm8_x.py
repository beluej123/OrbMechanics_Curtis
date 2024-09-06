""" 
Chapter 8 collection of algorithms; Interplanetary Trajectories.
2024-08-30, a bunch of clean up remains...

Notes:
----------
    Some of these algorithms endup in functionCollection.py, used by other routines.
    This file is organized with each algorithm as a function.  Testing is called
        at the end of the file...    
    Supporting functions for the test functions below, may be found in other
        files, for example ..., etc. Also note, the test examples are
        collected right after this document block.  However, the example test
        functions are defined/enabled at the end of this file.  Each example
        function is designed to be stand-alone, but, if you use the function
        as stand alone you will need to copy the imports...
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally angles are saved in [rad], distance [km].
    
    # see pdf; http://www.nssc.ac.cn/wxzygx/weixin/201607/P020160718380095698873.pdf    
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

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import functionCollection as funColl
from functionCollection import lambert, planet_elements_and_sv

"""
Appendix D.18, Algorithm 8.2: calculation of the spacecraft trajectory 649
function [planet1, planet2, trajectory] = interplanetary (depart, arrive)

interplanetary(planet1, planet2, trajectory) determines the spacecraft
trajectory from the sphere of influence of planet 1 to that of planet 2 using
algorithm 8.2.
mu  - gravitational parameter of the sun (km^3/s^2)
coe - not required in this procedure
planet_id - planet identifier:
    1 = Mercury
    2 = Venus
    3 = Earth
    4 = Mars
    5 = Jupiter
    6 = Saturn
    7 = Uranus
    8 = Neptune
    9 = Pluto

jd1, jd2- Julian day numbers at departure and arrival
tof- time of flight from planet 1 to planet 2 (s)

Heliocentric coordinates (explore barycentric in the future)
Rp1, Vp1 - planet1 departure state vector (km, km/s)
Rp2, Vp2 - planet2 arrival state vector (km, km/s)
R1, V1 - spacecraft departure state vector
R2, V2 - spacecraft arrival state vector

User functions required:
    planet_elements_and_sv, lambert
"""


# Appendix 8.1
def rv_from_date(planet_id, date_UT, mu):
    """
    Find r_vec, v_vec, given the planet and date/time.
    This is algorithm 8.1 from Curtis, pp.471; also see Curtis example 8.7.

    Since orbital elements and the Julian date are computed, they are available as output.
    [r_vec, v_vec, coe_t0, jd_t0] = rv_from_date().

     Notes:
    ----------
        Uses Curtis, pp.471, algorithm 8.1.  Note Curtis p.277, example 5.4, Sideral time.
        Note curtis_ex4_7()
        Also see Vallado functions: pp. 296, planetRV() (algotithm 33),
            cov2rv() (algorithm 11), et.al
        Orbital elements tables kept in functionCollection.py


        Orbital elements in this function:
            sma   = [km] semi-major axis (aka a)
            ecc   = [-] eccentricity
            incl  = [deg] inclination angle; to the ecliptic
            RAAN  = [deg] right ascension of ascending node (aka capital W)
            w_bar = [deg] longitude of periapsis (NOT arguement of periapsis, w)
                    Note, w_bar = w + RAAN
            L     = [deg] mean longitude (NOT mean anomaly, M)
                    Note, L = w_bar + M
    """
    # mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    # mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    # mu_mars_km = 4.305e4  # [km^3/s^2], Vallado p.1041, tbl.D-3

    # given date/time for t0, find Julian date
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT

    # appendix steps 1, 2, 3, of Curtis p.471-472, part of algorithm 8.1.
    coe_t0, jd_t0 = funColl.coe_from_date(planet_id, date_UT)
    # coe_t0, elements names= ["sma", "ecc", "incl", "RAAN", "w_hat", "L_"]
    sma, ecc, incl, RAAN, w_hat, L_ = coe_t0

    # appendix: Curtis, p.473, step 4
    h_mag = math.sqrt(mu * sma * (1 - ecc**2))

    # Earth: Curtis, p.473, step 5
    w_ = (w_hat - RAAN) % (2 * math.pi)  # [rad] limit value 0->2*pi
    M_ = (L_ - w_hat) % (2 * math.pi)  # [rad] limit value 0->2*pi

    # appendix: Curtis, p.474, step 6; find eccentric angle/anomaly
    E_ = funColl.solve_for_E(Me=M_, ecc=ecc)  # [rad]

    # appendix: Curtis, p.474, step 7; find true angle/anomaly
    TA = 2 * math.atan(math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E_ / 2))  # [rad]
    TA_ = TA % (2 * math.pi)  # [rad] limit value 0->2*pi

    # appendix: Curtis, p.474, step 8; find r_vec, v_vec
    # note Curtis, pp.232, example 4.7 & p.231, algorithm 4.5
    r_vec, v_vec = funColl.sv_from_coe(
        h=h_mag, ecc=ecc, RA=RAAN, incl=incl, w=w_, TA=TA_, mu=mu
    )

    return r_vec, v_vec, coe_t0, jd_t0


# Appendix 8.2
def interplanetary(depart, arrive):
    # mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5

    planet_id = depart[0]
    year = depart[1]
    month = depart[2]
    day = depart[3]
    hour = depart[4]
    minute = depart[5]
    second = depart[6]

    # ...Use Algorithm 8.1 to obtain planet 1's state vector (don't
    # ...need its orbital elements ["dum"]):
    dum, Rp1, Vp1, jd1 = planet_elements_and_sv(
        planet_id=planet_id,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        mu=mu_sun_km,
    )

    planet_id = arrive[0]
    year = arrive[1]
    month = arrive[2]
    day = arrive[3]
    hour = arrive[4]
    minute = arrive[5]
    second = arrive[6]

    # ...Likewise use Algorithm 8.1 to obtain planet 2's state vector:
    dum, Rp2, Vp2, jd2 = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second, mu=mu_sun_km
    )

    tof = (jd2 - jd1) * 24 * 3600

    # ...Patched conic assumption:
    R1 = Rp1
    R2 = Rp2

    # ...Use Algorithm 5.2 to find the spacecraft's velocity at
    #    departure and arrival, assuming a prograde trajectory:
    V1, V2 = lambert(R1=R1, R2=R2, tof=tof, mu=mu_sun_km, string="pro")

    planet1 = [Rp1, Vp1, jd1]
    planet2 = [Rp2, Vp2, jd2]
    trajectory = [V1, V2]

    return planet1, planet2, trajectory


def test_rv_from_date():

    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    # mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    # mu_mars_km = 4.305e4  # [km^3/s^2], Vallado p.1041, tbl.D-3

    # planet position/velocity at date/time
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT
    date_UT = [2003, 8, 27, 12, 0, 0]  # [UT] date/time python list
    planet_id = 3  # earth
    r_vec, v_vec, coe_t0, jd_t0 = rv_from_date(
        planet_id=planet_id, date_UT=date_UT, mu=mu_sun_km
    )
    print(f"Julian date, t0, jd_t0= {jd_t0}")
    print(f"planet position vector, r_vec= {r_vec}")
    print(f"planet velocity vector,v_vec= {v_vec}")

    print(f"planet orbital elements:\ncoe_t0=\n{coe_t0}")
    print(f"     sma,      ecc,      incl,      RAAN,     w_hat,    L_")
    return


def test_interplanetary():
    print(f"\nTest Curtis algorithm 8.1, interplanetary()")
    # note, pp.476, example 8.8

    # given date/time for t0, find Julian date
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT
    date_UT = [2003, 8, 27, 12, 0, 0]  # [UT] date/time python list
    yr, mo, d, hr, minute, sec = date_UT

    planet_id = 3  # earth
    interplanetary(depart, arrive)

    return None


# use the following to test/examine functions
if __name__ == "__main__":

    test_rv_from_date()  # test curtis algorithm 8.1
    # test_interplanetary()  #
