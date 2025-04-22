""" 
Chapter 8 collection of algorithms.
    Appendix D.18, Algorithm 8.2: calculation of the spacecraft trajectory.
2024-08-30, a bunch of clean up remains...


Notes:
----------
    Some of these algorithms endup in functions.py, used by other routines.
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
References: (see references.py for references list)
"""

import math

import functions as funColl
from constants import GM_SUN
from functions import lambert, planet_elements_and_sv


# Appendix 8.1
def rv_from_date(planet_id, date_UT, mu):
    """
    Find r_vec, v_vec, given the planet and date/time.
    Algorithm 8.1, Curtis [3], pp.471; also Curtis [3] example 8.7.

    Since orbital elements and the Julian date are computed, they are available as output.
    [r_vec, v_vec, coe_t0, jd_t0] = rv_from_date().

     Notes:
    ----------
        Uses Curtis, pp.471, algorithm 8.1.  Note Curtis p.277, example 5.4, Sideral time.
        Note curtis_ex4_7()
        Also see Vallado [2] functions: pp. 296, planetRV() (algotithm 33),
            cov2rv() (algorithm 11), et.al
        Orbital elements tables kept in functions.py
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
    # MU_SUN_KM = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    # mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    # mu_mars_km = 4.305e4  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3

    # given date/time for t0, find Julian date
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT

    # appendix steps 1, 2, 3, of Curtis p.471-472, part of algorithm 8.1.
    coe_t0, jd_t0 = funColl.coe_from_date(planet_id=planet_id, date_UT=date_UT)
    # coe_t0; element names ["sma[km]", "ecc", "incl[rad]", "RAAN[rad]", "w_hat[rad]", "L_[rad]"]
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
        h=h_mag, ecc=ecc, RA_rad=RAAN, incl_rad=incl, w_rad=w_, TA_rad=TA_, mu=mu
    )

    return r_vec, v_vec, coe_t0, jd_t0


# Appendix 8.2 interplanetary; note there is a version ss_transfer()
def interplanetary(depart, arrive):
    """
    algorithm 8.2.
    mu  - sun gravitational parameter [km^3/s^2]
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
        mu=GM_SUN,
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
        planet_id, year, month, day, hour, minute, second, mu=GM_SUN
    )

    tof = (jd2 - jd1) * 24 * 3600

    # ...Patched conic assumption:
    R1 = Rp1
    R2 = Rp2

    # ...Use Algorithm 5.2 to find the spacecraft's velocity at
    #    departure and arrival, assuming a prograde trajectory:
    V1, V2 = lambert(R1=R1, R2=R2, tof=tof, mu=GM_SUN, prograde=True)

    planet1 = [Rp1, Vp1, jd1]
    planet2 = [Rp2, Vp2, jd2]
    trajectory = [V1, V2]
    return planet1, planet2, trajectory


# Appendix 8.2 interplanetary renamed to ss_transfer
def ss_transfer(depart, arrive):
    """
    algorithm 8.2. units aware version of interplanetary
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
        mu=GM_SUN,
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
        planet_id, year, month, day, hour, minute, second, mu=GM_SUN
    )

    tof = (jd2 - jd1) * 24 * 3600

    # ...Patched conic assumption:
    R1 = Rp1
    R2 = Rp2

    # ...Use Algorithm 5.2 to find the spacecraft's velocity at
    #    departure and arrival, assuming a prograde trajectory:
    V1, V2 = lambert(R1=R1, R2=R2, tof=tof, mu=GM_SUN, prograde=True)

    planet1 = [Rp1, Vp1, jd1]
    planet2 = [Rp2, Vp2, jd2]
    trajectory = [V1, V2]

    return planet1, planet2, trajectory


def time_to_soi(
    object_position, object_velocity, planet_position, planet_velocity, soi_radius
):
    """
    TODO: clean-up; adjust, to calculate time to exit soi
    Calculates the time to enter a planet's soi (sphere of influence).
    From google search: "time to planet sphere of influence python"

    Args:
    object_position: A tuple or list representing the object's position (x, y, z).
    object_velocity: A tuple or list representing the object's velocity (vx, vy, vz).
    planet_position: A tuple or list representing the planet's position (x, y, z).
    planet_velocity: A tuple or list representing the planet's velocity (vx, vy, vz).
    soi_radius: The radius of the planet's sphere of influence.

    Returns:
    The time in seconds to enter the sphere of influence, or None if the object is already inside.
    """


#     # Calculate the position vector from the object to the planet
#     pos_vector = (planet_position[0] - object_position[0], planet_position[1] - object_position[1], planet_position[2] - object_position[2])

#     # Calculate the relative velocity vector
#     rel_vel_vector = (object_velocity[0] - planet_velocity[0], object_velocity[1] - planet_velocity[1], object_velocity[2] - planet_velocity[2])

#     # Calculate the distance from the object to the planet's sphere of influence
#     distance = math.sqrt(sum(x**2 for x in pos_vector))

#     # Check if the object is already inside the sphere of influence
#     if distance <= soi_radius:
#         return None  # Or any appropriate value indicating it's already inside

#     # Calculate the relative velocity magnitude
#     rel_vel_magnitude = math.sqrt(sum(x**2 for x in rel_vel_vector))

#     # Calculate the time to enter the sphere of influence
#     time = (distance - soi_radius) / rel_vel_magnitude

#     return time

# # Example usage (replace with your actual values)
# object_position = (1000, 2000, 3000)
# object_velocity = (10, 20, 30)
# planet_position = (5000, 6000, 7000)
# planet_velocity = (5, 10, 15)
# soi_radius = 1000

# time = time_to_soi(object_position, object_velocity, planet_position, planet_velocity, soi_radius)

# if time is not None:
#     print(f"Time to enter the sphere of influence: {time:.2f} seconds")
# else:
#     print("Object is already inside the sphere of influence.")


def test_rv_from_date():
    """single line description"""

    MU_SUN_KM = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    # mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
    # mu_mars_km = 4.305e4  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3

    # planet position/velocity at date/time
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT
    date_UT = [2003, 8, 27, 12, 0, 0]  # [UT] date/time python list
    planet_id = 3  # earth
    r_vec, v_vec, coe_t0, jd_t0 = rv_from_date(
        planet_id=planet_id, date_UT=date_UT, mu=MU_SUN_KM
    )
    print(f"Julian date, t0, jd_t0= {jd_t0}")
    print(f"planet position vector, r_vec= {r_vec}")
    print(f"planet velocity vector,v_vec= {v_vec}")

    print(f"planet orbital elements:\ncoe_t0=\n{coe_t0}")
    print("     sma,      ecc,      incl,      RAAN,     w_hat,    L_")
    return


def test_interplanetary():
    """single line description"""
    print("\nTest Curtis algorithm 8.1, interplanetary()")
    # note, pp.476, example 8.8

    # given date/time for t0, find Julian date
    # yr, mo, d, hr, minute, sec = 2003, 8, 27, 12, 0, 0  # UT
    date_UT = [2003, 8, 27, 12, 0, 0]  # [UT] date/time python list
    yr, mo, d, hr, minute, sec = date_UT

    planet_id = 3  # earth
    # interplanetary(depart, arrive)

    return None


# use the following to test/examine functions
if __name__ == "__main__":

    test_rv_from_date()  # test curtis algorithm 8.1
    # test_interplanetary()  #
    # test_interplanetary()  #
