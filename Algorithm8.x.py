#  based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#  by Howard D. Curtis
# see pdf; http://www.nssc.ac.cn/wxzygx/weixin/201607/P020160718380095698873.pdf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def interplanetary(planet1, planet2, trajectory):
    """
    Appendix D.18, Algorithm 8.2: calculation of the spacecraft trajectory 649
    function [planet1, planet2, trajectory] = interplanetary (depart, arrive)

    This function determines the spacecraft trajectory from the sphere of
    influence of planet 1 to that of planet 2 using Algorithm 8.2.
    mu  - gravitational parameter of the sun (kmˆ3/sˆ2)
    dum - a dummy vector not required in this procedure
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

    departure (km, km/s)- heliocentric state vector of spacecraft at
    arrival (km, km/s)- [planet_id, year, month, day, hour, minute,
    second] at departure- [planet_id, year, month, day, hour, minute,
    second] at arrival- [Rp1, Vp1, jd1]- [Rp2, Vp2, jd2]
    trajectory- [V1, V2]

    User functions required: planet_elements_and_sv, lambert
    """
    """
    Use Algorithm 8.1 to obtain planet1 and planet2 state vector (don't need its orbital elements [''dum'']):
    [dum, Rp1, Vp1, jd1] = planet_elements_and_sv
    (planet_id, year, month, day, hour, minute, second);
    """


# the following is an on-line matlab -> python converter
# https://www.codeconvert.ai/matlab-to-python-converter
def interplanetary(depart, arrive):
    global mu

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
        planet_id, year, month, day, hour, minute, second
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
        planet_id, year, month, day, hour, minute, second
    )

    tof = (jd2 - jd1) * 24 * 3600

    # ...Patched conic assumption:
    R1 = Rp1
    R2 = Rp2

    # ...Use Algorithm 5.2 to find the spacecraft's velocity at
    #    departure and arrival, assuming a prograde trajectory:
    V1, V2 = lambert(R1, R2, tof, "pro")

    planet1 = [Rp1, Vp1, jd1]
    planet2 = [Rp2, Vp2, jd2]
    trajectory = [V1, V2]

    return planet1, planet2, trajectory
