"""
ALGORITHM 8.2: spacecraft trajectory, planet 1 -> planet 2
    Curtis [3] p.393, Curtis [9] p.428
2025-04-16. Python importing remains an issue for me to figure out!
    For now, ignore the linting import errors below.
Explore class structure for solar system bodies:
    https://thepythoncodingbook.com/2021/12/11/simulating-3d-solar-system-python-matplotlib/
    https://github.com/codetoday-london/simulation-3d-solar-system-in-python-using-matplotlib
"""

from lambert import lambert
from planet_elements_and_sv import planet_elements_and_sv


def interplanetary(depart, arrive, mu):
    """
    Determine spacecraft trajectory from the sphere
    of influence of planet 1 to that of planet 2 using Algorithm 8.2

    mu          - gravitational parameter of the sun (km^3/s^2)
    dum         - a dummy vector not required in this procedure

    planet_id   - planet identifier:
                  1 = Mercury
                  2 = Venus
                  3 = Earth
                  4 = Mars
                  5 = Jupiter
                  6 = Saturn
                  7 = Uranus
                  8 = Neptune
                  9 = Pluto

    year        - range: 1901 - 2099
    month       - range: 1 - 12
    day         - range: 1 - 31
    hour        - range: 0 - 23
    minute      - range: 0 - 60
    second      - range: 0 - 60

    jd1, jd2    - Julian day numbers at departure and arrival
    tof         - time of flight from planet 1 to planet 2 (s)
    Rp1, Vp1    - state vector of planet 1 at departure (km, km/s)
    Rp2, Vp2    - state vector of planet 2 at arrival (km, km/s)
    R1, V1      - heliocentric state vector of spacecraft at
                  departure (km, km/s)
    R2, V2      - heliocentric state vector of spacecraft at
                  arrival (km, km/s)

    depart      - [planet_id, year, month, day, hour, minute, second]
                  at departure
    arrive      - [planet_id, year, month, day, hour, minute, second]
                  at arrival

    planet1     - [Rp1, Vp1, jd1]
    planet2     - [Rp2, Vp2, jd2]
    trajectory  - [V1, V2]

    User py-functions required: planet_elements_and_sv, lambert
    """
    # Unpack departure details
    planet_id = depart[0]
    year = depart[1]
    month = depart[2]
    day = depart[3]
    hour = depart[4]
    minute = depart[5]
    second = depart[6]

    # Algorithm 8.1 for planet 1's state vector
    #   _ = placeholder, means function return variable ignored
    _, rp1, vp1, jd1 = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second
    )

    # Unpack arrival details
    planet_id = arrive[0]
    year = arrive[1]
    month = arrive[2]
    day = arrive[3]
    hour = arrive[4]
    minute = arrive[5]
    second = arrive[6]

    # Likewise use Algorithm 8.1 to obtain planet 2's state vector
    #   _ = placeholder, means function return variable ignored
    _, rp2, vp2, jd2 = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second
    )

    tof = (jd2 - jd1) * 24 * 3600

    # Patched conic assumption
    r1 = rp1
    r2 = rp2

    # Use Algorithm 5.2 to find the spacecraft's velocity at
    # departure and arrival, assuming a prograde trajectory
    v1, v2 = lambert(r1, r2, tof, "pro", mu)

    # Output as structured data
    planet1 = (rp1, vp1, jd1)
    planet2 = (rp2, vp2, jd2)
    trajectory = (v1, v2)

    return planet1, planet2, trajectory
