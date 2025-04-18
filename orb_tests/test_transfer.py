"""
Orbit transfer test description goes here.
    2025-04-16. Python importing remains an issue for me to figure out!
        For now, ignore the import errors below.
Curtis 3rd. ed. pp.393, example 8.8
"""

import os
import sys

import numpy as np

# Set import path to orb_src; this import scheme works for me for now...
#   I cannot figure out the python import to get modules from another directory,
#       that also import modules from that other directory.
cwd = os.getcwd()
print(f"current dir:\n{cwd}")
module_dir = os.path.join(cwd, "orb_src")  # places correct dir seperator
print(f"Set module dir:\n{module_dir}")
sys.path.insert(0, module_dir)
from coe_from_sv import coe_from_sv
from month_planet_names import month_planet_names
from transfer_orb import interplanetary


def test_transfer():
    """
    This program uses Algorithm 8.2 to solve interplanetary trajectory.

    mu           - gravitational parameter of the sun (km^3/s^2)
    deg          - conversion factor between degrees and radians
    pi           - 3.1415926...

    planet_id    - planet identifier:
                    1 = Mercury
                    2 = Venus
                    3 = Earth
                    4 = Mars
                    5 = Jupiter
                    6 = Saturn
                    7 = Uranus
                    8 = Neptune
                    9 = Pluto

    year         - range: 1901 - 2099
    month        - range: 1 - 12
    day          - range: 1 - 31
    hour         - range: 0 - 23
    minute       - range: 0 - 60
    second       - range: 0 - 60

    depart       - [planet_id, year, month, day, hour, minute, second]
                    at departure
    arrive       - [planet_id, year, month, day, hour, minute, second]
                    at arrival

    planet1      - [Rp1, Vp1, jd1]
    planet2      - [Rp2, Vp2, jd2]
    trajectory   - [V1, V2]

    coe          - orbital elements [h e RA incl w TA]
                    where
                    h    = angular momentum (km^2/s)
                    e    = eccentricity
                    RA   = right ascension of the ascending
                            node (rad)
                    incl = inclination of the orbit (rad)
                    w    = argument of perigee (rad)
                    TA   = true anomaly (rad)
                    a    = semimajor axis (km)

    jd1, jd2     - Julian day numbers at departure and arrival
    tof          - time of flight from planet 1 to planet 2 (days)

    Rp1, Vp1     - state vector of planet 1 at departure (km, km/s)
    Rp2, Vp2     - state vector of planet 2 at arrival (km, km/s)
    R1, V1       - heliocentric state vector of spacecraft at
                    departure (km, km/s)
    R2, V2       - heliocentric state vector of spacecraft at
                    arrival (km, km/s)

    vinf1, vinf2 - hyperbolic excess velocities at departure
                    and arrival (km/s)

    User py-functions required: interplanetary, coe_from_sv,
                               month_planet_names
    """
    # Define constants
    mu = 1.327124e11  # Gravitational parameter of the Sun (km^3/s^2)
    deg = np.pi / 180

    # Departure data
    depart = [
        3,  # planet_id (Earth)
        1996,  # year
        11,  # month
        7,  # day
        0,  # hour
        0,  # minute
        0,  # second
    ]

    # Arrival data
    arrive = [
        4,  # planet_id (Mars)
        1997,  # year
        9,  # month
        12,  # day
        0,  # hour
        0,  # minute
        0,  # second
    ]

    # Algorithm 8.2: Compute planetary states and trajectory
    planet1, planet2, trajectory = interplanetary(depart, arrive, mu)

    # Extract state vectors and Julian dates
    Rp1, Vp1, jd1 = planet1
    Rp2, Vp2, jd2 = planet2
    V1, V2 = trajectory

    # Time of flight (days)
    tof = jd2 - jd1

    # Orbital elements of the spacecraft
    coe_depart = coe_from_sv(Rp1, V1, mu)
    coe_arrive = coe_from_sv(Rp2, V2, mu)

    # Hyperbolic excess velocities
    vinf1 = np.array(V1) - np.array(Vp1)
    vinf2 = np.array(V2) - np.array(Vp2)

    # Departure details; month=not used
    month, planet = month_planet_names(depart[2], depart[0])
    print("-----------------------------------------------------")
    print(" Departure:")
    print(f"   Planet: {planet}")
    print(f"   Date      : {depart[1]}-{depart[2]}-{depart[3]} {depart[4]}:{depart[5]}:{depart[6]}")
    print(f"   Julian day: {jd1:.3f}")
    print(f"   Planet position vector (km) = {Rp1}")
    print(f"     Magnitude                 = {np.linalg.norm(Rp1):.3f}")
    print(f"   Planet velocity (km/s)      = {Vp1}")
    print(f"     Magnitude                 = {np.linalg.norm(Vp1):.3f}")
    print(f"   Spacecraft velocity (km/s)  = {V1}")
    print(f"     Magnitude                 = {np.linalg.norm(V1):.3f}")
    print(f"   v-infinity at departure (km/s) = {vinf1}")
    print(f"     Magnitude                    = {np.linalg.norm(vinf1):.3f}")

    # Time of flight
    print(f"\n Time of flight = {tof:.3f} days")

    # Arrival details; month=not used
    month, planet = month_planet_names(arrive[2], arrive[0])
    print("\nArrival:")
    print(f"   Planet: {planet}")
    print(f"   Date      : {arrive[1]}-{arrive[2]}-{arrive[3]} {arrive[4]}:{arrive[5]}:{arrive[6]}")
    print(f"   Julian day: {jd2:.3f}")
    print(f"   Planet position vector (km) = {Rp2}")
    print(f"     Magnitude                 = {np.linalg.norm(Rp2):.3f}")
    print(f"   Planet velocity (km/s)      = {Vp2}")
    print(f"     Magnitude                 = {np.linalg.norm(Vp2):.3f}")
    print(f"   Spacecraft Velocity (km/s)  = {V2}")
    print(f"     Magnitude                 = {np.linalg.norm(V2):.3f}")
    print(f"   v-infinity at arrival (km/s)  = {vinf2}")
    print(f"     Magnitude                   = {np.linalg.norm(vinf2):.3f}")

    # Orbital elements
    print("\nOrbital elements of flight trajectory:")
    print(f"  Angular momentum (km^2/s)                   = {coe_depart[0]:.3e}")
    print(f"  Eccentricity                                = {coe_depart[1]:.6f}")
    print(f"  Right ascension of the ascending node (deg) = {coe_depart[2] / deg:.3f}")
    print(f"  Inclination to the ecliptic (deg)           = {coe_depart[3] / deg:.3f}")
    print(f"  Argument of perihelion (deg)                = {coe_depart[4] / deg:.3f}")
    print(f"  True anomaly at departure (deg)             = {coe_depart[5] / deg:.3f}")
    print(f"  True anomaly at arrival (deg)               = {coe_arrive[5] / deg:.3f}")
    print(f"  Semimajor axis (km)                         = {coe_depart[6]:.3e}")

    if coe_depart[1] < 1:  # If orbit is an ellipse, output the period
        period = 2 * np.pi / np.sqrt(mu) * coe_depart[6] ** 1.5 / 24 / 3600
        print(f"  Period (days)                               = {period:.3f}")

    print("-----------------------------------------------------")


# Call the function
def main():
    """Execute module..."""
    test_transfer()


if __name__ == "__main__" and __package__ is None:
    # 2025-04-16, python importing is very ugly!
    main()
