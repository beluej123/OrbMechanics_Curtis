"""
Constants and their management; units aware
    2025-04-21; figure what works best for me, astropy, pint, or other
    See my units.py and units_test.py files for managing
Ideally each constant will have the following information:
    1) Name
    2) Value
    3) Uncertainty
    4) Units
    5) Reference
"""
import math
from dataclasses import dataclass

import astropy
from pint import UnitRegistry  # units management

ureg = UnitRegistry()

# pint unit definitions
AU_M = 149597870700  # per IAU 2012 Resolution B2
AU_KM = 149597870.700*ureg.km
AU_ = 149598023.0*ureg.km  # [km], Vallado [4] p.1059, tbl.D-5
ASEC360 = 1296000.0
DAY_S = 86400.0  # [sec/day]

# angles
ASEC2RAD = 4.848136811095359935899141e-6
DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi
PI = math.pi
TAU = math.tau  # 2*pi

# physics
C = 299792458.0  # [m/s] speed of light
G_ = 100e-10
GS = 1.32712440017987e20  # [m^3/s^2] heliocentric from DE-405
GM_SUN = 1.32712428e11  # [km^3/s^2], Vallado [4] p.1059, tbl.D-5
GM_EARTH_KM = 3.986004415e5  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3
GM_MARS_KM = 4.305e4  # [km^3/s^2], Vallado [2] p.1041, tbl.D-3

MASS_SUN_KG = 1.9891e30  # [kg], Vallado [4] p.1059, tbl.D-5

# Earth and its orbit.
ANGVEL = 7.2921150e-5  # radians/s
ERAD = 6378136.6  # meters
IERS_2010_INVERSE_EARTH_FLATTENING = 298.25642

# time conversions
T0 = 2451545.0
B1950 = 2433282.4235

C_AUDAY = C * DAY_S / AU_M


# constants.py
@dataclass(frozen=True)
class Constant:
    """
    Constant object for general management.
    google search: "python how to manage constants like astropy"
    """
    value: float
    unit: str
    uncertainty: float
    name: str
    reference: str


def test_constants_1():
    """understand methods to manage constants."""
    print(f"astropy version: {astropy.__version__}")

    SPEED_OF_LIGHT = Constant(
        value=299792458,
        unit="m/s",
        uncertainty=0.0,
        name="Speed of light in vacuum",
        reference="CODATA",
    )
    GRAVITATIONAL_CONSTANT = Constant(
        value=6.6743e-11,
        unit="m^3 kg^-1 s^-2",
        uncertainty=0.0,
        name="Gravitational constant",
        reference="CODATA",
    )
    print(f"speed of light: {SPEED_OF_LIGHT}")
    print(f"speed of light: {SPEED_OF_LIGHT.value}")
    print(f"gravitational constant: {GRAVITATIONAL_CONSTANT}")
    # pint units
    print(f"AU_KM: {AU_KM:~}")


if __name__ == "__main__":
    test_constants_1()
