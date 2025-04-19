"""Constants used all over."""

import math

# Definitions.
AU_M = 149597870700  # per IAU 2012 Resolution B2
AU_KM = 149597870.700
AU_ = 149598023.0  # [km], Vallado [4] p.1059, tbl.D-5
ASEC360 = 1296000.0
DAY_S = 86400.0  # [sec/day]

# Angles.
ASEC2RAD = 4.848136811095359935899141e-6
DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi
pi = math.pi
tau = math.tau  # lower case, for symmetry with math.pi

# Physics.
C = 299792458.0  # [m/s] speed of light
G_ = 100e-10  # [??]
GS = 1.32712440017987e20  # [m^3/s^2] heliocentric from DE-405
GM_SUN_Pitjeva_2005_km3_s2 = 132712440042  # Elena Pitjeva, 2015JPCRD..44c1210P
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

C_AUDAY = C * DAY_S / AU_M  # ?
