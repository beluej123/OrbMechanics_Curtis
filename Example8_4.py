"""
Curtis example 8.4 (p.446).  Earth->Mars Mission.
helpful interplanetary flight http://www.braeunig.us/space/interpl.htm

Given:
        earth orbit launch, alt=300 [km] circular, parabolic launch trajectory
            thus ecc=1, and Earth GM (or mu)
        r1: periapsis altitude 500 [km];
        r2: earth-sun SOI (sphere of influence); soi calculation known

    Find:
        (a) delta-v required
        (b) departure hyperbola perigee location
        (c) propellant as a percentage of the spacecraft, before delta-v burn
            assume Isp (specific impulse) = 300 [s]
        
    
References
    ----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.)
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import math

# constants; mostly from Vallado not Curtis
au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
mu_sun = GM_sun_km  # [km^3/s^2]
mu_earth = GM_earth_km  # [km^3/s^2]

r_earth_orb = 149598023  # [km], Vallado p.1041, tbl.D-3
r_mars_orb = 227939186  # [km], Vallado p.1041, tbl.D-3

r_earth = 6378.1363  # [km], Vallado p.1041, tbl.D-3
alt_earth = 300  # [km], given altitude above earth

# part a
# Curtis p.442, eqn 8.35
v_inf = math.sqrt(mu_sun / r_earth_orb) * (
    math.sqrt(2 * r_mars_orb / (r_earth_orb + r_mars_orb)) - 1
)
print(f"depart v_infinity, v_inf = {v_inf:.5g} [km/s]")

# spacecraft speed in 300km circular parking orbit; Curtis p.444, eqn 8.41
v_c = math.sqrt(
    mu_earth / (r_earth + alt_earth)
)  # departure from circular parking orbit
print(f"departure parking orbit, v_c= {v_c:.5g} [km/s]")

# Delta_v required to enter departure hyperbola; eqn 8.42, p444
delta_v = v_c * (math.sqrt(2 + (v_inf / v_c) ** 2) - 1)
print(f"delta_v to enter departure hyperbola = {delta_v:.5g} [km/s]")

# part b
# Perigee of the departure hyperbola, relative to the earth’s orbital velocity vector
# eqn 8.43, p444
r_p = r_earth + alt_earth  # periapsis
beta_depart = math.acos(1 / (1 + r_p * v_inf**2 / mu_earth))
print(f"departure hyperbola beta angle= {beta_depart*180/math.pi:.5g} [deg]")
ecc_depart = 1 + (r_p * v_inf**2) / mu_earth
print(f"eccentricity, departure hyperbola = {ecc_depart:.5g}")

# part c
# Perigee can be located on either the sun lit or darkside of the earth.
# It is likely that the parking orbit would be a prograde orbit (west to east),
# which would place the burnout point on the darkside.
I_sp = 300  # [s]
g_0 = 9.81e-3  # [km/s^2]
delta_mRatio = 1 - math.exp(-delta_v / (I_sp * g_0))
print(f"Propellant mass ratio = {delta_mRatio:.5g}")
