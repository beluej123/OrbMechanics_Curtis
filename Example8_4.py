# Curtis example 8.4 (p.446+ in my book). H.W. Curtis
# Orbital Mechanics for Engineering Students, 2nd ed., 2009
import math

# A spacecraft is launched on a Mars mission, starting from a 300km circular parking
# orbit. Calculate (a) the delta-v required; (b) the location of perigee of the departure
# hyperbola; (c) the amount of propellant required as a percentage of the spacecraft
# mass before the delta-v burn, assuming a specific impulse of 300 seconds.
mu_sun = 1.327e11  # [km^3/s^2]
mu_earth = 398600  # [km^3/s^2]
r_earth_orb = 149.6e6  # earth orbit [km]
r_mars_orb = 227.9e6  # mars orbit [km]

r_earth = 6378  # earth radius [km]
alt_earth = 300  # altitude above earth [km]

# part a
# from eqn 8.35, p442
v_inf = math.sqrt(mu_sun / r_earth_orb) * (
    math.sqrt(2 * r_mars_orb / (r_earth_orb + r_mars_orb)) - 1
)
print(f"depart v_infinity, v_inf = {v_inf:.5g} [km/s]")

# spacecraft speed in 300km circular parking orbit; eqn 8.41
v_c = math.sqrt(
    mu_earth / (r_earth + alt_earth)
)  # velocity circular parking orbit - need better name
print(f"spacecraft speed in parking orbit = {v_c:.5g} [km/s]")

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
