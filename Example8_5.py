# Curtis example 8.5 (p.456+ in my book). H.W. Curtis
# Orbital Mechanics for Engineering Students, 2nd ed., 2009
import math

# After a Hohmann transfer from earth to mars, calculate:
# (a) the minimum delta_v required to place spacecraft in an orbit with 7hour period
# (b) periapsis radius, the aiming radius and the angle between periapse and Mars’ velocity vector.
# (c) aiming radius
# (d) angle between periapsis and Mars' velocity vector

mu_sun = 1.327e11  # [km^3/s^2]
mu_earth = 398600  # [km^3/s^2]
mu_mars = 42830  # [km^3/s^2]

r_earth_orb = 149.6e6  # earth solar orbit [km]
r_mars_orb = 227.9e6  # mars solar orbit [km]

r_earth = 6378  # earth radius [km]
r_mars = 3396  # earth radius [km]
alt_earth = 300  # altitude above earth [km]

T_mars_orb = 7 * 60 * 60  # satellite period in mars orbit [s]

# part a
# from eqn 8.4
v_inf = math.sqrt(mu_sun / r_mars_orb) * (
    1 - math.sqrt(2 * r_earth_orb / (r_earth_orb + r_mars_orb))
)
print(f"arrive v_infinity, v_inf = {v_inf:.5g} [km/s]")

# Semi-major axis of capture orbit
a_capture = (T_mars_orb * math.sqrt(mu_mars) / (2 * math.pi)) ** (2 / 3)
print(f"arrive semi-major axis = {a_capture:.5g} [km]")

# from eqn 8.67, not sure my print description below is correct
ecc_mars_orb = (2 * mu_mars / (a_capture * v_inf**2)) - 1
print(f"eccentricity, at mars = {ecc_mars_orb:.5g}")

# from eqn 8.70
delta_v = v_inf * math.sqrt((1 - ecc_mars_orb) / 2)
print(f"delta_v enter mars = {delta_v:.5g} [km/s]")

# part b
# periapsis radius at mars capture, from eqn 8.67
r_p = (2 * mu_mars / v_inf**2) * (1 - ecc_mars_orb) / (1 + ecc_mars_orb)
print(f"periapsisr_p at mars = {r_p:.5g} [km]")

# part c
# aiming radius from eqn 8.71
aim_radius = r_p * math.sqrt(2 / (1 - ecc_mars_orb))
print(f"aiming radius (aka delta) at mars = {aim_radius:.5g} [km]")

# part d
# angle to periapsis from eqn 8.43
beta_p = math.acos(1 / (1 + r_p * v_inf**2 / mu_mars))
print(f"angle to periapsis at mars = {(beta_p*180/math.pi):.5g} [km]")
