# Curtis pp.462, example 8.6.
# Orbital Mechanics for Engineering Students, 2nd ed., 2009
# helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
import math

import numpy as np  # for vector math

np.set_printoptions(precision=4)  # set vector printing

# TODO update variable names to be consistant with darkside & lightside calculations

# A venus flyby mission.  Spacecraft departs earth with a velocity perpendicular to the sun line.
# Encounter occurs at a true anomaly in the approach trajectory of −30◦.
# Periapse altitude 300 km.
# (a) Dark side Venus apporach, show the post-flyby orbit is as shown in Figure 8.20.
# (b) Sunlit side Venus approach, show the post-flyby orbit is as shown in Figure 8.21.
#
# Leading-side flyby results in a decrease in the spacecraft's heliocentric speed.
# Trailing-side flyby increases helliocentric speed;
# e1, h1, and θ1 are eccentricity, angular momentum, and true anomaly of heliocentric approach trajectory.

mu_sun = 1.327e11  # [km^3/s^2]
mu_venus = 324900  # [km^3/s^2]
mu_earth = 398600  # [km^3/s^2]
mu_mars = 42830  # [km^3/s^2]

r_venus_orb = 108.2e6  # venus orbit around sun [km]
r_earth_orb = 149.6e6  # earth orbit [km]
r_mars_orb = 227.9e6  # mars orbit [km]

r_venus = 6052  # venus radius [km]
r_earth = 6378  # earth radius [km]
r_mars = 3396  # mars radius [km]
alt_earth = 300  # altitude above earth [km]
alt_venus = 300  # altitude above venus [km]

nu_venus = -30 * math.pi / 180  # venus approach true anomaly (nu); saved as [rad]

# part a, Pre-Flyby ellipse; p.462+
# orbit id (1), transfer orbit eccentricity; p.464
ecc1_venus_orb = (r_earth_orb - r_venus_orb) / (
    r_earth_orb + r_venus_orb * math.cos(nu_venus)
)
print(f"eccentricity, at venus, ecc1_venus_orb = {ecc1_venus_orb:.5g}")

# orbit 1 angular momentum; p.464
h1 = math.sqrt(mu_sun * r_earth_orb * (1 - ecc1_venus_orb))
print(f"angular momentum, orbit1, h1 = {h1:.5g} [km^2/s]")

# Calculate spacecraft radial and transverse components heliocentric velocity at
# the inbound crossing of Venus’s sphere of influence.
v1_perp = h1 / r_venus_orb  # perpendicular velocity orbit 1[km/s]
v1_radi = (
    (mu_sun / h1) * (ecc1_venus_orb) * math.sin(nu_venus)
)  # radial velocity orbit 1[km/s]
print(f"velocity inbound perpendicular, v1_perp = {v1_perp:.5g} [km/s]")
print(f"velocity inbound radial, v1_radi = {v1_radi:.5g} [km/s]")

# flight path angle; p.464; eqn 2.51 on p.xx
# The following negative sign is consistent with the spacecraft flying towards
#   perihelion of the pre-flyby elliptical trajectory (orbit 1).
gamma1 = math.atan(v1_radi / v1_perp)
print(f"flight path angle, gamma1 = {gamma1*180/math.pi:.5g} [deg]")
# Speed of the space vehicle at the inbound crossing
v_in = math.sqrt(v1_perp**2 + v1_radi**2)
print(f"velocity inbound (from SOI) = {v_in:.5g} [km/s]")

# part a, Flyby Hyperbola; p.464+
print("********** darkside flyby hyperbola **********")
# velocity inbound (1) vector, planet, sun direction coordinates
v1p_vec = np.array([v1_perp, -v1_radi])  # [km/s]
v1p_mag = np.linalg.norm(v1p_vec)  # [km/s]
print(f"velocity inbound vector, v1p_vec = {v1p_vec} [km/s]")
# assume venus in circular orbit; velocity planet (venus) relative vector
vp_vec = np.array([math.sqrt(mu_sun / r_venus_orb), 0])  # [km/s]
print(f"velocity planet (venus) vector, vp_vec = {vp_vec} [km/s]")

# p.465
v1_infty_vec = v1p_vec - vp_vec
print(f"velocity,inbound from infinity, v1_infty_vec = {v1_infty_vec} [km/s]")
v1_infty = np.linalg.norm(v1_infty_vec)
print(f"velocity inbound, magnitude, v1_infty = {v1_infty:.5g} [km/s]")

# hyperbola periapsis radius; p.465
rp_venus = r_venus + alt_venus  # [km]
# planetcentric angular momentum & eccentricity; eqns 8.39, 8.39; p.465
h2 = rp_venus * math.sqrt(v1_infty**2 + 2 * mu_venus / rp_venus)
ecc1_venus = 1 + (rp_venus * v1_infty**2) / mu_venus
print(f"angular momentum, h2 = {h2:.5g} [km^2/s]")
print(f"eccentricity, inbound, ecc1_venus = {ecc1_venus:.5g} [km^2/s]")

# turn angle and true anomaly of asymptote
delta_turn1 = 2 * math.asin(1 / ecc1_venus)
nu_asym = math.acos(-1 / ecc1_venus)
print(f"turn angle inbound, delta_turn1 = {delta_turn1*180/math.pi:.5g} [deg]")
print(f"true anomaly of asymptote, nu_asym = {nu_asym*180/math.pi:.5g} [deg]")

# aiming radius; p.465; eqns. 2.50, 2.103, 2.107
delta_aim = rp_venus * math.sqrt((ecc1_venus + 1) / (ecc1_venus - 1))
print(f"aiming radius, delta_aim = {delta_aim:.5g} [km]")

# angle between v1_infty and v_venus; p.465
phi1 = math.atan(v1_infty_vec[1] / v1_infty_vec[0])
print(f"true anomaly of asymptote, inbound, phi1 = {phi1*180/math.pi:.5g} [deg]")

# part a, Dark Side Approach; p.466
# There are two flyby approaches:
# (1) Dark side approach, the turn angle is counterclockwise (+102.9◦)
# (2) Sunlit side approach, the turn anble is clockwise (−102.9◦).

# angle between v_infty & V_venus_vec, darkside turn; eqn 8.85; p.466
phi2 = phi1 + delta_turn1
print(f"darkside turn angle, phi2 = {phi2*180/math.pi:.5g} [deg]")

# eqn 8.86; p.466
v2_infty_vec = v1_infty * np.array([math.cos(phi2), math.sin(phi2)])  # [km/s]
print(f"darkside velocity infinity, v2_infty_vec = {v2_infty_vec} [km/s]")

# outbound velocity vector, planet, sun direction coordinates; p.466
v2p_vec = vp_vec + v2_infty_vec  # [km/s]
print(f"outbound velocity vector, v2p_vec = {v2p_vec} [km/s]")
v2p_mag = np.linalg.norm(v2p_vec)
print(f"outbound crossing velocity, magnitude, v2p_mag = {v2p_mag:.5g} [km/s]")
print(f"compare darkside inbound/outbound speeds: {(v2p_mag-v1p_mag):.5g} [km/s]")

# part a, Post Flyby Ellipse (orbit 2) for Darkside Approach; p.467
# The heliocentric post flyby trajectory, orbit 2.
# Angular momentum orbit 2; eqn 8.90.
ho2 = r_venus_orb * v2p_vec[0]
print(f"angular momentum, orbit 2, ho2 = {ho2:.5g} [km/s]")
ecc_cos = (ho2**2 / (mu_sun * r_venus_orb)) - 1
ecc_sin = -v2p_vec[1] * ho2 / (mu_sun)
ecc_tan = ecc_sin / ecc_cos
print(f"interium, ecc_cos = {ecc_cos:.5g}")
print(f"interium, ecc_sin = {ecc_sin:.5g}")
print(f"interium, ecc_tan = {ecc_tan:.5g}")
theta2 = math.atan(ecc_tan)
print(f"theta2, 1st possibility = {theta2*180/math.pi:.5g} [deg]")
print(f"theta2, 2nd possibility = {(theta2*180/math.pi)+180:.5g} [deg]")
# based on cos and sin quadrants select angle
if (ecc_cos < 0 and ecc_sin < 0) or (ecc_cos > 0 and ecc_sin < 0):
    # quadrant 3 or quadrant 4
    theta2 = theta2 + math.pi
    print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")
else:
    print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")

print(f"perihelion of departure, theta2 = {theta2*180/math.pi:.5g} [deg]")
ecc2_venus = ecc_cos / math.cos(theta2)
print(f"eccentricity, orbit 2, ecc2_venus = {ecc2_venus:.5g}")

r2_perihelion = (ho2**2 / mu_sun) * (1 / (1 + ecc2_venus))
print(f"radius orbit2, perihelion, r2_perihelion = {r2_perihelion:.5g}")

# part b, Sunlit side approach; p.467+
print("\n********** sunlit approach **********")  # make line seperation in print list
# angle lightside, v_infty & V_venus_vec, outbound crossing; p.467
phi2 = phi1 - delta_turn1
print(f"lightside turn angle, phi2 = {phi2*180/math.pi:.5g} [deg]")

# velocity 2 lightside vector; p.468
v2l_infty_vec = v1_infty * np.array([math.cos(phi2), math.sin(phi2)])  # [km/s]
print(f"lightside velocity infinity, v2l_infty_vec = {v2l_infty_vec} [km/s]")

# velocity outbound lightside vector, planet, sun direction coordinates; p.468
v2pl_vec = vp_vec + v2l_infty_vec  # [km/s]
print(f"outbound velocity vector lightside, v2pl_vec = {v2pl_vec} [km/s]")
v2pl_mag = np.linalg.norm(v2pl_vec)
print(
    f"outbound crossing velocity lightside, magnitude, v2pl_mag = {v2pl_mag:.5g} [km/s]"
)
print(f"compare lightside inbound/outbound speeds: {(v2pl_mag-v1p_mag):.5g} [km/s]")

print("********** post flyby ellipse **********")
# Angular momentum lightside orbit 2; eqn 8.90.
h_lo2 = r_venus_orb * v2pl_vec[0]
print(f"angular momentum, lightside orbit 2, ho2 = {h_lo2:.5g} [km/s]")
ecc_cos = (h_lo2**2 / (mu_sun * r_venus_orb)) - 1
ecc_sin = -v2pl_vec[1] * h_lo2 / mu_sun
ecc_tan = ecc_sin / ecc_cos
print(f"interium, ecc_cos = {ecc_cos:.5g}")
print(f"interium, ecc_sin = {ecc_sin:.5g}")
print(f"interium, ecc_tan = {ecc_tan:.5g}")
theta2 = math.atan(ecc_tan)
print(f"theta2, 1st possibility = {theta2*180/math.pi:.5g} [deg]")
print(f"theta2, 2nd possibility = {(theta2*180/math.pi)+180:.5g} [deg]")
# based on cos and sin quadrants select angle
if (ecc_cos < 0 and ecc_sin < 0) or (ecc_cos > 0 and ecc_sin < 0):
    # quadrant 3 or quadrant 4
    theta2 = theta2 + math.pi
    print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")
else:
    print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")

print(f"departure perihelion, lightside, theta2 = {theta2*180/math.pi:.5g} [deg]")
ecc2_venus = ecc_cos / math.cos(theta2)
print(f"eccentricity, orbit 2, ecc2_venus = {ecc2_venus:.5g}")

r2_perihelion = (h_lo2**2 / mu_sun) * (1 / (1 + ecc2_venus))
print(f"radius orbit2, perihelion lightside, r2_perihelion = {r2_perihelion:.5g}")
