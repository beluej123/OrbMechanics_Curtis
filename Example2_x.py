# Curtis example's 2.12 (p.116), 2.13 (p.123), 2.14 (p.124)
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
# May shorten development; see https://github.com/jkloser/OrbitalMechanics
import numpy as np


def find_r(h, mu_e, r0, vr0, d_theta):
    A = h**2 / mu_e
    B = ((h**2 / (mu_e * r0)) - 1) * (np.cos(d_theta))
    C = -(h * vr0 / mu_e) * (np.sin(d_theta))
    return A * (1 / (1 + B + C))


def find_f(mu_e, r, h, d_theta):
    A = mu_e * r / (h**2)
    B = 1 - np.cos(d_theta)
    return 1 - A * B


def find_g(r, r0, h, d_theta):
    A = r * r0 / h
    B = np.sin(d_theta)
    return A * B


def find_f_dot(mu_e, h, d_theta, r0, r):
    A = mu_e / h
    B = (1 - np.cos(d_theta)) / (np.sin(d_theta))
    C = mu_e / (h**2)
    D = 1 - np.cos(d_theta)
    E = 1 / r0
    F = 1 / r
    return A * B * (C * D - E - F)


def find_g_dot(mu_e, r0, h, d_theta):
    A = mu_e * r0 / (h**2)
    B = 1 - np.cos(d_theta)
    return 1 - A * B


def find_position(f, g, r0, v0):
    return f * r0 + g * v0


def find_velocity(f_dot, g_dot, r0, v0):
    return f_dot * r0 + g_dot * v0


def e_from_r0v0(r0_v, v0_v, mu):
    # Find eccentricity
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)

    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)
    vr0 = np.dot(r0_vector, v0_vector) / r0

    A = (v0**2 - (mu / r0)) * r0_vector
    B = -r0 * vr0 * v0_vector
    e_vector = (1 / mu) * (A + B)
    e = np.linalg.norm(e_vector)
    return e


# constants:
mu_e = 3.986e5  # earth mu [km^3/s^2]
r_ea = 6378  # earth radius [km]

# Example 2.12; perifocal frame
# Given: r0 & v0 (vectors, pqw frame; i.e. orbital plane)
# Find: specific angular momentum, eccentricity, true anomaly
print("-- Example 2.12 (p.116), Perifocal Frame --")
r0_vector = np.array([7000, 9000, 0])  # pqw frame [km]
v0_vector = np.array([-5, 7, 0])  # pqw frame [km/s]

r0 = np.linalg.norm(r0_vector)
h_vector = np.cross(r0_vector, v0_vector)
h = np.linalg.norm(h_vector)
print("angular momentum=", h_vector, "[km^2/s]")
print("angular momentum=", h, "[km^2/s]")

theta = np.arccos(r0_vector / r0)[0]  # chose p coordinate in pqw frame
if r0_vector[1] < 0:
    theta = -theta
theta_deg = theta * 180 / np.pi
print("true anomaly=", theta_deg, "[deg]")

ecc = (((h**2) / (r0 * mu_e)) - 1) / np.cos(theta)
print("eccentricity=", ecc)


##########################################################
# Example 2.13 (p.123), Lagrange Coefficients
# Given: r0 & v0 (vectors, 2D), and angle change from r0->r1
# Find: new position (r1) and new velocity (v1) vectors
print("\n-- Example 2.13 (p.123), Lagrange Coefficients --")
r0_vector = np.array([8182.4, -6865.9, 0])
v0_vector = np.array([0.47572, 8.8116, 0])
d_angle_deg = 120  # [deg]
d_angle = d_angle_deg * (np.pi / 180)
# Calculate h (constant), r0, vr0
h = np.linalg.norm(np.cross(r0_vector, v0_vector))
r0 = np.linalg.norm(r0_vector)
vr0 = np.dot(v0_vector, (r0_vector / r0))  # velocity @ r0

r = find_r(h, mu_e, r0, vr0, d_angle)

f = find_f(mu_e, r, h, d_angle)
g = find_g(r, r0, h, d_angle)
f_dot = find_f_dot(mu_e, h, d_angle, r0, r)
g_dot = find_g_dot(mu_e, r0, h, d_angle)

final_position = find_position(f, g, r0_vector, v0_vector)
final_velocity = find_velocity(f_dot, g_dot, r0_vector, v0_vector)
# r0 & v0 are in plane; thus no 3rd vector element
print("given; initial position:", r0_vector[:-1], "[km]")
print("given; initial velocity:", v0_vector[:-1], "[km/s]")
print("given; position angle change:", d_angle_deg, "[deg]")
print("\nfinal position:", final_position[:-1], "[km]")
print("final velocity:", final_velocity[:-1], "[km/s]")

##########################################################
# Example 2.14 (p.124), follow up to example 2.13
# Given: r0 & v0 (vectors, 2D), and change in position angle
# Find: new position and velocity vectors
print(
    "\n-- Example 2.14 (p.124), uses results of example 2.13, Lagrange Coefficients --"
)
print("r0=", r0, "[km]")
if vr0 < 0:
    print("vr0<0, approaching periapsis: vr0=", vr0, "[km/s]")
else:
    print("vr0>=0, leaving periapsis: vr0=", vr0, "[km/s]")
print("angular momentum=", h, "[km^2/s]")

# used Maple to verify algebra for ecc expression
ecc_sq = (h**2 / (r0 * mu_e) - 1) ** 2 + (vr0**2) * h**2 / mu_e**2
ecc = np.sqrt(ecc_sq)
print("eccentricity=", ecc)

theta = np.arccos((h**2 - mu_e * r0) / (r0 * mu_e * ecc))
theta_deg = theta * 180 / np.pi
if vr0 < 0:
    theta = 2 * np.pi - theta
    theta_deg = theta * 180 / np.pi
    print("vr0<0, approaching, true anomaly", theta_deg, "[deg]")
else:
    print("vr0>=0, leaving, true anomaly", theta_deg, "[deg]")


# Plot orbit; note figure 2.31, p.126
# setup 3D orbit plot
# import matplotlib.pyplot as plt

# def plot_orbit():

#     z_array = np.linspace(-10, 10, 200)
#     s_array = [stumpff_S(z) for z in z_array]
#     c_array = [stumpff_C(z) for z in z_array]

#     plt.figure(dpi=120)
#     plt.plot(z_array, s_array)
#     plt.plot(z_array, c_array)

#     fig = plt.figure(dpi=120)
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set_title("Orbit defined by r0 & v0", fontsize=10)
#     ax.view_init(elev=15, azim=55, roll=0)
#     # plot orbit array
#     ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])

#     # plot reference point, r0 and coordinate axis origin
#     ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], ".")
#     ax.plot([0], [0], [0], "o", color="black")  # coordinate axis origin
#     # plot position lines, origin -> to r0
#     ax.plot([0, r0_vector[0]], [0, r0_vector[1]], [0, r0_vector[2]], color="orange")
#     ax.text(r0_vector[0], r0_vector[1], r0_vector[2], "r(t0)", color="black")

#     # plot axis labels
#     ax.set_xlabel("x-axis")
#     ax.set_ylabel("y-axis")
#     ax.set_zlabel("z-axis")
#     # coordinate origin point & origin label
#     ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], "o", color="orange")
#     ax.text(0, 0, 0, "coordinate axis", color="black", fontsize=6)
#     # plot origin axis with a 3D quiver plot
#     x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
#     u, v, w = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
#     ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")
