# Curtis example 5.2, p.270 in my book; also see Orbit_from_r0v0.py
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
# Given r1, r2, and dt, find orbital elements; solve Lamberts problem
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D


# Auxiliary functions
# Stumpff functions originated by Karl Stumpff, circa 1947
# Stumpff functions (C(z), S(z)) are part of a universal variable solution,
#   which is works regardless of eccentricity.
def stumpff_S(z):
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x)) / (x) ** 3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y) / (y) ** 3
    else:
        return 1 / 6


def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2


def y_lambert(z, r1, r2, A):
    K = (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))
    return r1 + r2 + A * K


def A_lambert(r1, r2, d_theta):
    K1 = np.sin(d_theta)
    K2 = np.sqrt((r1 * r2) / (1 - np.cos(d_theta)))
    return K1 * K2


def lambert_zerosolver(z, args):
    dt, mu, r1, r2, A = args
    K1 = ((y_lambert(z, r1, r2, A) / stumpff_C(z)) ** 1.5) * stumpff_S(z)
    K2 = A * np.sqrt(y_lambert(z, r1, r2, A))
    K3 = -1 * dt * np.sqrt(mu)
    return K1 + K2 + K3


def find_f_y(y, r1):
    return 1 - y / r1


def find_g_y(y, A, mu):
    return A * np.sqrt(y / mu)


def find_f_dot_y(y, r1, r2, mu, z):
    K1 = np.sqrt(mu) / (r1 * r2)
    K2 = np.sqrt(y / stumpff_C(z))
    K3 = z * stumpff_S(z) - 1
    return K1 * K2 * K3


def find_g_dot_y(y, r2):
    return 1 - y / r2


# Main function
# Default is prograde trajectory; calling routine may change to retrograde
def Lambert_v1v2_solver(r1_v, r2_v, dt, mu, prograde=True):
    r1 = np.linalg.norm(r1_v)
    r2 = np.linalg.norm(r2_v)

    r1r2z = np.cross(r1_v, r2_v)[2]
    cos_calc = np.dot(r1_v, r2_v) / (r1 * r2)

    if prograde:
        if r1r2z < 0:
            d_theta = 2 * np.pi - np.arccos(cos_calc)
        else:
            d_theta = np.arccos(cos_calc)
    else:
        if r1r2z < 0:
            d_theta = np.arccos(cos_calc)
        else:
            d_theta = 2 * np.pi - np.arccos(cos_calc)

    A = A_lambert(r1, r2, d_theta)
    # set the starting estimate for Lambert solver
    z = scipy.optimize.fsolve(lambert_zerosolver, x0=1.5, args=[dt, mu, r1, r2, A])[0]
    y = y_lambert(z, r1, r2, A)

    f_dt = find_f_y(y, r1)
    g_dt = find_g_y(y, A, mu)
    f_dot_dt = find_f_dot_y(y, r1, r2, mu, z)
    g_dot_dt = find_g_dot_y(y, r2)

    v1_v = (1 / g_dt) * (r2_v - f_dt * r1_v)
    v2_v = (g_dot_dt / g_dt) * r2_v - (
        (f_dt * g_dot_dt - f_dot_dt * g_dt) / g_dt
    ) * r1_v
    return v1_v, v2_v


# Auxiliary functions
def R1(angle):
    A = [1, 0, 0]
    B = [0, np.cos(angle), np.sin(angle)]
    C = [0, -1 * np.sin(angle), np.cos(angle)]
    return [A, B, C]


def R3(angle):
    A = [np.cos(angle), np.sin(angle), 0]
    B = [-1 * np.sin(angle), np.cos(angle), 0]
    C = [0, 0, 1]
    return [A, B, C]


def r_vector_perifocal(theta, h, mu, e):
    A = h**2 / mu
    B = 1 + e * np.cos(theta)
    C = np.array([np.cos(theta), np.sin(theta), 0])
    return (A / B) * C


def v_vector_perifocal(theta, h, mu, e):
    A = mu / h
    B = np.array([-1 * np.sin(theta), e + np.cos(theta), 0])
    return A * B


def geo_to_peri(arg_p, incl, ra_node):
    A = np.array(R3(arg_p))
    B = np.array(R1(incl))
    C = np.array(R3(ra_node))
    return A @ B @ C


# Main functions
def orbit_elements_from_vector(r0_v, v0_v, mu):
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)
    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)

    vr0 = np.dot(r0_vector, v0_vector) / r0

    # Find h=angular momentum
    h_vector = np.cross(r0_vector, v0_vector)
    h = np.linalg.norm(h_vector)

    # Find inclination
    incl = np.arccos(h_vector[2] / h)

    # Find node vector
    N_vector = np.cross([0, 0, 1], h_vector)
    N = np.linalg.norm(N_vector)

    # Find right ascension of ascending node (RAAN)
    ra_node = np.arccos(N_vector[0] / N)
    if N_vector[1] < 0:
        ra_node = 2 * np.pi - ra_node

    # Find eccentricity
    A = (v0**2 - (mu / r0)) * r0_vector
    B = -r0 * vr0 * v0_vector
    e_vector = (1 / mu) * (A + B)
    e = np.linalg.norm(e_vector)
    if e < 0.00005:
        e = 0.0
        arg_p = 999  # meaning undefined
        theta = 999  # meaning undefined
    else:
        # Find argument of periapsis
        if e_vector[2] < 0:
            arg_p = 2 * np.pi - np.arccos(np.dot(N_vector, e_vector) / (N * e))
        else:
            arg_p = np.arccos(np.dot(N_vector, e_vector) / (N * e))

        # Find true anomaly:
        if vr0 < 0:
            theta = 2 * np.pi - np.arccos(np.dot(e_vector, r0_vector) / (e * r0))
        else:
            theta = np.arccos(np.dot(e_vector, r0_vector) / (e * r0))

    # Convert to degrees (can change units here)
    deg_conv = 180 / np.pi
    incl *= deg_conv
    ra_node *= deg_conv
    if e != 0.0:  # make sure theta & arg_p are defined; non-circular orbit
        theta *= deg_conv
        arg_p *= deg_conv

    return [h, e, theta, ra_node, incl, arg_p]


def dt_from_x(x, args):
    r0, vr0, mu, a = args
    # Equation 3.46
    A = (r0 * vr0 / np.sqrt(mu)) * (x**2) * stumpff_C(x**2 / a)
    B = (1 - r0 / a) * (x**3) * stumpff_S(x**2 / a)
    C = r0 * x
    LHS = A + B + C
    return LHS / np.sqrt(mu)


def r_from_x(r0_vector, v0_vector, x, dt, a, mu):
    r0 = np.linalg.norm(r0_vector)
    f = find_f_x(x, r0, a)
    g = find_g_x(x, dt, mu, a)
    return f * r0_vector + g * v0_vector


def e_from_r0v0(r0_v, v0_v, mu):
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)

    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)
    vr0 = np.dot(r0_vector, v0_vector) / r0

    # Find eccentricity
    A = (v0**2 - (mu / r0)) * r0_vector
    B = -r0 * vr0 * v0_vector
    e_vector = (1 / mu) * (A + B)
    e = np.linalg.norm(e_vector)
    return e


# Actual function:
def find_f_x(x, r0, a):
    A = x**2 / r0
    B = stumpff_C(x**2 / a)
    return 1 - A * B


def find_g_x(x, dt, mu, a):
    A = x**3 / np.sqrt(mu)
    return dt - A * stumpff_S(x**2 / a)


def find_f_dot_x(x, mu, r, r0, a):
    A = np.sqrt(mu) / (r * r0)
    B = stumpff_S(x**2 / a) * (x**3 / a)
    return A * (B - x)


def find_g_dot_x(x, r, a):
    A = x**2 / r
    return 1 - A * stumpff_C(x**2 / a)


def plot_orbit_r0v0(r0_v, v0_v, mu, resolution=1000, hyp_span=1):
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)

    # Use Algorithm 3.4
    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)

    vr0 = np.dot(r0_vector, v0_vector) / r0
    a_orbit = 1 / ((2 / r0) - (v0**2 / mu))

    # Check for orbit type, define x_range
    # resolution = number of points plotted
    # span = width of parabolic orbit plotted\
    e = e_from_r0v0(r0_v, v0_v, mu)
    if e >= 1:
        x_max = np.sqrt(np.abs(a_orbit))
        x_array = np.linspace(-hyp_span * x_max, hyp_span * x_max, resolution)
        pos_array = np.array(
            [
                r_from_x(
                    r0_vector,
                    v0_vector,
                    x,
                    dt_from_x(x, [r0, vr0, mu, a_orbit]),
                    a_orbit,
                    mu,
                )
                for x in x_array
            ]
        )

    else:
        x_max = np.sqrt(a_orbit) * (2 * np.pi)
        x_array = np.linspace(0, x_max, resolution)
        pos_array = np.array(
            [
                r_from_x(
                    r0_vector,
                    v0_vector,
                    x,
                    dt_from_x(x, [r0, vr0, mu, a_orbit]),
                    a_orbit,
                    mu,
                )
                for x in x_array
            ]
        )

    # setup 3D orbit plot; also se file Orbit_from_r0v0.py
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Orbit defined by r0 & v0", fontsize=10)
    ax.view_init(elev=30, azim=57, roll=0)
    # set 3D axis labels
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    # plot orbit array
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])
    # plot reference point, r0 and coordinate axis origin
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], ".")
    ax.plot([0], [0], [0], "o", color="black")  # coordinate axis origin
    # plot position line, origin -> to r0
    ax.plot([0, r0_vector[0]], [0, r0_vector[1]], [0, r0_vector[2]], color="orange")
    ax.text(r0_vector[0], r0_vector[1], r0_vector[2], "r(t0)", color="black")

    # coordinate origin point & origin label
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], "o", color="orange")
    ax.text(0, 0, 0, "coordinate axis", color="black", fontsize=6)
    # plot origin axis with a 3D quiver plot
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")


# run the code
r1 = np.array([5000, 10000, 2100])
r2 = np.array([-14600, 2500, 7000])
dt = 60 * 60  # time seperation between r1 and r2
mu_earth_km = 3.986e5  # earth mu [km^3/s^2]

v1, v2 = Lambert_v1v2_solver(r1, r2, dt, mu=mu_earth_km)
print(f"v1= {v1}, v2= {v2}")

orbit_els = orbit_elements_from_vector(r1, v1, mu=mu_earth_km)
# print the orbital elements
# orbit_els() returns [h, e, theta, ra_node, incl, arg_p]
orbit_els_list = ["h", "e", "theta", "ra_node", "incl", "arg_p"]
print("list of orbital element values:")
for x in range(len(orbit_els)):
    print(orbit_els_list[x], "=", orbit_els[x])

plot_orbit_r0v0(
    r2, v2, mu=mu_earth_km, resolution=3000
)  # plot setup, next show() ready

# trying to plot a transparent plane in x,y; the ecliptic plane
# https://stackoverflow.com/questions/56981153/draw-a-transparent-flat-surface-using-mplot3d-in-python
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-10.0, 10.0, .1)
# X, Y = np.meshgrid(x, y)
# Z = f(X,Y)
# ax.plot_surface(X, Y, Z, color='gray',alpha=.8)
# To plot the surface at 100, use your same grid but make all your numbers zero
# Z2 = Z*0.+100
# ax.plot_surface(X, Y, Z2,color='r',alpha=.3) #plot the surface

plt.show()