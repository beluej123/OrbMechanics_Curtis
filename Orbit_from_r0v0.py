#  based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#  by Howard D. Curtis
# Also see Example5_x.py, ex5.2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Stumpff functions originated by Karl Stumpff, circa 1947
# Stumpff functions (stumpff_C(z), stumpff_S(z)) are part of a universal variable solution,
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


# Plot function:
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
    # span = width of orbit plot
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

    # setup 3D orbit plot
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Orbit defined by r0 & v0", fontsize=10)
    ax.view_init(elev=15, azim=55, roll=0)
    # plot orbit array
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])

    # plot reference point, r0 and coordinate axis origin
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], ".")
    ax.plot([0], [0], [0], "o", color="black")  # coordinate axis origin
    # plot position lines, origin -> to r0
    ax.plot([0, r0_vector[0]], [0, r0_vector[1]], [0, r0_vector[2]], color="orange")
    ax.text(r0_vector[0], r0_vector[1], r0_vector[2], "r(t0)", color="black")

    # plot axis labels
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    # coordinate origin point & origin label
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], "o", color="orange")
    ax.text(0, 0, 0, "coordinate axis", color="black", fontsize=6)
    # plot origin axis with a 3D quiver plot
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")


# Units: r0 [km], v0 [km/s], mu [km3/s2]
# Change units as needed (make sure all units are consistent)
# mu is G*M, m mass of primary body, G is gravitational constant
plot_orbit_r0v0([10, -15, -10], [47, 19, -21], 74000)  # plot setup, next show() ready
plt.show()

################################################################
# # Solve equation of motion numerically.
# #     not sure I want to keep the following
# # https://kyleniemeyer.github.io/space-systems-notes/orbital-mechanics/two-body-problems.html
# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt


# def rhs(t, z):
#     # 3D orbital motion ODE
#     mu = 398.6e3  # earth mu value [km^3/s^2]
#     r = np.sqrt(z[0] ** 2 + z[1] ** 2 + z[2] ** 2)
#     dzdt = np.zeros(6)
#     dzdt[0] = z[3]
#     dzdt[1] = z[4]
#     dzdt[2] = z[5]
#     dzdt[3] = (-mu / r**3) * z[0]
#     dzdt[4] = (-mu / r**3) * z[1]
#     dzdt[5] = (-mu / r**3) * z[2]
#     return dzdt


# r0 = [20000, -105000, -19000]  # [km]
# v0 = [0.900, -3.4000, -1.500]  # [km/s]
# T = 360000.0  # delta time after r0 & v0

# sol = solve_ivp(rhs, [0, 2 * T], np.array(r0 + v0))
# # plot solution
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # plot axis labels
# ax.set_xlabel("x-axis")
# ax.set_ylabel("y-axis")
# ax.set_zlabel("z-axis")

# ax.plot3D(sol.y[0, :], sol.y[1, :], sol.y[2, :])
# plt.show()
