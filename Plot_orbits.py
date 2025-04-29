"""
2024-10-30, a bunch of clean-up remains; redundancy + comments + cohesion with the other files.
See Example5_x.py, ex5.2

Notes:
----------
    This file is organized with each example as a function; example function name:
        def curtis_ex5_1():
    
    All supporting functions for all examples are collected right after this
    document block, and all example test functions are defined/enabled at the
    end of this file.  Each example function is designed to be stand-alone,
    however, you need to copy the imports and the supporting functions.

References:
----------
    See references.py for references list.
"""

import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from astroquery.jplhorizons import Horizons
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.axes3d import Axes3D

from func_gen import hohmann_transfer_a
from Stumpff_1 import stumpff_C, stumpff_S


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


# Plot function:
def plot_orbit_r0v0(r0_v, v0_v, mu, resolution=1000, hyp_span=1):
    """
    Need help clearing matplotlib errors:
        https://www.scaler.com/topics/matplotlib/matplotlib-3d-plot/

    Input Parameters:
    ----------
        r0_v       : np.array, initial position
        v0_v       : np.array, initial velocity
        mu         : orbit gravitational parameter
        resolution : int
        hyp_span   :
    Notes:
    ----------

    """
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
    # fig = plt.figure(dpi=120)
    ax = plt.figure().add_subplot(111, projection="3d")
    print(f"type {type(ax)}")
    ax.set_title("Orbit defined by r0 & v0", fontsize=10)
    ax.view_init(elev=20, azim=-48, roll=0)
    # plot orbit array
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])

    # plot reference point, r0 and coordinate axis origin
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], ".")
    ax.plot([0], [0], [0], "o", color="black")  # coordinate axis origin
    # plot position lines, origin -> to r0
    ax.plot([0, r0_vector[0]], [0, r0_vector[1]], [0, r0_vector[2]], color="orange")
    ax.text(r0_vector[0], r0_vector[1], r0_vector[2], "r(t0)", color="black")

    # plot axis labels
    ax.set(
        xlabel="x-axis",
        ylabel="y-axis",
        zlabel="z-axis",
        # zticks=[0, -150, -300, -450],
    )

    # coordinate origin point & origin label
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], "o", color="orange")
    ax.text(0, 0, 0, "coordinate axis", color="black", fontsize=6)

    # plot origin axis with a 3D quiver plot
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")

    return


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

# plot axis labels
# ax.set(
#     xlabel="x-axis",
#     ylabel="y-axis",
#     zlabel="z-axis",
#     # zticks=[0, -150, -300, -450],
# )

# ax.plot3D(sol.y[0, :], sol.y[1, :], sol.y[2, :])
# plt.show()
# return None


def orbits_2D_animation():
    """
    Solar system simulation. On-line import initial planets state from JPL
        Horizons, using astroquery.
        https://ssd-api.jpl.nasa.gov/doc/horizons.html
    This function uses only the sun as the gravitational influence - does not
        include influence from other bodies.  Uses a modified version of Euler's
        integration method, where the first equation is forward and the second
        equation is backward. Unlike the normal Euler's method, this modified
        version is stable.
    Returns:
    ----------
    Notes:
    ----------
        Began with 2D animation from ChongChong He, "Simulating a real solar
            system with 70 lines of Python code". I made changes to make it work.
    """

    # The next commented out code is just exploring the the JPL Horizons on-line
    #   download data
    # https://medium.com/analytics-vidhya/simulating-the-solar-system-with-under-100-lines-of-python-code-5c53b3039fc6
    # JPL Horizons, https://ssd-api.jpl.nasa.gov/doc/horizons.html
    # obj = Horizons(id=1, location="@sun", epochs=Time("2017-01-01").jd).vectors()
    # print(f"{obj}")
    # obj = Horizons(id=1, location="@sun", epochs=Time("2017-01-01").jd).elements()
    # p_elements = [np.double(obj[xi]) for xi in ["e", "q", "nu", "a", "Q", "P"]]
    # print(f"{obj}")
    # =====================================================

    sim_start_date = "2018-01-01"  # simulating a solar system starting from this date
    sim_duration = 2 * 365  # simulation duration in days
    m_earth = 5.9722e24 / 1.98847e30  # Mass of Earth relative to mass of the sun
    m_moon = 7.3477e22 / 1.98847e30

    class Object:  # define the objects: the Sun, Earth, Mercury, etc
        def __init__(self, name, rad, color, r, v):
            self.name = name
            self.r = np.array(r, dtype=float)
            self.v = np.array(v, dtype=float)
            self.xs = []
            self.ys = []
            self.plot = ax.scatter(
                r[0], r[1], color=color, s=rad**2, edgecolors=None, zorder=10
            )
            (self.line,) = ax.plot([], [], color=color, linewidth=1.2)

    class SolarSystem:
        def __init__(self, thesun):
            self.thesun = thesun
            self.planets = []
            self.time = Time("2018-01-01")  # placeholder time
            self.timestamp = ax.text(
                0.03,
                0.94,
                "Date: ",
                color="w",
                transform=ax.transAxes,
                fontsize="x-large",
            )

        def add_planet(self, planet):
            self.planets.append(planet)

        def evolve(self):  # evolve the trajectories
            dt = 1.0
            self.time += TimeDelta(1, format="jd")
            plots = []
            lines = []
            # in units of AU/day^2
            for p in self.planets:
                p.r += p.v * dt
                acc = -2.959e-4 * p.r / np.sum(p.r**2) ** (3.0 / 2)
                p.v += acc * dt
                p.xs.append(p.r[0])
                p.ys.append(p.r[1])
                p.plot.set_offsets(p.r[:2])
                p.line.set_xdata(p.xs)
                p.line.set_ydata(p.ys)
                plots.append(p.plot)
                lines.append(p.line)
            self.timestamp.set_text(f"Date: {self.time.iso}")
            return plots + lines + [self.timestamp]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=[6, 6])
    ax = plt.axes([0.0, 0.0, 1.0, 1.0], xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ss = SolarSystem(Object("Sun", 28, "red", [0, 0, 0], [0, 0, 0]))
    ss.time = Time(sim_start_date)
    ss.time.format = "jd"
    colors = ["gray", "orange", "green", "chocolate"]
    sizes = [0.38, 0.45, 0.5, 0.53]
    names = ["Mercury", "Venus", "Earth", "Mars"]
    texty = [0.47, 0.73, 1, 1.5]
    # The 1st, 2nd, 3rd, 4th planet in solar system
    for i, nasaid in enumerate([1, 2, 3, 4]):
        obj = Horizons(id=nasaid, location="@sun", epochs=ss.time).vectors()
        # print() troubleshooting
        print(f"Initial Horizons fetched state:")
        print(f"planet {i}, {[np.double(obj[xi]) for xi in ["x", "y", "z"]]}")

        ss.add_planet(
            Object(
                nasaid,
                20 * sizes[i],
                colors[i],
                [np.double(obj[xi]) for xi in ["x", "y", "z"]],
                [np.double(obj[vxi]) for vxi in ["vx", "vy", "vz"]],
            )
        )
        ax.text(
            0,
            -(texty[i] + 0.1),
            names[i],
            color=colors[i],
            zorder=1000,
            ha="center",
            fontsize="large",
        )

    def animate(i):
        return ss.evolve()

    ani = animation.FuncAnimation(
        fig,
        animate,
        repeat=False,
        frames=sim_duration,
        blit=True,
        interval=20,
    )
    plt.show()
    # ani.save('solar_system_6in_150dpi.mp4', fps=60, dpi=150)
    return


def orbits_3D_animation():
    # # Set up the figure and 3D axis
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Define the number of frames for the animation
    # num_frames = 100
    # # Generate data for the orbits
    # theta = np.linspace(0, 2 * np.pi, num_frames)
    # r = 1  # Radius of the orbit
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    # z = np.zeros(num_frames)  # Assuming the orbit is in the xy-plane
    # # Initialize the plot
    # planet, = ax.plot([], [], [], 'bo', markersize=10)  # Blue planet
    # star, = ax.plot([0], [0], [0], 'yo', markersize=20)  # Yellow star
    # # Set the limits of the plot
    # # ax.set_xlim(-1.5, 1.5)
    # # ax.set_ylim(-1.5, 1.5)
    # # ax.set_zlim(-1.5, 1.5)
    # # plot axis labels
    # ax.set(
    #     xlabel='x-axis',
    #     ylabel='y-axis',
    #     zlabel='z-axis',
    #     xlim=(-1.5,1.5),
    #     ylim=(-1.5,1.5),
    #     zlim=(-1.5,1.5),
    # )
    # # Animation update function
    # def update(frame):
    #     planet.set_data(x[frame], y[frame])
    #     planet.set_3d_properties(z[frame])
    #     return planet,
    # # Create the animation
    # ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    # # Show the animation
    # plt.show()
    return


# =============================================================================
def orbit_r0v0(r0_v, v0_v, mu, resolution=1000, hyp_span=1):
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
    return pos_array


def maneuver_plot(r0_v, v0_v, dv_v, mu, resolution=1000, hyp_span=1):
    # Plot initial orbit
    initial_orbit = orbit_r0v0(r0_v, v0_v, mu, resolution=resolution, hyp_span=hyp_span)

    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(initial_orbit[:, 0], initial_orbit[:, 1], initial_orbit[:, 2])
    ax.plot([np.array(r0_v)[0]], [np.array(r0_v)[1]], [np.array(r0_v)[2]], ".")
    ax.plot([0], [0], [0], "o", color="k")

    # Find new orbit
    v0_dv = np.array(v0_v) + np.array(dv_v)
    new_orbit = orbit_r0v0(r0_v, v0_dv, mu, resolution=resolution, hyp_span=hyp_span)

    # Plot new orbit
    ax.plot(new_orbit[:, 0], new_orbit[:, 1], new_orbit[:, 2])
    return None


def test_plot_maneuver():
    # Units of r0 in km, v0 in km/s, mu in km3/s2
    # Change units as necessary (all consistent)
    # mu is G*M, m mass of primary body, G is gravitational constant

    maneuver_plot([203, -30, 0], [-18, 33, 0], [23, 1, -30], 344000, hyp_span=5)
    plt.show()

    return None


def plot_earth_mars():
    """
    From google search: python plot planetary hohmann transfer
    """
    # Constants
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    M_sun = 1.989e30  # Mass of the Sun (kg)
    AU = 1.496e11  # Astronomical Unit (m)

    # Orbit radii (Earth and Mars, approximated as circular)
    r_earth = 1 * AU
    r_mars = 1.524 * AU

    # Hohmann transfer orbit semi-major axis
    a_transfer = (r_earth + r_mars) / 2

    # Calculate velocities
    v_earth = np.sqrt(G * M_sun / r_earth)
    v_transfer_perigee = np.sqrt(G * M_sun * (2 / r_earth - 1 / a_transfer))
    v_transfer_apogee = np.sqrt(G * M_sun * (2 / r_mars - 1 / a_transfer))
    v_mars = np.sqrt(G * M_sun / r_mars)

    # Calculate delta-v
    delta_v1 = v_transfer_perigee - v_earth
    delta_v2 = v_mars - v_transfer_apogee

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("Hohmann Transfer Orbit from Earth to Mars")
    ax.grid(True)

    # Plot Sun
    ax.plot(0, 0, "yo", markersize=10, label="Sun")

    # Plot Earth orbit
    theta_earth = np.linspace(0, 2 * np.pi, 100)
    x_earth = r_earth * np.cos(theta_earth) / AU
    y_earth = r_earth * np.sin(theta_earth) / AU
    ax.plot(x_earth, y_earth, "b--", label="Earth Orbit")

    # Plot Mars orbit
    theta_mars = np.linspace(0, 2 * np.pi, 100)
    x_mars = r_mars * np.cos(theta_mars) / AU
    y_mars = r_mars * np.sin(theta_mars) / AU
    ax.plot(x_mars, y_mars, "r--", label="Mars Orbit")

    # Plot transfer orbit
    theta_transfer = np.linspace(0, np.pi, 100)
    x_transfer = a_transfer * np.cos(theta_transfer) / AU
    y_transfer = (
        a_transfer
        * np.sin(theta_transfer)
        * np.sqrt(1 - ((a_transfer - r_earth) / a_transfer) ** 2)
        / AU
    )  # Corrected y-coordinate calculation
    x_transfer -= (a_transfer - r_earth) / AU
    ax.plot(x_transfer, y_transfer, "g-", label="Hohmann Transfer Orbit")

    # Mark Earth and Mars positions at transfer start/end
    ax.plot(r_earth / AU, 0, "bo", label="Earth (Start)")
    ax.plot(-r_mars / AU, 0, "ro", label="Mars (Arrival)")

    ax.legend()
    plt.show()
    return


def ellipse_position(a, b, angle_rad):
    # Function to calculate the position of a point on an ellipse
    x = a * np.cos(angle_rad)
    y = b * np.sin(angle_rad)
    return x, y


def plotA_ellipse_2_ellipse():
    """
    2025-02-07, transfer orbit graphic NOT correct.
    Calculates and plots the Hohmann transfer orbit between two elliptical
        orbits, showing the initial, transfer, and final paths, as well as
        marking key points and the central body.
    From google search: plot hohmann transfer between elliptical orbits python
    """
    # Define the parameters of the initial elliptical orbit
    a1 = 10  # Semi-major axis of the initial orbit
    b1 = 6  # Semi-minor axis of the initial orbit
    e1 = math.sqrt(1 - (b1**2) / (a1**2))  # Eccentricity of the initial orbit
    theta1 = np.linspace(0, 2 * np.pi, 100)  # Angle values for the initial orbit

    # Define the parameters of the final elliptical orbit
    a2 = 20  # Semi-major axis of the final orbit
    b2 = 12  # Semi-minor axis of the final orbit
    e2 = math.sqrt(1 - (b2**2) / (a2**2))  # Eccentricity of the final orbit
    theta2 = np.linspace(0, 2 * np.pi, 100)  # Angle values for the final orbit

    # Calculate the semi-major axis of the transfer orbit
    at = (a1 + a2) / 2

    # Calculate the positions on the orbits
    x1, y1 = ellipse_position(a1, b1, theta1)
    x2, y2 = ellipse_position(a2, b2, theta2)

    # Calculate the transfer orbit
    theta_t = np.linspace(0, np.pi, 100)
    xt = at * np.cos(theta_t)
    yt = at * np.sin(theta_t)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")

    # Plot the initial and final elliptical orbits
    ax.plot(x1, y1, label="Initial Orbit")
    ax.plot(x2, y2, label="Final Orbit")

    # Plot the transfer orbit
    ax.plot(xt, yt, label="Transfer Orbit", linestyle="--")

    # Mark the starting and ending points of the transfer
    ax.plot(a1, 0, "go", label="Start of Transfer")
    ax.plot(a2, 0, "ro", label="End of Transfer")

    # Add a central body (e.g., a planet)
    ax.plot(0, 0, "ko", markersize=10, label="Central Body")

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Hohmann Transfer between Elliptical Orbits")
    ax.legend()
    ax.grid(True)

    plt.show()
    return None


def plotB_ellipse_2_ellipse():
    """
    2025-02-07, transfer orbit graphic ALMOST correct.
    Calculates and plots the Hohmann transfer orbit between two elliptical
        orbits. It visualizes the initial, final, and transfer orbits, along
        with the central body.
    From google search: plot hohmann transfer between two elliptical orbits python
    """
    # Constants
    mu = 1  # Gravitational parameter

    # Ellipse parameters
    a1 = 1.0  # Semi-major axis of initial ellipse
    e1 = 0.5  # Eccentricity of initial ellipse
    a2 = 2.0  # Semi-major axis of final ellipse
    e2 = 0.3  # Eccentricity of final ellipse

    # Calculate periapsis and apoapsis distances
    r_p1 = a1 * (1 - e1)
    r_a1 = a1 * (1 + e1)
    r_p2 = a2 * (1 - e2)
    r_a2 = a2 * (1 + e2)

    # Transfer ellipse semi-major axis
    a_t = (r_a1 + r_p2) / 2

    # Calculate velocities
    v1_a = np.sqrt(mu * (2 / r_a1 - 1 / a1))
    v_trans_a = np.sqrt(mu * (2 / r_a1 - 1 / a_t))
    v2_p = np.sqrt(mu * (2 / r_p2 - 1 / a2))
    v_trans_p = np.sqrt(mu * (2 / r_p2 - 1 / a_t))

    # Calculate delta-v
    delta_v1 = v_trans_a - v1_a
    delta_v2 = v2_p - v_trans_p

    # Generate points for plotting ellipses
    theta = np.linspace(0, 2 * np.pi, 100)
    x1 = a1 * (np.cos(theta) - e1)
    y1 = a1 * np.sqrt(1 - e1**2) * np.sin(theta)
    x2 = a2 * (np.cos(theta) - e2)
    y2 = a2 * np.sqrt(1 - e2**2) * np.sin(theta)

    # Generate points for transfer ellipse
    x_trans = a_t * np.cos(np.linspace(np.pi, 2 * np.pi, 100))
    y_trans = (
        a_t
        * np.sin(np.linspace(np.pi, 2 * np.pi, 100))
        * np.sqrt(1 - ((r_a1 - a_t) / a_t) ** 2)
    )

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(x1, y1, label="Initial Orbit")
    plt.plot(x2, y2, label="Final Orbit")
    plt.plot(x_trans, y_trans, label="Transfer Orbit")
    plt.scatter(0, 0, color="red", marker="+", s=100, label="Central Body")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Hohmann Transfer between Elliptical Orbits")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
    return None


def plot_orbits(planets, start_planet, intermediate_planet, target_planet):
    """Plots the orbits of the planets and the Hohmann transfer orbits.

    Args:
        planets: Dictionary of planet data (name and semi-major axis).
        start_planet: Name of the starting planet
        intermediate_planet: Name of the intermediate planet
        target_planet: Name of the target planet
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_title("Hohmann Transfers")

    # Plotting the Sun
    ax.plot(0, 0, "yo", markersize=10, label="Sun")

    # Plotting planet orbits
    for name, radius in planets.items():
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot(x, y, "--", label=f"{name} Orbit")

    # Calculate and plot the first transfer orbit
    transfer1_r, transfer1_theta = hohmann_transfer_a(
        planets[start_planet], planets[intermediate_planet]
    )
    x1 = transfer1_r * np.cos(transfer1_theta)
    y1 = transfer1_r * np.sin(transfer1_theta)
    ax.plot(x1, y1, "r-", label=f"{start_planet} to {intermediate_planet} Transfer")

    # Calculate and plot the second transfer orbit
    transfer2_r, transfer2_theta = hohmann_transfer_a(
        planets[intermediate_planet], planets[target_planet]
    )
    x2 = transfer2_r * np.cos(transfer2_theta + np.pi)
    y2 = transfer2_r * np.sin(transfer2_theta + np.pi)
    ax.plot(x2, y2, "g-", label=f"{intermediate_planet} to {target_planet} Transfer")

    ax.legend()
    ax.grid(True)
    plt.show()
    return None


def test_plot_orbits():
    # Plot Hohmann transfer orbits between three planets
    # Planet parameters (semi-major axis in AU)
    planet_data = {"Earth": 1.00, "Mars": 1.52, "Jupiter": 5.20}
    plot_orbits(planet_data, "Earth", "Mars", "Jupiter")
    return


def test_plot_orbit_r0v0():
    """
    Development of general purpose orbit ploting

    Notes:
    ----------
        Units: r0 [km], v0 [km/s], mu [km3/s2]
        Change units as needed (make sure all units are consistent)
        mu is G*M, m mass of primary body, G is gravitational constant
    """
    plot_orbit_r0v0([10, -15, -10], [47, 19, -21], 74000)
    plt.show()
    return None


def main():
    # just a placeholder to help with editor navigation:--)
    pass
    return


# use the following to test/examine functions
if __name__ == "__main__":

    # test_plot_orbit_r0v0()  #
    # plot_earth_mars()  # hohmann orbit
    # plotA_ellipse_2_ellipse()  #
    plotB_ellipse_2_ellipse()  #
    # test_plot_orbits() #
    # orbits_2D_animation()  #
    # orbits_3D_animation()  #
    # test_plot_maneuver()  #
    main()
