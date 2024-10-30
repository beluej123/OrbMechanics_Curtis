"""
Orbital Mechanics for Engineering Students, 2nd ed., 2009
See Example5_x.py, ex5.2
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


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


def orbit_animation():
    """
    Solar system simulation. On-line import initial planets state from JPL
        Horizons, using astroquery.
        https://ssd-api.jpl.nasa.gov/doc/horizons.html
    This function uses only the sun as the gravitational influence - does not
        include influence from other bodies.  Uses a modified version of Euler's
        integration method, where the first equation is forward and the second
        equation is backward. Unlike the normal Euler's method, this modified
        version is stable.
    

    Returns
    ----------
    Notes:
    ----------
        Began with 2D animation from ChongChong He, "Simulating a real solar
            system with 70 lines of Python code". I made changes to make it work.


    """

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.time import Time, TimeDelta
    from astroquery.jplhorizons import Horizons

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
            self.time = Time("2018-01-01") # placeholder time
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
            self.time += TimeDelta(1, format='jd')
            plots = []
            lines = []
            # in units of AU/day^2
            for p in self.planets:
                p.r += p.v * dt
                acc = (
                    -2.959e-4 * p.r / np.sum(p.r**2) ** (3.0 / 2)
                )
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

    # =====================================================

    # ani.save('solar_system_6in_150dpi.mp4', fps=60, dpi=150)

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


def main():
    # just a placeholder to help with editor navigation:--)
    return


# use the following to test/examine functions
if __name__ == "__main__":

    # test_plot_orbit_r0v0()  #
    orbit_animation()  #
