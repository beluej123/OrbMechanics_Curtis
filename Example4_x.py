"""
Curtis chapter 4, examples collection.

Notes:
----------
    This file is organized with each example as a function; example function name:
        def curtis_ex4_1():
    
    Supporting functions for the test functions below, may be found in other
        files, for example ..., etc. Also note, the test examples are
        collected right after this document block.  However, the example test
        functions are defined/enabled at the end of this file.  Each example
        function is designed to be stand-alone, but, if you use the function
        as stand alone you will need to copy the imports...
    
References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D

import functionCollection as funColl  # includes planetary tables


def node_regression(peri, apo, i, mu, J2, R):
    # inspired by Curtis example 4.9.
    i *= np.pi / 180
    e = (apo - peri) / (apo + peri)
    a = 0.5 * (apo + peri)
    A = np.sqrt(mu) * J2 * R**2
    B = ((1 - e**2) ** 2) * a**3.5
    node_r = -1.5 * (A / B) * np.cos(i)
    # Convert back to deg
    return (180 / np.pi) * node_r


def perigee_advance(peri, apo, i, mu, J2, R):
    # inspired by Curtis example 4.9.
    i *= np.pi / 180
    e = (apo - peri) / (apo + peri)
    a = 0.5 * (apo + peri)
    A = np.sqrt(mu) * J2 * R**2
    B = ((1 - e**2) ** 2) * a**3.5
    C = 2.5 * (np.sin(i)) ** 2 - 2
    peri_adv = -1.5 * (A / B) * C
    # Convert back to deg
    return (180 / np.pi) * peri_adv


def position_to_RA_dec(pos):
    # inspired by Curtis example 4.1.
    i, j, k = pos
    magnitude = np.linalg.norm(pos)
    unit_vec = np.array(pos) / magnitude
    # Find declination
    dec = np.arcsin(unit_vec[2])

    if min(i, j) == 0:
        return ["One coordinate is 0, check again!"]

    # Find right ascension; check for corrrect quadrant

    if np.sign(j) == 1:
        RA = np.arccos(unit_vec[0] / np.cos(dec))
    else:
        RA = 2 * np.pi - np.arccos(unit_vec[0] / np.cos(dec))

    return [RA * (180 / np.pi), dec * (180 / np.pi), magnitude, unit_vec]
    # Return degrees; can convert to hours if necessary


# Find x
def stumpff_S(z):
    # inspired by Curtis example 4.2
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x)) / (x) ** 3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y) / (y) ** 3
    else:
        return 1 / 6


def stumpff_C(z):
    # inspired by Curtis example 4.2
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2


def universalx_zerosolver(x, args):
    # inspired by Curtis example 4.2
    r0, vr0, mu, dt, a = args

    A = stumpff_C((x**2) / a) * ((r0 * vr0) / (np.sqrt(mu))) * (x**2)
    B = stumpff_S((x**2) / a) * (1 - r0 / a) * (x**3)
    C = r0 * x
    D = np.sqrt(mu) * dt
    return A + B + C - D


# write f,g functions for x
# inspired by Curtis example 4.2
def find_f_x(x, r0, a):
    A = x**2 / r0
    B = stumpff_C(x**2 / a)
    return 1 - A * B


def find_g_x(x, dt, mu, a):
    # inspired by Curtis example 4.2
    A = x**3 / np.sqrt(mu)
    return dt - A * stumpff_S(x**2 / a)


def find_f_dot_x(x, mu, r, r0, a):
    # inspired by Curtis example 4.2
    A = np.sqrt(mu) / (r * r0)
    B = stumpff_S(x**2 / a) * (x**3 / a)
    return A * (B - x)


def find_g_dot_x(x, r, a):
    # inspired by Curtis example 4.2
    A = x**2 / r
    return 1 - A * stumpff_C(x**2 / a)


# Find delta_t for each x
def dt_from_x(x, args):
    # inspired by Curtis example 4.2
    r0, vr0, mu, a = args
    # Equation 3.46
    A = (r0 * vr0 / np.sqrt(mu)) * (x**2) * stumpff_C(x**2 / a)
    B = (1 - r0 / a) * (x**3) * stumpff_S(x**2 / a)
    C = r0 * x
    LHS = A + B + C
    return LHS / np.sqrt(mu)


def r_from_x(r0_vector, v0_vector, x, dt, a, mu):
    # inspired by Curtis example 4.2
    r0 = np.linalg.norm(r0_vector)
    f = find_f_x(x, r0, a)
    g = find_g_x(x, dt, mu, a)
    return f * r0_vector + g * v0_vector


def curtis_ex4_1():
    """
    Curtis pp.205 , example 4.1; algorithm 4.1. r0_vector -> right ascension, declination

    Given:
        r0_vector
    Find:
        right ascension, declination

    Notes:
    ----------
        Also see interplanetary flight http://www.braeunig.us/space/interpl.htm
        References: see list at file beginning.
    """
    ra, dec, magP, uVec = position_to_RA_dec([-5368, -1784, 3691])
    print(
        "right ascension =",
        ra,
        "\ndeclination=",
        dec,
        "\ndistance=",
        magP,
        "\nunit vector=",
        uVec,
    )
    return None


def curtis_ex4_2():
    """
    Curtis pp.206 , example 4.2 including plot; note example 3.7.
        r0_vector, v0_vector + tof (time-of-flight) -> r1_vector, v1_vector + orbit plot.

    Given:
        r0_vector, v0_vector, tof
    Find:
        r1_vector, v1_vector

    References: see list at file beginning.
    Notes:
    ----------
        Also see interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    # At time t0, the state vector of an Earth satellite is
    r0_vector = np.array([1600, 5310, 3800])  # [km]
    v0_vector = np.array([-7.350, 0.4600, 2.470])  # [km/s]

    # Determine the position and velocity 3200 seconds later
    # and plot the orbit in three dimensions.
    mu = 398600
    dt = 3200

    # Use Algorithm 3.4; see example 3.7
    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)

    vr0 = np.dot(r0_vector, v0_vector) / r0

    # semimajor axis
    a_orbit = 1 / ((2 / r0) - (v0**2 / mu))

    x0_guess = dt * np.sqrt(mu) * np.absolute(1 / a_orbit)

    x_dt = scipy.optimize.fsolve(
        universalx_zerosolver, x0=x0_guess, args=[r0, vr0, mu, dt, a_orbit]
    )[0]

    f_dt = find_f_x(x_dt, r0, a_orbit)
    g_dt = find_g_x(x_dt, dt, mu, a_orbit)

    r_dt_vector = f_dt * r0_vector + g_dt * v0_vector
    r_dt = np.linalg.norm(r_dt_vector)

    f_dot_dt = find_f_dot_x(x_dt, mu, r_dt, r0, a_orbit)
    g_dot_dt = find_g_dot_x(x_dt, r_dt, a_orbit)

    v_dt_vector = f_dot_dt * r0_vector + g_dot_dt * v0_vector
    g_dt = np.linalg.norm(v_dt_vector)

    # display position and velocity
    print(
        "position(3200)=", r_dt_vector, "[km]\nvelocity(3200)=", v_dt_vector, "[km/s]"
    )

    # Plot the orbit
    # First, find x_max
    x_max = np.sqrt(a_orbit) * (2 * np.pi)

    # Create a linspace for x, set resolution
    resolution = 1000
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

    # setup 3D plot
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Ex4.2; Find r(t1) & v(t1), given r(t0), v(t0)", fontsize=10)
    ax.view_init(elev=25, azim=-115, roll=0)
    # plot orbit array
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])
    # plot r0 & r1 vector end points
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], "o", color="orange")
    ax.plot([r_dt_vector[0]], [r_dt_vector[1]], [r_dt_vector[2]], "x", color="red")
    # plot text; start, stop, coordinate axis
    ax.text(r0_vector[0], r0_vector[1], r0_vector[2], "r(t0)", color="black")
    ax.text(r_dt_vector[0], r_dt_vector[1], r_dt_vector[2], "r(t1)", color="black")
    ax.text(0, 0, 0, "coordinate axis", color="black", fontsize=6)
    # plot position lines, origin -> to, t1(stop)
    ax.plot([0, r0_vector[0]], [0, r0_vector[1]], [0, r0_vector[2]], color="orange")
    ax.plot([0, r_dt_vector[0]], [0, r_dt_vector[1]], [0, r_dt_vector[2]], color="red")
    # plot axis labels
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    # plot origin axis with a 3D quiver plot
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[4000, 0, 0], [0, 4000, 0], [0, 0, 4000]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")
    # ax.set_axis_off()

    plt.show()

    return None


def curtis_ex4_7():
    """
    Orbital elements (coe) -> state vectors (IJK).  Curtis p.232 , example 4.7, algorithm 4.5.
    For sv -> coe, Curtis pp.209, algorithm 4.2, & Curtis pp.212, example 4.3.
    Also see interplanetary flight http://www.braeunig.us/space/interpl.htm

    Given:
        earth orbit, sets mu value
        h    = [km^3/s^2] angular mumentum, 
        ecc  = [-] eccentricity
        incl = [deg] inclination angle; to the ecliptic
        RA   = [deg] RAAN, right ascension of ascending node (aka capital W)
        w    = [deg] arguement of periapsis (NOT longitude of periapsis, w_bar)
        TA   = [deg] true angle/anomaly at time x (aka theta, or nu)
        
        Other Elements (not given, but useful to understand):
        sma    : semi-major axis (aka a; often replaces h)
        t_peri : time of periapsis passage
        w_bar  : [deg] longitude of periapsis (NOT arguement of periapsis, w)
                Note, w_bar = w + RAAN
        L_     : [deg] mean longitude (NOT mean anomaly, M)
                Note, L = w_bar + M
        M_     : mean anomaly (often replaces TA)
    Find:
        r_vec
        v_vec

    Notes:
    ----------
        Uses Curtis, pp.471, algorithm 8.1.  Note Curtis p.277, example 5.4, Sideral time.
        Also see Vallado functions: pp. 296, planetRV() (algotithm 33),
            cov2rv() (algorithm 11), et.al
        Orbital elements tables kept in functionCollection.py
        For my code, generally angles are saved in [rad].

        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
        References: see list at file beginning.
    """
    mu_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    h, ecc, incl_deg, RA_deg, w_deg, TA_deg = 80000, 1.4, 30, 40, 60, 30
    
    incl = incl_deg * math.pi / 180
    RA = RA_deg * math.pi / 180
    w = w_deg * math.pi / 180
    TA = TA_deg * math.pi / 180

    r_vec, v_vec = funColl.sv_from_coe(
        h=h, ecc=ecc, RA=RA, incl=incl, w=w, TA=TA, mu=mu_earth_km
    )
    print(f"r_vec= {r_vec} [km]")
    print(f"v_vec= {v_vec} [km/s]")

    return None


def curtis_ex4_9():
    # Example 4.9
    # A satellite is to be launched into a sun-synchronous circular orbit with
    #   a period of 100 minutes.
    # Determine the required altitude (r) and orbit inclination (incl).
    period = 100 * 60  # [s]
    mu = 398600  # earth mu value [km^3 / s^2]
    rE = 6378.0  # earth radius [km]

    # T = 2pi/rt(mu) * r^1.5

    r = (period * np.sqrt(mu) / (2 * np.pi)) ** (2 / 3)

    d_node_r = 0.9856 * (np.pi / 180) / (24 * 3600)

    cos_incl = (
        -1 * d_node_r / (1.5 * ((np.sqrt(mu) * 0.00108263 * 6378**2) / ((r**3.5))))
    )
    incl = np.arccos(cos_incl) * (180 / np.pi)
    print("orbit altitude = ", r - rE)
    print("orbit inclination = ", incl, "[deg]")
    return None


def test_curtis_ex4_1():
    print(f"\nTest Curtis example 4.1, ... :")
    # function does not need input parameters.
    curtis_ex4_1()
    return None


def test_curtis_ex4_2():
    print(f"\nTest Curtis example 4.2, ... :")
    # function does not need input parameters.
    curtis_ex4_2()
    return None


def test_curtis_ex4_7():
    print(f"\nTest Curtis example 4.7, ... :")
    # function does not need input parameters.
    curtis_ex4_7()
    return None


def test_curtis_ex4_9():
    print(f"\nTest Curtis example 4.9, ... :")
    # function does not need input parameters.
    curtis_ex4_9()
    return None


# use the following to test/examine functions
if __name__ == "__main__":

    # test_curtis_ex4_1()  # test curtis example 4.1
    # test_curtis_ex4_2()  # test curtis example 4.2
    test_curtis_ex4_7()  # test curtis example 4.7
    # test_curtis_ex4_9()  # test curtis example 4.9
