"""
Skyfield repo + Curtis book functions collection; may need to break this up.
Functions & operations repeatedly used throughout Skyfield repo.
    Curtis examples and problems.
TODO ***** need to put some Curtis vectors into python numpy syntax *****
TODO ***** eliminate global variables *****
    
Notes:
----------
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally angles are saved in [rad], distance [km].
    
References:
----------
    See references.py for references list.
"""

import datetime
import math
from pkgutil import get_data

import numpy as np
import pint
import scipy.optimize  # used to solve kepler E
from numpy import (
    arcsin,
    arctan2,
    array,
    cos,
    einsum,
    finfo,
    float64,
    full_like,
    load,
    nan,
    rollaxis,
    sin,
    sqrt,
    where,
)

from astro_time import g_date2jd, julian_date
from constants_1 import AU_KM, CENT, DEG, DEG2RAD, RAD, RAD2DEG, TAU
from func_coords import ecliptic_to_equatorial
from Stumpff_1 import stumpff_C, stumpff_S

ureg = pint.UnitRegistry()
_AVOID_DIVIDE_BY_ZERO = finfo(float64).tiny


class reify:
    """
    This class from google search, python how to make reify as a class
    Skyfield repo uses a different version; not sure if it is better.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.__doc__ = wrapped.__doc__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            value = self.wrapped(instance)
            setattr(instance, self.wrapped.__name__, value)
            return value


# class reify(object):
#   commented out from skyfield repo
#     """Adapted from Pyramid's `reify()` memoizing decorator."""
#     def __init__(self, method):
#         self.method = method
#         update_wrapper(self, method)

#     def __get__(self, instance, objtype=None):
#         if instance is None:
#             return self
#         value = self.method(instance)
#         instance.__dict__[self.__name__] = value
#         return value


class A(object):
    """Allow literal NumPy arrays to be spelled A[1, 2, 3]."""

    __getitem__ = array


A = A()


def sqrt_nan(n):
    """Return the square root of n, or nan if n < 0."""
    # (See design/sqrt_nan.py for a speed comparison of approaches.)
    return where(n < 0.0, nan, sqrt(abs(n)))


def dots(v, u):
    """
    Return dot product of u, v; works with multiple vectors.
    Works whether u, v have shape (3,), or whole arrays of corresponding
        x, y, and z coordinates and have shape (3, N).
    """
    return (v * u).sum(axis=0)


def T(M):
    """Swap the first two dimensions of an array."""
    return rollaxis(M, 1)


def mxv(M, v):
    """Matrix times vector: multiply an NxN matrix by a vector."""
    return einsum("ij...,j...->i...", M, v)


def mxm(M1, M2):
    """Matrix times matrix: multiply two NxN matrices."""
    return einsum("ij...,jk...->ik...", M1, M2)


def mxmxm(M1, M2, M3):
    """Matrix times matrix times matrix: multiply 3 NxN matrices together."""
    return einsum("ij...,jk...,kl...->il...", M1, M2, M3)


_T, _mxv, _mxm, _mxmxm = T, mxv, mxm, mxmxm  # In case anyone imported old name


def length_of(xyz):
    """
    Given a 3-element array |xyz|, return its length.
    The three elements can be simple scalars, or the array can be two
        dimensions and offer three whole series of x, y, and z coordinates.
    For small arrays (<1000) this tends to be much faster than np.linalg.norm().
        See for your self with "%timeit np.linalg.norm(array)"
    """
    return sqrt((xyz * xyz).sum(axis=0))


def angle_between(u, v):
    """
    Angle between vectors u, v in radians.
    Works whether u, v each have the shape (3,), or are whole arrays of
        corresponding x, y, and z coordinates with shape (3, N).
    Returned angle will be between 0 and TAU/2.

    This formula is from Section 12 of:
    https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf

    """
    a = u * length_of(v)
    b = v * length_of(u)
    return 2.0 * arctan2(length_of(a - b), length_of(a + b))


def to_spherical(xyz):
    """Convert |xyz| to spherical coordinates (r,theta,phi).

    ``r`` - vector length
    ``theta`` - angle above (+) or below (-) the xy-plane
    ``phi`` - angle around the z-axis

    Note that ``theta`` is an elevation angle measured up and down from
    the xy-plane, not a polar angle measured from the z-axis, to match
    the convention for both latitude and declination.

    """
    r = length_of(xyz)
    x, y, z = xyz
    theta = arcsin(z / (r + _AVOID_DIVIDE_BY_ZERO))
    phi = arctan2(y, x) % TAU
    return r, theta, phi


def _to_spherical_and_rates(r, v):
    # Convert Cartesian rate and velocity vectors to angles and rates.
    x, y, z = r
    xdot, ydot, zdot = v

    length = length_of(r)
    lat = arcsin(z / (length + _AVOID_DIVIDE_BY_ZERO))
    lon = arctan2(y, x) % TAU
    range_rate = dots(r, v) / length_of(r)

    x2 = x * x
    y2 = y * y
    x2_plus_y2 = x2 + y2 + _AVOID_DIVIDE_BY_ZERO
    lat_rate = (x2_plus_y2 * zdot - z * (x * xdot + y * ydot)) / (
        (x2_plus_y2 + z * z) * sqrt(x2_plus_y2)
    )
    lon_rate = (x * ydot - xdot * y) / x2_plus_y2

    return length, lat, lon, range_rate, lat_rate, lon_rate


def from_spherical(r, theta, phi):
    """Convert (r,theta,phi) to Cartesian coordinates |xyz|.

    ``r`` - vector length
    ``theta`` - angle in radians above (+) or below (-) the xy-plane
    ``phi`` - angle in radians around the z-axis

    Note that ``theta`` is an elevation angle measured up and down from
    the xy-plane, not a polar angle measured from the z-axis, to match
    the convention for both latitude and declination.

    """
    rxy = r * cos(theta)
    return array((rxy * cos(phi), rxy * sin(phi), r * sin(theta)))


# Support users who might have imported these under their old names.
# I'm not sure why I called what are clearly spherical coordinates "polar".
to_polar = to_spherical
from_polar = from_spherical


def rot_x(theta):
    c = cos(theta)
    s = sin(theta)
    zero = theta * 0.0
    one = zero + 1.0
    return array(((one, zero, zero), (zero, c, -s), (zero, s, c)))


def rot_y(theta):
    c = cos(theta)
    s = sin(theta)
    zero = theta * 0.0
    one = zero + 1.0
    return array(((c, zero, s), (zero, one, zero), (-s, zero, c)))


def rot_z(theta):
    c = cos(theta)
    s = sin(theta)
    zero = theta * 0.0
    one = zero + 1.0
    return array(((c, -s, zero), (s, c, zero), (zero, zero, one)))


def angular_velocity_matrix(angular_velocity_vector):
    x, y, z = angular_velocity_vector
    zero = x * 0.0
    return array(((zero, -z, y), (z, zero, -x), (-y, x, zero)))


def _to_array(value):
    """
    Convert plain Python sequences into NumPy arrays.
    This lets users pass plain old Python lists and tuples to Skyfield,
        instead of always having to remember to build NumPy arrays.  We pass
        any kind of generic sequence to the NumPy ``array()`` constructor
        and wrap any other kind of value in a NumPy ``float64`` object.
    """
    if hasattr(value, "shape"):
        return value
    elif hasattr(value, "__len__"):
        return array(value)
    else:
        return float64(value)


def _reconcile(a, b):
    """Coerce two NumPy generics-or-arrays to the same number of dimensions."""
    an = getattr(a, "ndim", 0)
    bn = getattr(b, "ndim", 0)
    difference = bn - an
    if difference > 0:
        if an:
            a.shape += (1,) * difference
        else:
            a = full_like(b, a)
    elif difference < 0:
        if bn:
            b.shape += (1,) * -difference
        else:
            b = full_like(a, b)
    return a, b


try:
    from io import BytesIO
except:
    from StringIO import StringIO as BytesIO


def load_bundled_npy(filename):
    """Load a binary NumPy array file that is bundled with Skyfield."""
    data = get_data("skyfield", "data/{0}".format(filename))
    return load(BytesIO(data))


# ********** Skyfield functions above ************************************
# ********** Curtis [3] or [9] functions below **************************************
def swap_columns(lst, col1, col2):
    """
    Swap columns of python list.
    expect (1) single row list, or (2) 2D list
    Input Parameters:
    ----------
        lst        : python list
        col1, col2 : int, move col2->col1 & col1->col2
    """
    if not type(lst) == list:
        print("** NOT a list, to swap_columns: **")
        raise NameError("swap_columns() needs a python list, not array.")
    shp = np.asarray(lst).shape  # python does not have a shape for list!
    # print(f"list rows, {len(lst)}") # uncomment for troubleshooting
    # print(f"list dim, {len(shp)}")

    if len(shp) == 1:  # one dimension, i.e. 1 row
        lst[col2], lst[col1] = lst[col1], lst[col2]
    elif len(shp) == 2:  # assume two dimension
        for row in lst:
            row[col1], row[col2] = row[col2], row[col1]
    else:
        raise NameError("swap_columns() found > 2D list.")
    return lst  # swap_columns()


def lambert(mu: float, R1, R2, tof: float, prograde=True):
    """
    Lambert solver, Curtis chapter 5.3, pp.263.  Algorithm 5.2, p270, and
        example 5.2 pp.270.
    2024-09-06, not yet got this to work; not sure I want to spend the time, now.

    Input Parameters:
    ----------
        mu       : float, description
        R1       : np.array, description
        R2       : np.array, description
        tof      : float, time of flight
        prograde : bool, optional, default=True

    Returns:
    -------
        v1_vec, v2_vec : description
    Notes:
    ----------
        The function Lambert_v1v2_solver() differs from this function, Lambert(),
            by using scipy.optimize.fsolve() to iterate to a solution.
    """
    # step 1. magnitudes of R1 and R2; eqn 5.24, p264.
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    # step 2. eqn choice based on prograde or retrograde
    c12 = np.cross(R1, R2)
    theta = np.arccos(np.dot(R1, R2) / (r1 * r2))

    # Determine whether the orbit is prograde or retrograde:
    if prograde:
        if c12[2] <= 0:
            theta = 2 * np.pi - theta
    else:
        if c12[2] >= 0:
            theta = 2 * np.pi - theta

    # step 3, equation 5.35:
    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))

    # determine approximately where F(z,tof) changes sign, and
    # use that value of z as the starting value for Equation 5.45:
    z = -100
    while F(z=z, tof=tof, mu=mu) < 0:
        z = z + 0.1

    # Set an error tolerance and a limit on the number of iterations:
    tol = 1e-8
    nmax = 5000

    # iterate on Equation 5.45 until z is determined to within the
    # error tolerance:
    ratio = 1
    n = 0
    while abs(ratio) > tol and n <= nmax:
        n = n + 1
        ratio = F(z=z, tof=tof, mu=mu) / dFdz(z)
        z = z - ratio

    # Report if the maximum number of iterations is exceeded:
    if n >= nmax:
        print(f"\n **Exceeded Lambert iterations {nmax} \n")

    # Equation 5.46a:
    f = 1 - y(z) / r1

    # Equation 5.46b:
    g = A * np.sqrt(y(z) / mu)

    # Equation 5.46d:
    gdot = 1 - y(z) / r2

    # Equation 5.28:
    V1 = 1 / g * (R2 - f * R1)

    # Equation 5.29:
    V2 = 1 / g * (gdot * R2 - R1)

    return V1, V2


def Lambert_v1v2_solver(r1_v, r2_v, dt, mu, prograde=True):
    """
    Lambert solver, Curtis chapter 5.3, pp.263.  Algorithm 5.2, p270, and
        pp.270, Example 5.2, appendix 5.2, copied from example5_x.py

    Input Parameters:
    ----------
        r1_v     : numpy.array, initial position vector
        r2_v     : numpy.array, final position vector
        dt       : float, tof (time-of-flight)
        mu       : float
        prograde : bool, optional, default=True

    Returns
    -------
        v1_vec   : numpy.array,
        v2_vec   : numpy.array,
    """
    # inspired by Curtis example 5.2
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

    A = a_lambert(r1, r2, d_theta)
    # set the starting estimate for Lambert solver
    z = scipy.optimize.fsolve(lambert_zerosolver, x0=1.5, args=[dt, mu, r1, r2, A])[0]
    y = y_lambert(z, r1, r2, A)

    f_dt = find_f_y(y, r1)
    g_dt = find_g_y(y, A, mu)
    f_dot_dt = find_f_dot_y(y, r1, r2, mu, z)
    g_dot_dt = find_g_dot_y(y, r2)

    v1_vec = (1 / g_dt) * (r2_v - f_dt * r1_v)
    v2_vec = (g_dot_dt / g_dt) * r2_v - (
        (f_dt * g_dot_dt - f_dot_dt * g_dt) / g_dt
    ) * r1_v
    return v1_vec, v2_vec


def y_lambert(z, r1, r2, A):
    """inspired by Curtis example 5.2"""
    K = (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))
    return r1 + r2 + A * K


def a_lambert(r1, r2, d_theta):
    """inspired by Curtis example 5.2"""
    K1 = np.sin(d_theta)
    K2 = np.sqrt((r1 * r2) / (1 - np.cos(d_theta)))
    return K1 * K2


def lambert_zerosolver(z, args):
    """inspired by Curtis example 5.2"""
    dt, mu, r1, r2, A = args
    K1 = ((y_lambert(z, r1, r2, A) / stumpff_C(z)) ** 1.5) * stumpff_S(z)
    K2 = A * np.sqrt(y_lambert(z, r1, r2, A))
    K3 = -1 * dt * np.sqrt(mu)
    return K1 + K2 + K3


def find_f_y(y, r1):
    # inspired by Curtis example 5.2
    return 1 - y / r1


def find_g_y(y, A, mu):
    # inspired by Curtis example 5.2
    return A * np.sqrt(y / mu)


def find_f_dot_y(y, r1, r2, mu, z):
    # inspired by Curtis example 5.2
    K1 = np.sqrt(mu) / (r1 * r2)
    K2 = np.sqrt(y / stumpff_C(z))
    K3 = z * stumpff_S(z) - 1
    return K1 * K2 * K3


def find_g_dot_y(y, r2):
    # inspired by Curtis example 5.2
    return 1 - y / r2


# Equation 5.38:
def y(z, r1, r2, A):
    return r1 + r2 + A * (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))


# Equation 5.40:
def F(z, tof, mu):
    return (
        (y(z) / stumpff_C(z)) ** 1.5 * stumpff_S(z)
        + A * np.sqrt(y(z))
        - np.sqrt(mu) * tof
    )


# Equation 5.43:
def dFdz(z):
    if z == 0:
        return np.sqrt(2) / 40 * y(0) ** 1.5 + A / 8 * (
            np.sqrt(y(0)) + A * np.sqrt(1 / (2 * y(0)))
        )
    else:
        return (y(z) / stumpff_C(z)) ** 1.5 * (
            1 / (2 * z) * (stumpff_C(z) - 3 * S(z) / (2 * stumpff_C(z)))
            + 3 * stumpff_S(z) ** 2 / (4 * stumpff_C(z))
        ) + A / 8 * (
            3 * S(z) / stumpff_C(z) * np.sqrt(y(z)) + A * np.sqrt(stumpff_C(z) / y(z))
        )


def depart_a(depart, arrival, cb_mu):
    """
    Earth_a->Mars_b, depart Earth.  Curtis [3] pp.446, example 8.4.
    Based on curtis_ex8_4_depart()
    Given:
        Earth orbit launch, from alt=300 [km] circular, hyperbolic launch trajectory;
            thus ecc=1, and Earth GM (or mu)
        r1: periapsis altitude 500 [km];
        r2: earth-sun SOI (sphere of influence)

    Find:
        (a) delta-v required
        (b) departure hyperbola perigee location
    """
    # departure planet 1 parameter list
    r1, rp1, rp1_alt, rp1_mu = depart
    # arrival planet 2 parameter list; rp2 variables not used
    r2, rp2, rp2_alt, rp2_mu = arrival

    # *****************************************************
    # Curtis [3] p.442, eqn 8.35
    v_inf = math.sqrt(cb_mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
    # spacecraft speed in circular parking orbit; Curtis p.444, eqn 8.41
    v_c = math.sqrt(rp1_mu / (rp1 + rp1_alt))
    # Delta_v required to enter departure hyperbola; eqn 8.42, p444
    delta_v = v_c * (math.sqrt(2 + (v_inf / v_c) ** 2) - 1)
    # eqn 8.43, p444
    r_p = rp1 + rp1_alt  # periapsis
    beta_depart = math.acos(1 / (1 + r_p * v_inf**2 / rp1_mu))
    ecc_depart = 1 + (r_p * v_inf**2) / rp1_mu

    # print(f"depart v_infinity, v_inf = {v_inf:.6g} [km/s]")
    # print(f"departure parking orbit, v_c= {v_c:.6g} [km/s]")
    # print(f"delta_v to enter departure hyperbola = {delta_v:.6g} [km/s]")
    # print(f"departure hyperbola beta angle= {beta_depart*180/math.pi:.6g} [deg]")
    # print(f"eccentricity, departure hyperbola = {ecc_depart:.6g}")

    return v_inf, v_c, delta_v, beta_depart, ecc_depart


def arrive_b(depart, arrive, cb_mu, p2_sat_T):
    """
    body1 (Earth_a) -> body2 (Mars_b), arrive at Mars.
        Related to Curtis [3] pp.456, example 8.5.
    After Hohmann transfer calculate arrival parameters, assuming satellite orbit period

    Input Parameters:
    ----------
        body1 departure list:

        body2 arrival list:

        central body:

        body2 satellite period:
    Return:
    ----------
        minimum delta_v
        satellite period
        periapsis radius
        aiming radius
        angle between periapse and body2 velocity vector

    Notes:
    ----------
        May help development; see https://github.com/jkloser/OrbitalMechanics
        Helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    # departure planet 1 parameter list
    r1, rp1, rp1_alt, rp1_mu = depart
    # arrive planet 2 parameter list
    r2, rp2, rp2_alt, rp2_mu, p2_sat_T = arrive

    # Curtis [3] eqn 8.4
    v_inf = math.sqrt(cb_mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
    # Semi-major axis of capture orbit, eqn 2.83
    a_capture = (p2_sat_T * math.sqrt(rp2_mu) / (2 * math.pi)) ** (2 / 3)
    # from eqn 8.67
    rp2_ecc = (2 * rp2_mu / (a_capture * v_inf**2)) - 1
    # from eqn 8.70
    delta_v = v_inf * math.sqrt((1 - rp2_ecc) / 2)
    # periapsis radius at mars capture, from eqn 8.67
    r_p = (2 * rp2_mu / v_inf**2) * ((1 - rp2_ecc) / (1 + rp2_ecc))
    # aiming radius from eqn 8.71
    aim_radius = r_p * math.sqrt(2 / (1 - rp2_ecc))
    # angle to periapsis from eqn 8.43
    beta_p = math.acos(1 / (1 + r_p * v_inf**2 / rp2_mu))

    # print(f"arrive v_infinity, v_inf = {v_inf:.5g} [km/s]")
    # print(f"arrive semi-major axis = {a_capture:.5g} [km]")
    # print(f"eccentricity, at mars = {rp2_ecc:.5g}")
    # print(f"delta_v enter mars = {delta_v:.5g} [km/s]")
    # print(f"periapsis at mars, r_p = {r_p:.5g} [km]")
    # print(f"aiming radius (aka delta) at mars = {aim_radius:.5g} [km]")
    # print(f"angle to periapsis at mars = {(beta_p*180/math.pi):.5g} [deg]")

    return v_inf, a_capture, rp2_ecc, delta_v, r_p, aim_radius, beta_p


def flyby(depart, arrive, cb_mu):
    """
    **** 2024-11-07 need to finish translating to general function ****

    Earth->Venus fly-by.  Curtis [3] pp.462, example 8.6.
        Spacecraft departs earth with a velocity perpendicular to the sun line.
        Encounter occurs at a true anomaly in the approach trajectory of 30[deg].
        Periapse altitude 300 km.
    (a) Dark side Venus apporach.
            Post fly-by orbit shown in Figure 8.20.
    (b) Sunlit side Venus approach.
            Post fly-by orbit shown in Figure 8.21.

    Leading-side flyby results in a decrease in the spacecraft's heliocentric speed.
    Trailing-side flyby increases helliocentric speed;
        e1, h1, and θ1 are eccentricity, angular momentum,
        and true anomaly of heliocentric approach trajectory.

    TODO update variable names to be consistant with darkside & lightside calculations
    Input Parameters:
    ----------
        TODO update
    Return:
    ----------
        TODO update
    Notes:
    ----------
        helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    """
    # departure planet 1 parameter list
    r1, rp1, rp1_alt, rp1_mu = depart
    # arrive planet 2 parameter list
    r2, rp2, rp2_alt, rp2_mu, TA_t = arrive

    # part a, Pre-Flyby ellipse; p.462+
    # orbit id (1), eccentricity of transfer orbit; p.464
    ecc1_t = (r1 - r2) / (r1 + r2 * math.cos(TA_t))
    # orbit 1 angular momentum; p.464
    h1 = math.sqrt(cb_mu * r1 * (1 - ecc1_t))
    # Calculate spacecraft radial and transverse components heliocentric velocity at
    # the inbound crossing of Venus’s sphere of influence.
    v1_perp = h1 / r2  # perpendicular velocity orbit 1[km/s]
    # radial velocity orbit 1[km/s]
    v1_radi = (cb_mu / h1) * (ecc1_t) * math.sin(TA_t)
    # flight path angle; p.464; eqn 2.51 on p.xx
    # a negative sign is consistent with the spacecraft flying towards
    #   perihelion of the pre-flyby elliptical trajectory (orbit 1).
    gamma1 = math.atan(v1_radi / v1_perp)
    # Speed of the space vehicle at the inbound crossing
    v_in = math.sqrt(v1_perp**2 + v1_radi**2)

    np.set_printoptions(precision=4)  # numpy, set vector printing size
    rad2deg = 180 / math.pi
    print(f"eccentricity, at venus, ecc1_t = {ecc1_t:.5g}")
    print(f"angular momentum, orbit1, h1 = {h1:.5g} [km^2/s]")
    print(f"velocity inbound perpendicular, v1_perp = {v1_perp:.5g} [km/s]")
    print(f"velocity inbound radial, v1_radi = {v1_radi:.5g} [km/s]")
    print(f"flight path angle, gamma1 = {gamma1*rad2deg:.5g} [deg]")
    print(f"velocity inbound (from SOI) = {v_in:.5g} [km/s]")

    # part a, Flyby Hyperbola; p.464+
    print("\n********** darkside flyby hyperbola **********")
    # velocity inbound (1) vector, planet, sun direction coordinates
    v1p_vec = np.array([v1_perp, -v1_radi])  # [km/s]
    v1p_mag = np.linalg.norm(v1p_vec)  # [km/s]
    # assume venus in circular orbit; velocity planet (venus) relative vector
    vp_vec = np.array([math.sqrt(cb_mu / r2), 0])  # [km/s]
    # p.465
    v1_infty_vec = v1p_vec - vp_vec
    v1_infty = np.linalg.norm(v1_infty_vec)
    # hyperbola periapsis radius; p.465
    rp2_p = rp2 + rp2_alt  # [km]
    # planetcentric angular momentum & eccentricity; eqns 8.39, 8.39; p.465
    h2 = rp2_p * math.sqrt(v1_infty**2 + 2 * rp2_mu / rp2_p)
    ecc1 = 1 + (rp2_p * v1_infty**2) / rp2_mu
    # turn angle and true anomaly of asymptote
    delta_turn1 = 2 * math.asin(1 / ecc1)
    nu_asym = math.acos(-1 / ecc1)
    # aiming radius; p.465; eqns. 2.50, 2.103, 2.107
    delta_aim = rp2_p * math.sqrt((ecc1 + 1) / (ecc1 - 1))
    # angle between v1_infty and v_venus; p.465
    phi1 = math.atan(v1_infty_vec[1] / v1_infty_vec[0])
    print(f"velocity inbound vector, v1p_vec = {v1p_vec} [km/s]")
    print(f"velocity planet vector, vp_vec = {vp_vec} [km/s]")
    print(f"velocity, inbound from infinity, v1_infty_vec = {v1_infty_vec} [km/s]")
    print(f"velocity inbound, v1_infty = {v1_infty:.5g} [km/s]")
    print(f"angular momentum, h2 = {h2:.5g} [km^2/s]")
    print(f"eccentricity, inbound, ecc1_venus = {ecc1:.5g} [km^2/s]")
    print(f"turn angle inbound, delta_turn1 = {delta_turn1*rad2deg:.5g} [deg]")
    print(f"true anomaly of asymptote, nu_asym = {nu_asym*rad2deg:.5g} [deg]")
    print(f"aiming radius, delta_aim = {delta_aim:.5g} [km]")
    print(f"true anomaly of asymptote, inbound, phi1 = {phi1*rad2deg:.5g} [deg]")

    # part a, Dark Side Approach; p.466
    # There are two flyby approaches:
    # (1) Dark side approach, the turn angle is counterclockwise (+102.9◦)
    # (2) Sunlit side approach, the turn anble is clockwise (−102.9◦).

    # angle between v_infty & V_venus_vec, darkside turn; eqn 8.85; p.466
    phi2 = phi1 + delta_turn1
    # eqn 8.86; p.466
    v2_infty_vec = v1_infty * np.array([math.cos(phi2), math.sin(phi2)])  # [km/s]
    # outbound velocity vector, planet, sun direction coordinates; p.466
    v2p_vec = vp_vec + v2_infty_vec  # [km/s]
    v2p_mag = np.linalg.norm(v2p_vec)
    # part a, Post Flyby Ellipse (orbit 2) for Darkside Approach; p.467
    # The heliocentric post flyby trajectory, orbit 2.
    # Angular momentum orbit 2; eqn 8.90.
    ho2 = r2 * v2p_vec[0]
    ecc_cos = (ho2**2 / (cb_mu * r2)) - 1
    ecc_sin = -v2p_vec[1] * ho2 / (cb_mu)
    ecc_tan = ecc_sin / ecc_cos
    theta2 = math.atan(ecc_tan)

    print(f"darkside turn angle, phi2 = {phi2*rad2deg:.5g} [deg]")
    print(f"darkside velocity infinity, v2_infty_vec = {v2_infty_vec} [km/s]")
    print(f"outbound velocity vector, v2p_vec = {v2p_vec} [km/s]")
    print(f"outbound crossing velocity, v2p_mag = {v2p_mag:.5g} [km/s]")
    print(f"compare darkside inbound/outbound speeds: {(v2p_mag-v1p_mag):.5g} [km/s]")
    print(f"angular momentum, orbit 2, ho2 = {ho2:.5g} [km/s]")
    print(f"interium, ecc_cos = {ecc_cos:.5g}")
    print(f"interium, ecc_sin = {ecc_sin:.5g}")
    print(f"interium, ecc_tan = {ecc_tan:.5g}")
    print(f"theta2, 1st possibility = {theta2*180/math.pi:.5g} [deg]")
    print(f"theta2, 2nd possibility = {(theta2*180/math.pi)+180:.5g} [deg]")

    # based on cos and sin quadrants select angle
    if (ecc_cos < 0 and ecc_sin < 0) or (ecc_cos > 0 and ecc_sin < 0):
        # quadrant 3 or quadrant 4
        theta2 = theta2 + math.pi
        print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")
    else:
        print(f"choose theta2; quadrant test: {theta2*180/math.pi:.5g} [deg]")
    ecc2 = ecc_cos / math.cos(theta2)
    r2_perihelion = (ho2**2 / cb_mu) * (1 / (1 + ecc2))

    print(f"perihelion of departure, theta2 = {theta2*180/math.pi:.5g} [deg]")
    print(f"eccentricity, orbit 2, ecc2_venus = {ecc2:.5g}")
    print(f"radius orbit2, perihelion, r2_perihelion = {r2_perihelion:.5g}")

    # part b, Sunlit side approach; p.467+
    print("\n********** sunlit approach **********")
    # angle lightside, v_infty & V_venus_vec, outbound crossing; p.467
    phi2 = phi1 - delta_turn1
    # velocity 2 lightside vector; p.468
    v2l_infty_vec = v1_infty * np.array([math.cos(phi2), math.sin(phi2)])  # [km/s]
    # velocity outbound lightside vector, planet, sun direction coordinates; p.468
    v2pl_vec = vp_vec + v2l_infty_vec  # [km/s]
    v2pl_mag = np.linalg.norm(v2pl_vec)

    print(f"lightside turn angle, phi2 = {phi2*180/math.pi:.5g} [deg]")
    print(f"lightside velocity infinity, v2l_infty_vec = {v2l_infty_vec} [km/s]")
    print(f"outbound velocity vector lightside, v2pl_vec = {v2pl_vec} [km/s]")
    print(f"outbound crossing velocity lightside, v2pl_mag = {v2pl_mag:.5g} [km/s]")
    print(f"compare lightside inbound/outbound speeds: {(v2pl_mag-v1p_mag):.5g} [km/s]")

    print("\n********** post flyby ellipse **********")
    # Angular momentum lightside orbit 2; eqn 8.90.
    h_lo2 = r2 * v2pl_vec[0]
    ecc_cos = (h_lo2**2 / (cb_mu * r2)) - 1
    ecc_sin = -v2pl_vec[1] * h_lo2 / cb_mu
    ecc_tan = ecc_sin / ecc_cos
    theta2 = math.atan(ecc_tan)

    print(f"angular momentum, lightside orbit 2, ho2 = {h_lo2:.5g} [km/s]")
    print(f"interium, ecc_cos = {ecc_cos:.5g}")
    print(f"interium, ecc_sin = {ecc_sin:.5g}")
    print(f"interium, ecc_tan = {ecc_tan:.5g}")
    print(f"theta2, 1st possibility = {theta2*rad2deg:.5g} [deg]")
    print(f"theta2, 2nd possibility = {(theta2*rad2deg)+180:.5g} [deg]")

    # based on cos and sin quadrants select angle
    if (ecc_cos < 0 and ecc_sin < 0) or (ecc_cos > 0 and ecc_sin < 0):
        # quadrant 3 or quadrant 4
        theta2 = theta2 + math.pi
        print(f"choose theta2; quadrant test: {theta2*rad2deg:.5g} [deg]")
    else:
        print(f"choose theta2; quadrant test: {theta2*rad2deg:.5g} [deg]")

    ecc2 = ecc_cos / math.cos(theta2)
    r2_perihelion = (h_lo2**2 / cb_mu) * (1 / (1 + ecc2))
    print(f"departure perihelion, lightside, theta2 = {theta2*rad2deg:.5g} [deg]")
    print(f"eccentricity, orbit 2, ecc2_venus = {ecc2:.5g}")
    print(f"radius orbit2, perihelion lightside, r2_perihelion = {r2_perihelion:.5g}")

    return


def sphere_of_influence(R: float, mass1: float, mass2: float):
    """
    Radius of the SOI (sphere of influence)

    Input Parameters:
    ----------
        R     : float, distance between mass1 , mass2.
                    for earth, R~= sma (semi-major axis)
        mass1 : float, generally the smaller of the 2 mass's (i.e. planet)
        mass2 : float, generally the larger of the 2 mass's (i.e. sun)
    Returns:
    -------
        r_SOI : float, radius of SOI (sphere of influence)
    """
    r_SOI = R * (mass1 / mass2) ** (2 / 5)
    return r_SOI


def E_zerosolver(E, args):
    Me = args[0]
    ecc = args[1]
    return E - ecc * np.sin(E) - Me


def solve_for_E(Me: float, ecc: float):
    """
    Solve Keplers equation.
    Note, Curtis, p.163, algorithm 3.1

    Parameters:
    ----------
        Me : float, mean angle/anomaly [rad]
        ecc : float, eccentricity

    Return:
    -------
        sols : float, E [rad]
    """
    # iterative solution process
    sols = scipy.optimize.fsolve(E_zerosolver, x0=Me, args=[Me, ecc])[0]
    return sols


def planet_elements_and_sv(planet_id, year, month, day, hour, minute, second, mu):
    """
    Curtis [3] pp.470, section 8.10; p.471-472, algorithm 8.1.; pp.473, example 8.7
        *** Depricated, 2024-August, instead use rv_from_date() ***
    Corroborate ephemeris with JPL Horizons;
        https://ssd.jpl.nasa.gov/horizons/app.html#/

    Parameters:
    ----------
        planet_id, year, month, day, hour, minute, second, mu
    Return:
    -------

    References: (see references.py for references list)
    """
    deg = math.pi / 180  # conversion [rad]->[deg]

    # Vallado [2] equivilent of Curtis p.276, eqn 5.48:
    # parameters of julian_date(yr, mo, d, hr, minute, sec, leap_sec=False)
    jd = julian_date(yr=year, mo=month, d=day, hr=hour, minute=minute, sec=second)

    # Planetary ephemeris pp.470, section 8.10; data, p.472, tbl 8.1.
    j2000_coe, rates = planetary_elements(planet_id)

    # Curtis [3], p.471, eqn 8.93a
    t0 = (jd - 2451545) / 36525
    # Curtis p.471, eqn 8.93b
    elements = j2000_coe + rates * t0

    a = elements[0]
    e = elements[1]

    # Curtis [3], p.89, eqn 2.71
    h = math.sqrt(mu * a * (1 - e**2))

    # Reduce the angular elements to range 0 - 360 [deg]
    incl = elements[2]
    RA = elements[3]  # [deg]
    w_hat = elements[4]  # [deg]
    L = elements[5]  # [deg]
    w = w_hat - RA  # [deg]
    M = L - w_hat  # [deg]

    # Curtis [3], p.163, algorithm 3.1 (M [rad]) in example 3.1
    # E = kepler_E(e, M * deg)  # [rad]
    E = solve_for_E(ecc=e, Me=M * deg)  # [rad]

    # Curtis [3], p.160, eqn 3.13
    TA = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))  # [rad]

    coe = [h, e, RA * deg, incl * deg, w * deg, TA * deg]

    # Curtis [3], p.231, algorithm 4.5; see p. 232, example 4.7
    r, v = sv_from_coe(coe, mu)

    return coe, r, v, jd


def get_transfer_angle(r1, r2, prograde=True):
    """
    Find transfer angle, given r1, r2, sense of rotation (long or short)
    2024-08-13 Copied from LambertHub angles.py

    Parameters
    ----------
    r1: np.array
        Initial position vector.
    r2: np.array
        Final position vector.
    prograde: bool
        If True, prograde motion, otherwise retrograde motion.

    Return
    -------
    dtheta: float, angle between vectors [rad]
    """
    # Verify position vectors are collinear. If so, verify the transfer
    # angle is 0 or pi.
    if np.all(np.cross(r1, r2) == 0):
        return 0 if np.all(np.sign(r1) == np.sign(r2)) else np.pi

    # Solve for a unitary vector normal to the vector plane. Its direction and
    # sense the one given by the cross product (right-hand) from r1 to r2.
    h = np.cross(r1, r2) / norm(np.cross(r1, r2))

    # Compute the projection of the normal vector onto the reference plane.
    alpha = np.dot(np.array([0, 0, 1]), h)

    # Get the minimum angle (0 <= dtheta <= pi) between r1 and r2.
    r1_norm, r2_norm = [norm(vec) for vec in [r1, r2]]
    theta0 = np.arccos(np.dot(r1, r2) / (r1_norm * r2_norm))

    # Fix theta as needed
    if prograde is True:
        dtheta = theta0 if alpha > 0 else 2 * np.pi - theta0
    else:
        dtheta = theta0 if alpha < 0 else 2 * np.pi - theta0

    return dtheta


def planetary_elements(planet_id, d_set=1):
    """
    Curtis planetary elements includes Pluto; heliocentric yes; equatorial yes.
    Horizons planetary elements excludes Pluto; heliocentric yes; equatorial NO.
    2025-04-22 made this function units-aware; cost is complexity.
        User optionally chooses elements data set.
        NOTE the Horizons table lists element IN A DIFFERENT ORDER THAN Curtis [3] !!
        NOTE Curtis table equatorial, Horizons table is ecliptic !!
        This routine aligns JPL and Curtis [3] elements; swap_columns()
    TODO: done 1) below 2025-04-22
        1) change j2000_elements[] & cent_rates[] from lists to arrays;
            numeric arrays are much more memory and processing efficient.
        2) verify changing 1) does not impact all the calling routines

    Input Parameters:
    ----------
        planet_id : int, for JPL 1->8, Mercury->Neptune; for Curtis 1->9
        d_set     : int, planet elements data set
                    0=JPL Horizons, Table 1, Keplerian Elements and Rates
                    1= Curtis [3] table 8.1, p.472

    Returns (for planet_id input):
    ----------
        j2000_coe   : nparray, [km & deg]
                    j2000 clasic orbital elements (Kepler).
        j2000_rates : nparray, [km/cent & deg/cent]
                    coe rate change (x/century) from 2000-01-01.
    Notes:
    ----------
        Element list output:
            sma   = [km] semi-major axis (aka a)
            ecc   = [--] eccentricity
            incl  = [deg] inclination angle; to the ecliptic
            raan  = [deg] right ascension of ascending node (aka capital W)
                    longitude node
            w_bar = [deg] longitude of periapsis
            L     = [deg] mean longitude
        Alternative element list names output:
            a, e, i_ecl, omega_ecl, argp_ecl, M_ecl

    References:
    ----------
    See references.py for references list.
    """
    if d_set == 0:  # data set JPL Horizons Table 1
        # fmt: off
        # JPL Horizons table 1, Keplerian Elements and Rates; EXCLUDES Pluto.
        # Horizons: mean ecliptic and equinox of j2000; time-interval 1800 AD->2050 AD.
        # NOTE Horizons elements lists IN A DIFFERENT ORDER THAN Curtis [3] !!
        # Below is a table copy, including column order from;
        #   https://ssd.jpl.nasa.gov/planets/approx_pos.html
        #
        #   JPL Table 1 order of the elements is different then the other list below.
        #   Also note, Table 1 list earth-moon barycenter, not just earth.
        #           sma   |    ecc      |     inc     | mean.long   | long.peri   |raan, Omega
        #       au, au/cy | ecc, ecc/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy
        j2000_elements = np.array(
            [
                [0.38709927, 0.20563593, 7.00497902, 252.25032350,  77.45779628,  48.33076593],
                [0.72333566, 0.00677672,  3.39467605, 181.97909950, 131.60246718, 76.67984255],
                [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193,  0.0],
                [1.52371034, 0.09339410,  1.84969142,  -4.55343205, -23.94362959, 49.55953891],
                [5.20288700, 0.04838624,  1.30439695,  34.39644051,  14.72847983, 100.47390909],
                [9.53667594, 0.05386179,  2.48599187,  49.95424423,  92.59887831, 113.66242448],
                [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630,  74.01692503],
                [30.06992276, 0.00859048, 1.77004347, -55.12002969,  44.96476227, 131.78422574]
            ], dtype=object
        )
        cent_rates = np.array(
            [
                [0.00000037,  0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
                [0.00000390, -0.00004107, -0.00078890,  58517.81538729, 0.00268329, -0.27769418],
                [0.00000562, -0.00004392, -0.01294668,  35999.37244981, 0.32327364,  0.0],
                [0.00001847,  0.00007882, -0.00813131,  19140.30268499, 0.44441088, -0.29257343],
                [-0.00011607, -0.00013253, -0.00183714, 3034.74612775,  0.21252668,  0.20469106],
                [-0.00125060, -0.00050991,  0.00193609, 1222.49362201, -0.41897216, -0.28867794],
                [-0.00196176, -0.00004397, -0.00242939,  428.48202785,  0.40805281,  0.04240589],
                [0.00026291,  0.00005105,  0.00035372,  218.45945325, -0.32241464, -0.00508664]
            ], dtype=object
        )
        # align JPL table with Curtis [3] table 8.1, swap columns
        j2000_elements[:,[2,4]] = j2000_elements[:,[2, 4]] # columns 5->3, and 3->5
        cent_rates[:,[2,4]] = cent_rates[:,[2, 4]] # columns 5->3, and 3->5

    if d_set == 1:  # data set Curtis [3] Table 8.1
        # fmt: off
        # Data below, copied Curtis [3] tbl 8.1, Standish et.al. 1992
        # Elements, an ndarray of dtype object since columns have units
        # semi-major axis|           |            |RAAN, Omega| omega_bar  | L
        #          sma   |   ecc     |    incl    |long.node  | long.peri  |mean.long
        #      au, au/cy |ecc, ecc/cy|deg, deg/cy |deg, deg/cy|deg, deg/cy |deg, deg/cy
        j2000_elements = np.array(
            [
                [0.38709893, 0.20563069, 7.00487, 48.33167, 77.4545, 252.25084],
                [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
                [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
                [1.52366231, 0.09341233, 1.845061, 49.57854, 336.04084, 355.45332],
                [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
                [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
                [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
                [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
                [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676,238.92881]
            ], dtype=object
        )
        # century [cy] rates, python list:
        # Data below, copied Curtis tbl 8.1, Standish et.al. 1992
        # Units of rates table: cy=century
        # "au/cy", "1/cy", "arc-sec/cy", "arc-sec/cy", "arc-sec/cy", "arc-sec/cy"
        # fmt: off
        cent_rates = np.array(
            [
                [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],
                [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
                [-0.0000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
                [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
                [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
                [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
                [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
                [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
                [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90],
            ], dtype=object
        )
    # fmt: on
    # extract user requested planet coe data & rates;
    #   reminder, coe=classic orbital elements (Kepler)
    j2000_coe = j2000_elements[planet_id - 1, :]
    j2000_rates = cent_rates[planet_id - 1, :]
    # note, some constants from Vallado, NOT Curtis
    au = AU_KM

    # elements & rates conversions
    j2000_coe[0] *= au  # [km] sma (semi-major axis, aka a)
    j2000_coe[1] *= ureg.dimensionless  # ecc (eccentricity)
    # j2000_coe[2:5] *= DEG # not sure why this does not work for nrxt cmds
    j2000_coe[2] *= DEG  # [deg]
    j2000_coe[3] *= DEG  # [deg]
    j2000_coe[4] *= DEG  # [deg]
    j2000_coe[5] *= DEG  # [deg]

    j2000_rates[0] *= au / CENT
    j2000_rates[1] *= 1 / CENT  # ecc (eccentricity)
    j2000_rates[2] *= DEG / CENT
    j2000_rates[3] *= DEG / CENT
    j2000_rates[4] *= DEG / CENT
    j2000_rates[5] *= DEG / CENT
    # Curtis [3] data set rates have units of seconds/century
    #   so conversion is required for calling routines.
    if d_set == 1:  # convert Curtis angles rate change
        # convert (arcsec/cy) to (deg/cy)
        j2000_rates[2:6] /= 3600

    return j2000_coe, j2000_rates


def rot_matrix(angle, axis: int):
    """
    Returns rotation matrix based on user axis choice.
        Function from github, lamberthub, utilities->elements.py

    Input Parameters:
    ----------
        angle      : [rad]
        axis       : axis=0 rotate x-axis;
                    axis=1 rotate y-axis;
                    axis=2 rotate z-axis
    Returns:
    -------
        np.array   : rotation matrix, 3x3

    Raises:
    ------
        ValueError : if invalid axis
    Notes:
    -----------

    """
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    elif axis == 2:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Invalid axis: axis=0, x; axis=1, y; axis=2, z")


def coe_from_rv(r_vec, v_vec, mu: float):
    """
    Convert position/velocity vectors in IJK frame to Keplerian orbital elements.
    Vallado [2] pp.113, algorithm 9, rv2cov(), and Vallado [2] pp.114, example 2-5.
    Vallado [4] pp.115, algorithm 9, rv2cov(), and Vallado [2] pp.116, example 2-5.
    See Curtis example 4.3 in Example4_x.py.

    TODO: 2024-Sept, test special orbit types; (1) circular & equatorial; (2) orbit limits
    TODO: 2024-Sept, improve efficiency by elliminating redundant calculations

    Converts position and velocity vectors in the IJK frame to Keplerian
    orbital elements.  Reference Vallado, section 2.5, p.113, algorithm 9.

    Input Parameters:
    ----------
        r_vec  : numpy.array, [km] row vector, position
        v_vec  : numpy.array, [km] row vector, velocity
        mu     : float, [km^3/s^2], gravitational parameter

    Returns:
    -------
        sp     : float, [km or au] semi-parameter (aka p)
        sma    : float, [km or au] semi-major axis (aka a)
        ecc    : float, [--] eccentricity
        incl   : float, [rad] inclination
        raan   : float, [rad] right ascension of ascending node (aka capital W)
        w_     : float, [rad] arguement of periapsis (aka aop, or arg_p)
        TA     : float, [rad] true angle/anomaly (aka t_anom, or theta)
        o_type : string, [-] string of orbit type, circular, equatorial, etc.)

    Other coe Elements:
        u_     : float, [rad], argument of lattitude (aka )
                    for circular inclined orbits
        Lt0    : float, [rad], true longitude at epoch
                    for circular equatorial orbits
        t_peri : time of periapsis passage
        w_hat  : [deg] longitude of periapsis (NOT arguement of periapsis, w)
                    Note, w_hat = w + RAAN
        wt_hat : [deg] true longitude of periapsis
                    measured in one plane
        L_     : [deg] mean longitude (NOT mean anomaly, M)
                    Note, L = w_hat + M
        M_     : mean anomaly (often replaces TA)

    Note
    ----
    This algorithm handles special cases (circular, equatorial, etc.) by
        setting raan, aop, and anom as needed by Vallado [4], coe2rv()
    """
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    r0_inv = 1.0 / r_mag  # store for efficiency
    h_vec = np.matrix(np.cross(r_vec, v_vec, axis=0))  # row vectors in, row vec out
    h_mag = np.linalg.norm(h_vec)
    print(f"h_vec= {h_vec}")
    print(f"h_mag= {h_mag}")
    h_vec = np.ravel(h_vec)  # flatten h_vec;  make row vector
    print(f"h_vec= {h_vec}")

    # note, k_hat = np.array([0, 0, 1])
    # if n_vec = 0 then equatorial orbit
    n_vec = np.cross([0, 0, 1], h_vec)
    n_mag = np.linalg.norm(n_vec)
    print(f"n_vec= {n_vec}")

    # eccentricity; if ecc = 0 then circular orbit
    A = (v_mag * v_mag - mu * r0_inv) * r_vec
    B = -(np.dot(r_vec, v_vec)) * v_vec
    ecc_vec = (1 / mu) * (A + B)
    ecc_mag = np.linalg.norm(ecc_vec)
    if ecc_mag < 1e-6:
        ecc_mag = 0.0
        ecc_inv = 1 / ecc_mag

    xi = (0.5 * v_mag * v_mag) - mu * r0_inv  # related to orbit energy
    if ecc_mag != 1.0:
        sma = -0.5 * mu / xi
        sp = sma * (1.0 - ecc_mag * ecc_mag)
    else:  # parabolic orbit
        sma = np.inf
        sp = h_mag * h_mag / mu

    incl = np.arccos(h_vec[2] / h_mag)  # no quadrent check needed
    print(f"incl= {incl:.6g} [rad], {incl*180/np.pi} [deg]")

    # test special cases & orbit type (o-type)
    #   elliptical equatorial, circular, circular equatorial
    if n_mag == 0.0:  # Equatorial
        if ecc_mag < 1e-6:  # circular equatorial
            Lt_ = np.arccos(r_vec[0] * r0_inv)
            if r_vec[1] < 0:
                Lt_ = 2.0 * np.pi - Lt_
            raan = 0.0
            w_ = 0.0  # aka aop
            TA = Lt_
            o_type = "equatorial circular"
        else:  # ecc > 0, thus ellipse, or parabola, or hyperbola
            wt_hat = np.arccos(ecc_vec[0] * ecc_inv)
            if ecc_vec[1] < 0:
                wt_hat = 2.0 * math.pi - wt_hat
            raan = 0.0
            w_ = wt_hat
            TA = np.arccos(np.dot(ecc_vec, r_vec) * ecc_inv * r0_inv)
            o_type = "equatorial elliptical, parabolic, hyperbolic"
    elif ecc_mag < 1e-6:  # circular
        n_inv = 1.0 / n_mag
        raan = np.arccos(n_vec[0] * n_inv)
        w_ = 0.0
        u_ = np.arccos(np.dot(n_vec, r_vec) * n_inv * r0_inv)
        if r_vec[2] < 0:
            u = 2.0 * math.pi - u_
        TA = u_  # remember, u_ = argument of lattitude
        o_type = "circular"
    else:
        n_inv = 1.0 / n_mag
        ecc_inv = 1 / ecc_mag

        raan = np.arccos(n_vec[0] * n_inv)
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan

        # w_ = arguement of periapsis (aka aop, or arg_p)
        w_ = math.acos(np.dot(n_vec, ecc_vec) * n_inv * ecc_inv)
        if ecc_vec[2] < 0:
            w_ = 2 * math.pi - w_

        TA = math.acos(np.dot(ecc_vec, r_vec) / (ecc_mag * r_mag))
        if np.dot(r_vec, v_vec) < 0:
            TA = 2 * math.pi - TA

        o_type = "not special orbit-type"
    return sp, sma, ecc_mag, incl, raan, w_, TA, o_type


def val_rv2coe(r_vec, v_vec, mu):
    """
    Convert position/velocity vectors to Keplerian orbital elements (coe).
    Vallado [4] section 2.5, pp.114, algorithm 9 pp.115, rv2cov(), example 2-5 pp.116.
    See Curtis [3] algorithm 4.2 pp.209, example 4.3, pp.212, in Example4_x.py.

    TODO: 2024-Sept, improve efficiency by elliminating redundant calculations
    TODO: 2024-Nov, finish test cases

    Input Parameters:
    ----------
        r_vec  : numpy.array, [km] row vector, position
        v_vec  : numpy.array, [km/s] row vector, velocity
        mu     : float, [km^3/s^2], gravitational parameter

    Returns:
    ----------
        o_type : int  , [-] orbit type:
                        1=circular, 2=circular equatorial
                        3=elliptical, 4=elliptical equatorial
                        5=parabolic, 6=parabolic equatorial
                        7=hyperbolic, 8=hyperbolic equatorial
        sp     : float, [km or au] semi-parameter (aka p)
        sma    : float, [km or au] semi-major axis (aka a)
        ecc    : float, [--] eccentricity
        incl   : float, [rad] inclination
        raan   : float, [rad] right ascension of ascending node (aka capital W)
        w_     : float, [rad] arguement of periapsis (aka aop, or arg_p)
        TA     : float, [rad] true angle/anomaly (aka t_anom, or theta)

        alternative coe's for circular & equatorial:
        Lt0    : float, [rad] true longitude at epoch, circular equatorial
                        when incl=0, ecc=0
        w_bar  : float, [rad] longitude of periapsis (aka II), equatorial
                    NOT argument of periapsis, w_
                    Note, w_bar = w + RAAN
        u_     : float, [rad] argument of lattitude (aka ), circular inclined

    Other potential elements:
        L_     : float, [deg] mean longitude
                    NOT mean anomaly, M
                    L_ = w_bar + M
        wt_bar : float, [rad] true longitude of periapsis
                    measured in one plane
        M_     : mean anomaly, often replaces TA
        t_peri : float, [jd] time of periapsis passage

    circular, e=0: w_ and TA = undefined;
        use argument of latitude, u_; u_=acos((n_vec X r_vec)/(n_mag * r_mag))
        If r_vec[2] < 0 then 180 < u < 360 degree

    equatorial, i=0 or 180 [deg]: raan and w_ = undefined
        use longitude of periapsis, II (aka w_bar); II=acos(e_vec[0]/e_mag)
        If e_vec[1] < 0 then 180 < II < 360 degree

    circular & equatorial, e=0 and i=0 or i=180: w_ and raan and TA = undefined;
        use true longitude, Lt0 = angle between r0 & I-axis; Lt0=acos(r_vec[1]/r_mag)
        If r_mag[1] < 0 then 180 < Lt0 < 360 degree

    Notes:
    ----------
        This algorithm handles special cases (circular, equatorial).
    """
    SMALL_c = 0.00015  # circle threshold value, defines ecc=0
    SMALL_p = 0.00001  # parabolic threshold value, defines ecc=1
    # initialize values since some will not be calculated...
    sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_ = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    # if n_vec = 0 then equatorial orbit
    n_vec = np.cross([0, 0, 1], h_vec)
    n_mag = np.linalg.norm(n_vec)

    # eccentricity; if ecc = 0 then circular orbit
    r0_inv = 1.0 / r_mag  # for calculation efficiency
    ecc_inv = 1  # for calculation efficiency
    A = (v_mag * v_mag - mu * r0_inv) * r_vec
    B = -(np.dot(r_vec, v_vec)) * v_vec
    ecc_vec = (1 / mu) * (A + B)
    ecc_mag = np.linalg.norm(ecc_vec)
    if ecc_mag < SMALL_c:
        ecc_mag = 0.0
        ecc_vec = ecc_vec * 0
        ecc_inv = np.inf
    elif ecc_mag > (1 - SMALL_p) and ecc_mag < (1 + SMALL_p):
        ecc_mag = 1
    else:
        ecc_inv = 1 / ecc_mag

    # xi=orbit energy
    xi = (0.5 * v_mag * v_mag) - mu * r0_inv

    if ecc_mag == 0:  # circular, avoids rounding issues
        sma = r_mag
        sp = r_mag
    elif ecc_mag != 0 and ecc_mag != 1:  # not parabolic
        sma = -0.5 * mu / xi
        sp = sma * (1.0 - ecc_mag * ecc_mag)
    elif ecc_mag == 1:  # parabolic orbit
        sma = np.inf
        sp = h_mag * h_mag / mu

    incl = np.arccos(h_vec[2] / h_mag)  # no quadrent check needed

    # examine orbit type (o-type)
    if n_mag == 0.0:  # Equatorial
        if ecc_mag < SMALL_c:  # circular equatorial
            # Lt0=true longitude at epoch (r0)
            Lt0 = math.acos(r_vec[0] * r0_inv)
            if r_vec[1] < 0:
                Lt0 = 2.0 * np.pi - Lt0
            raan = np.nan
            w_ = np.nan  # aka aop
            o_type = 1  # circular equatorial
        else:  # ecc > 0, thus equatorial -> ellipse, parabola, hyperbola
            # w_bar=longitude of periapsis (aka II)
            w_bar = math.acos(ecc_vec[0] * ecc_inv)
            if ecc_vec[1] < 0:
                w_bar = 2.0 * math.pi - w_bar
            raan = np.nan
            TA = math.acos(np.dot(ecc_vec, r_vec) / (ecc_mag * r_mag))
            if np.dot(r_vec, v_vec) < 0:
                TA = 2 * math.pi - TA
            # equatorial, for either ellipse, parabola, hyperbola
            if ecc_mag < 1:
                o_type = 4  # equatorial ellipse
            elif ecc_mag > 1:
                o_type = 8  # equatorial hyperbola
            else:
                o_type = 6  # equatorial parabola
    elif ecc_mag < SMALL_c:  # not equatorial, but circular (inclined)
        w_ = np.nan
        TA = np.nan

        n_inv = 1.0 / n_mag
        raan = math.acos(n_vec[0] * n_inv)
        # u_ = argument of lattitude
        u_ = math.acos(np.dot(n_vec, r_vec) * n_inv * r0_inv)
        if r_vec[2] < 0:
            u_ = 2.0 * math.pi - u_
        o_type = 1  # circular (inclined)
    else:  # elements: non-equatorial, non-circular
        n_inv = 1.0 / n_mag
        ecc_inv = 1 / ecc_mag

        raan = math.acos(n_vec[0] * n_inv)
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan

        # w_ = arguement of periapsis (aka aop, or arg_p)
        w_ = math.acos(np.dot(n_vec, ecc_vec) * n_inv * ecc_inv)
        if ecc_vec[2] < 0:
            w_ = 2 * math.pi - w_

        TA = math.acos(np.dot(ecc_vec, r_vec) / (ecc_mag * r_mag))
        if np.dot(r_vec, v_vec) < 0:
            TA = 2 * math.pi - TA

        # finish identifying orbit type
        if ecc_mag < 1:
            o_type = 3  # ellipse
        elif ecc_mag > 1:
            o_type = 7  # hyperbola
        else:
            o_type = 5  # parabola

    elements = np.array([sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_])
    return o_type, elements


def o_type_decode(o_type):
    """
    Print orbit type based on input value.
        Supports val_rv2coe() definitions.
    o_type, orbit type python dictionary list:
        0:"circular", 1:"circular inclined", 2:"circular equatorial",
        3:"elliptical", 4:"elliptical equatorial",
        5:"parabolic", 6:"parabolic equatorial",
        7:"hyperbolic", 8:"hyperbolic equatorial"
    """
    # python dictionary
    o_type_list = {
        0: "circular",
        1: "circular inclined",
        2: "circular equatorial",
        3: "elliptical",
        4: "elliptical equatorial",
        5: "parabolic",
        6: "parabolic equatorial",
        7: "hyperbolic",
        8: "hyperbolic equatorial",
    }
    print(f"{o_type_list.get(o_type)}")
    return None


def print_coe(o_type, elements):
    """supports val_rv2coe() definitions"""
    rad2deg = 180 / math.pi
    sp, sma, ecc_mag, incl, raan, w_, TA, Lt0, w_bar, u_ = elements

    o_type_decode(o_type=o_type)  # prints orbit type
    print(f"semi-parameter, sp= {sp} [km]")
    print(f"semi-major axis, sma= {sma} [km]")
    print(f"eccentricity, ecc_mag= {ecc_mag}")
    print(f"incl= {incl} [rad], {incl*rad2deg} [deg]")
    print(f"raan= {raan} [rad], {raan*rad2deg} [deg]")
    print(f"arguement of periapsis, w_= {raan} [rad], {w_*rad2deg} [deg]")
    print(f"true anomaly/angle, TA= {TA} [rad], {TA*rad2deg} [deg]")
    print(f"Lt0= {Lt0} [rad], {Lt0*rad2deg} [deg]")
    print(f"w_bar= {w_bar} [rad], {w_bar*rad2deg} [deg]")
    print(f"u_= {u_} [rad], {u_*rad2deg} [deg]")
    return None


def coe_from_date(planet_id: int, date_UT):
    """
    2025, make this units-aware using constants_1.py
    Compute planetary coe (clasic orbital elements), given earth date [ut].
    Keep in mind, outputs are in [km] & [rad]

    Input Parameters:
    ----------
        planet_id   : int, Mercury->Pluto
        date_UT, t0 : python date/time list;
                        yr, mo, d, hr, minute, sec

    Returns:
    ----------
        t0_coe     : python list, coe at t0.
            See details and variable list in planetary_elements():
            sma [km], ecc [-], incl [rad], RAAN [rad], w_hat [rad], L [rad]
        jd_t0      : float, julian date of planet coe

    Notes:
    ----------
        Julian day (jd) algorithm updated from ex8.7 algorithm to
            cover full range of jd.
    """
    # Steps 1, 2, 3, of Curtis [3] p.473.  Part of algorithm 8.1.
    yr, mo, d, hr, minute, sec = date_UT
    jd_t0 = g_date2jd(yr=yr, mo=mo, d=d, hr=hr, minute=minute, sec=sec, leap_sec=False)
    # Julian centuries at j2000, fixed, no need to calculate
    # Curtis [3] p.471, eqn 8.93a
    t0_j_cent = (jd_t0 - 2451545.0) / 36525  # julian centuries since t0

    # orbital elements tables kept in functions.py
    # Curtis, p.473, step 3; Note,
    #   data set=1 means Curtis [3] table 8.1; data set=0 means JPL Horizons Table 1
    j2000_coe, j2000_rates = planetary_elements(planet_id=planet_id, d_set=1)

    # apply century rates to j2000 coe rates
    #   python list multiply
    t0_rates = [j2000_rates[x] * t0_j_cent for x in range(len(j2000_rates))]
    # python list add
    t0_coe = [j2000_coe[x] + t0_rates[x] for x in range(len(j2000_coe))]

    # [deg] values need to be 0-360; better method must exist for below
    # angular output values converted to [rad]
    t0_coe[2] = (t0_coe[2] % 360) * math.pi / 180  # note modulo arithmetic, %
    t0_coe[3] = (t0_coe[3] % 360) * math.pi / 180  # note modulo arithmetic, %
    t0_coe[4] = (t0_coe[4] % 360) * math.pi / 180  # note modulo arithmetic, %
    t0_coe[5] = (t0_coe[5] % 360) * math.pi / 180  # note modulo arithmetic, %

    # coe orbital elements list; see variable list in planetary_elements():
    # coe_elements_names= ["sma", "ecc", "incl", "RAAN", "w_hat", "L"]
    return t0_coe, jd_t0


def sv_from_coe(h, ecc, RA_rad, incl_rad, w_rad, TA_rad, mu):
    """
    Compute state vector (r,v) IJK, from classic orbital elements (coe).
    Curtis [3] p.232, example 4.7, algorithm 4.5.
    For sv->coe, Curtis [3] pp.209, algorithm 4.2, & Curtis pp.212, example 4.3.
    Check out alternative coe2rv() from Vallado [4].

    Also see interplanetary flight http://www.braeunig.us/space/interpl.htm

    2024-August, many edits from MatLab translation!
    TODO cleanup trig naming; I was in a rush; there are some un-necessary variables.
    NOTE consider using quaternions to avoid the gimbal lock of euler angles.

    Input Parameters:
        mu   - [km^3 / s^2] gravitational parameter
        coe  - orbital elements (h, ecc, RA, incl, w, TA)
            h    = [km^2/s] magnitude, angular momentum
                    p=h^2 / mu ; thus h=sqrt(p * mu)
            ecc  = [-] eccentricity
            RA   = [rad] right ascension of the ascending node;
                    (aka RAAN, or capital W, or Omega)
            incl = [rad] inclination of the orbit
            w    = [rad] argument of perigee
                    (aka omega, aop)
            TA   = [rad] true angle/anomaly
        R3_w - Rotation matrix about the z-axis through the angle w
        R1_i - Rotation matrix about the x-axis through the angle i
        R3_RA- Rotation matrix about the z-axis through the angle RA
        Q_pX - Matrix of the transformation from perifocal to geocentric
                equatorial frame
        rp   - position vector in the perifocal frame [km]
        vp   - velocity vector in the perifocal frame [km/s]
        r    - position vector in the geocentric equatorial frame [km]
        v    - velocity vector in the geocentric equatorial frame [km/s]
    """
    # saved trig computations save computing time
    cosv = math.cos(TA_rad)
    sinv = math.sin(TA_rad)
    cosi = math.cos(incl_rad)
    sini = math.sin(incl_rad)
    cosw = math.cos(w_rad)
    sinw = math.sin(w_rad)
    coso = math.cos(RA_rad)
    sino = math.sin(RA_rad)

    # Curtis eqns 4.45 and 4.46 (rp and vp are column vectors):
    rp = (
        (h**2 / mu)
        * (1 / (1 + ecc * cosv))
        * (cosv * np.array([1, 0, 0]) + sinv * np.array([0, 1, 0]))
    )
    rp = rp.reshape(-1, 1)  # convert to column vector
    vp = (mu / h) * (-sinv * np.array([1, 0, 0]) + (ecc + cosv) * np.array([0, 1, 0]))
    vp = vp.reshape(-1, 1)  # convert to column vector

    # Create rotation matrices/arrays
    # rotate z-axis thru angle RA, Curtis, eqn 4.34
    # R3_RA = [ math.cos(RA)  math.sin(RA)  0
    #         -math.sin(RA)  math.cos(RA)  0
    #             0        0     1]
    c_RA, s_RA = coso, sino  #
    R3_RA = np.array([[c_RA, s_RA, 0], [-s_RA, c_RA, 0], [0, 0, 1]])

    # rotation about x-axis, inclination, Curtis, eqn 4.32
    # R1_i = [1       0          0
    #         0   cos(incl)  sin(incl)
    #         0  -sin(incl)  cos(incl)]
    c_in, s_in = np.cos(incl_rad), np.sin(incl_rad)
    R1_i = np.array([[1, 0, 0], [0, c_in, s_in], [0, -s_in, c_in]])
    # print(f"incl= {incl_rad*180/np.pi} [rad]")
    # print(f"R1_i= {R1_i}")

    # rotation about z-axis, Curtis, eqn 4.34
    # R3_w = [ cos(w)  sin(w)  0
    #         -sin(w)  cos(w)  0
    #         0       0     1]
    c_w, s_w = np.cos(w_rad), np.sin(w_rad)
    R3_w = np.array([[c_w, s_w, 0], [-s_w, c_w, 0], [0, 0, 1]])
    # print(f"R3_w= {R3_w}")

    # Curtis, eqn 4.49
    Q_pX = R3_w @ R1_i @ R3_RA  # matrix multiply
    # print(f"Q_px= {Q_pX}")
    Q_Xp = np.transpose(Q_pX)

    # Curtis, eqn 4.51 (r and v are column vectors):
    r = Q_Xp @ rp
    v = Q_Xp @ vp

    # Convert r and v column vectors to row vectors:
    r = np.ravel(r)  # flatten the array; row vectors
    v = np.ravel(v)  # flatten the array; row vectors
    return r, v


def coe2rv(p, ecc, inc_rad, raan_rad, aop_rad, anom_rad, mu):
    """
    Convert Keplerian orbital elements to position/velocity vectors; IJK frame.
    Vallado [2], section 2.6, algorithm 10, pp.118 or Vallado [4] pp.120
    Maybe more efficient then sv_from_coe(); defined in Curtis [3].

    Input Parameters:
    ----------
        p    : float, [km] semi-parameter
                p=h^2 / mu ; thus h=sqrt(p * mu)
        ecc  : float, [--] eccentricity
        inc  : float, [rad] inclination
        raan : float, [rad] right ascension of the ascending node
        aop  : float, [rad] argument of periapsis (aka w, or omega)
                w=w_bar-RAAN; undefined for RAAN=0, undefined for circular
        anom : float, [rad] true angle/anomaly (aka TA)
        mu   : float, [km^3/s^2] Gravitational parameter
    Returns:
    -------
        r_ijk : numpy.array, [km] position vector in IJK frame
        v_ijk : numpy.array, [km/s] velocity vector in IJK frame
    Notes:
    ----
        Algorithm assumes raan, aop, and anom have been set to account for
        special cases (circular, equatorial, etc.) as in rv2coe (Algorithm 9)
        Also see Curtis, p.473 example 8.7.
    """
    # saved trig computations save computing time
    cosv = np.cos(anom_rad)
    sinv = np.sin(anom_rad)
    cosi = np.cos(inc_rad)
    sini = np.sin(inc_rad)
    cosw = np.cos(aop_rad)
    sinw = np.sin(aop_rad)
    coso = np.cos(raan_rad)
    sino = np.sin(raan_rad)

    r_pqw = np.matrix(
        [p * cosv / (1.0 + ecc * cosv), p * sinv / (1.0 + ecc * cosv), 0.0]
    )
    r_pqw = r_pqw.T  # Make column vector

    v_pqw = np.matrix([-np.sqrt(mu / p) * sinv, np.sqrt(mu / p) * (ecc + cosv), 0.0])
    v_pqw = v_pqw.T  # Make column vector

    m_pqw2ijk = [
        [
            coso * cosw - sino * sinw * cosi,
            -coso * sinw - sino * cosw * cosi,
            sino * sini,
        ],
        [
            sino * cosw + coso * sinw * cosi,
            -sino * sinw + coso * cosw * cosi,
            -coso * sini,
        ],
        [sinw * sini, cosw * sini, cosi],
    ]
    m_pqw2ijk = np.matrix(m_pqw2ijk)
    #    m_pqw2ijk = np.matrix([[row1], [row2], [row3]])
    # Convert to IJK frame; then make row vectors
    r_ijk = m_pqw2ijk * r_pqw
    v_ijk = m_pqw2ijk * v_pqw
    r_ijk = np.ravel(r_ijk)  # flatten the array; row vectors
    v_ijk = np.ravel(v_ijk)  # flatten the array; row vectors
    return r_ijk, v_ijk


def date_to_jd(year, month, day):
    # Convert a date to Julian Day.
    # Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
    # 4th ed., Duffet-Smith and Zwart, 2011.
    # This function extracted from https://gist.github.com/jiffyclub/1294443
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if (
        (year < 1582)
        or (year == 1582 and month < 10)
        or (year == 1582 and month == 10 and day < 15)
    ):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.0)
        B = 2 - A + math.trunc(A / 4.0)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
    D = math.trunc(30.6001 * (monthp + 1))
    jd = B + C + D + day + 1720994.5
    return jd  # date_to_jd()


def sun_rise_set1():
    """
    Code by Procrastilearner under the CC BY-SA 4.0 license.

    Source for the sunrise calculation:
        https://en.wikipedia.org/wiki/Sunrise_equation
        https://steemit.com/steemstem/@procrastilearner/killing-time-with-recreational-math-calculate-sunrise-and-sunset-times-using-python

    Some sample locations
    Toronto Ontario Canada
    """

    latitude_deg = 43.65
    longitude_deg = -79.38
    timezone = (
        -4.0
    )  # Daylight Savings Time is in effect, this would be -5 for winter time

    # Whitehorse Yukon Territories Canada
    # latitude_deg =60.7
    # longitude_deg = -135.1
    # timezone = -7.0 #Daylight Savings Time is in effect, this would be -8 for winter time

    # Paris France
    # latitude_deg =48.85
    # longitude_deg = 2.35
    # timezone = 2.0

    # Hong Kong PRC
    # latitude_deg =22.32
    # longitude_deg =114.1
    # timezone = 8.0

    # Perth Australia
    # latitude_deg =-31.9
    # longitude_deg =115.9
    # timezone = 8.0
    # *****************************************

    pi = 3.14159265359
    latitude_deg = 43.65
    longitude_deg = -79.38
    # Daylight Savings Time is in effect, this would be -5 for winter time
    timezone = -4.0

    latitude_radians = math.radians(latitude_deg)
    longitude__radians = math.radians(longitude_deg)

    jd2000 = 2451545  # the julian date for Jan 1 2000 at noon

    currentDT = datetime.datetime.now()
    current_year = currentDT.year
    current_month = currentDT.month
    current_day = currentDT.day
    current_hour = currentDT.hour

    jd_now = g_date2jd(current_year, current_month, current_day)

    n = jd_now - jd2000 + 0.0008

    jstar = n - longitude_deg / 360

    M_deg = (357.5291 + 0.98560028 * jstar) % 360
    M = M_deg * pi / 180

    C = 1.9148 * math.sin(M) + 0.0200 * math.sin(2 * M) + 0.0003 * math.sin(3 * M)

    lamda_deg = math.fmod(M_deg + C + 180 + 102.9372, 360)

    lamda = lamda_deg * pi / 180

    Jtransit = 2451545.5 + jstar + 0.0053 * math.sin(M) - 0.0069 * math.sin(2 * lamda)

    earth_tilt_deg = 23.44
    earth_tilt_rad = math.radians(earth_tilt_deg)

    sin_delta = math.sin(lamda) * math.sin(earth_tilt_rad)
    angle_delta = math.asin(sin_delta)

    sun_disc_deg = -0.83
    sun_disc_rad = math.radians(sun_disc_deg)

    cos_omega = (
        math.sin(sun_disc_rad) - math.sin(latitude_radians) * math.sin(angle_delta)
    ) / (math.cos(latitude_radians) * math.cos(angle_delta))

    omega_radians = math.acos(cos_omega)
    omega_degrees = math.degrees(omega_radians)

    # Output section
    print("------------------------------")
    print("Today's date is " + currentDT.strftime("%Y-%m-%d"))
    print("------------------------------")
    # ("%Y-%m-%d %H:%M")

    print("Latitude =  " + str(latitude_deg))
    print("Longitude = " + str(longitude_deg))
    print("Timezone =  " + str(timezone))
    print("------------------------------")

    Jrise = Jtransit - omega_degrees / 360
    numdays = Jrise - jd2000
    numdays = numdays + 0.5  # offset because Julian dates start at noon
    numdays = numdays + timezone / 24  # offset for time zone
    sunrise = datetime.datetime(2000, 1, 1) + datetime.timedelta(numdays)
    print("Sunrise is at " + sunrise.strftime("%H:%M"))

    Jset = Jtransit + omega_degrees / 360
    numdays = Jset - jd2000
    numdays = numdays + 0.5  # offset because Julian dates start at noon
    numdays = numdays + timezone / 24  # offset for time zone
    sunset = datetime.datetime(2000, 1, 1) + datetime.timedelta(numdays)
    print("Sunset is at  " + sunset.strftime("%H:%M"))
    print("------------------------------")

    return  # sunRiseSet1()


def hohmann_transfer(r1, r2, mu):
    """
    Calculate Hohmann transfer parameters.
    From google search: python generate hohmann transfer table

    Args:
        r1: Radius of initial orbit (AU).
        r2: Radius of final orbit (AU).
        mu: Gravitational parameter of the Sun (AU^3/day^2).

    Returns:
        A tuple containing:
        - Transfer time (days).
        - Delta-v at departure (AU/day).
        - Delta-v at arrival (AU/day).
    """
    a_transfer = (r1 + r2) / 2  # Semi-major axis of transfer orbit
    transfer_time = math.pi * math.sqrt(a_transfer**3 / mu)
    v1 = math.sqrt(mu / r1)  # Velocity in initial orbit
    v2 = math.sqrt(mu / r2)  # Velocity in final orbit
    v_transfer_1 = math.sqrt(mu * (2 / r1 - 1 / a_transfer))
    v_transfer_2 = math.sqrt(mu * (2 / r2 - 1 / a_transfer))
    delta_v1 = abs(v_transfer_1 - v1)
    delta_v2 = abs(v_transfer_2 - v2)
    return transfer_time, delta_v1, delta_v2


def hohmann_transfer_a(r1, r2):
    """
    Calculates and returns the points for a Hohmann transfer orbit.

    Args:
        r1: Radius of the initial orbit (AU).
        r2: Radius of the target orbit (AU).

    Returns:
        A tuple containing:
            - transfer_r: Array of radial distances for the transfer orbit.
            - transfer_theta: Array of angles for the transfer orbit.
    """

    a_transfer = (r1 + r2) / 2  # Semi-major axis of the transfer orbit
    transfer_theta = np.linspace(0, np.pi, 100)  # Angles for half an ellipse
    transfer_r = a_transfer * (1 - ((r2 - r1) / (r1 + r2)) * np.cos(transfer_theta))
    return transfer_r, transfer_theta


def create_hohmann_table(planets, mu):
    """
    Create a Hohmann transfer table.

    Args:
        planets: A dictionary of planet names and their orbital radii (AU).
        mu: Gravitational parameter of the Sun (AU^3/day^2).

    Returns:
        A string representing the Hohmann transfer table.
    """
    table = "Planet | Planet | Transfer Time (days) | Delta-v Departure (AU/day) | Delta-v Arrival (AU/day)\n"
    table += "------|--------|---------------------|--------------------------|-------------------------\n"

    for p1, r1 in planets.items():
        for p2, r2 in planets.items():
            if p1 != p2:
                transfer_time, delta_v1, delta_v2 = hohmann_transfer(r1, r2, mu)
                table += f"{p1} | {p2} | {transfer_time:.2f} | {delta_v1:.4f} | {delta_v2:.4f}\n"
    return table


def main():
    """just a placeholder to help with editor navigation:--)"""
    return


# the following for temporary test/examine functions
#   see test_xxxx_xxxx.py for detailed tests
if __name__ == "__main__":
    main()  # do nothing for now
