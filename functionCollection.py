"""
Collection of functions for Curtis [3] examples and problems
TODO ***** need to put some vectors into python numpy syntax *****
TODO ***** eliminate global variables *****

    The following is an on-line matlab -> python converter
    https://www.codeconvert.ai/matlab-to-python-converter
Notes:
----------
    The following is an on-line matlab -> python converter
    https://www.codeconvert.ai/matlab-to-python-converter
    
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

import numpy as np
import scipy.optimize  # used to solve kepler E

from astro_time import julian_date


def lambert(mu: float, R1, R2, t: float, string: str):
    """
    Lambert Solver

    Parameters:
    ----------
        mu : float
            _description_
        R1 : np.array()
            _description_
        R2 : np.array
            _description_
        t : float
            _description_
        string : str
            _description_

    Returns:
    -------
        _type_
            _description_
    """
    # Magnitudes of R1 and R2:
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)

    c12 = np.cross(R1, R2)
    theta = np.arccos(np.dot(R1, R2) / (r1 * r2))

    # Determine whether the orbit is prograde or retrograde:
    if string != "retro" and string != "pro":
        string = "pro"
        print("\n ** Prograde trajectory assumed.\n")

    if string == "pro":
        if c12[2] <= 0:
            theta = 2 * np.pi - theta
    elif string == "retro":
        if c12[2] >= 0:
            theta = 2 * np.pi - theta

    # Equation 5.35:
    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))

    # Determine approximately where F(z,t) changes sign, and
    # use that value of z as the starting value for Equation 5.45:
    z = -100
    while F(z, t) < 0:
        z = z + 0.1

    # Set an error tolerance and a limit on the number of iterations:
    tol = 1e-8
    nmax = 5000

    # Iterate on Equation 5.45 until z is determined to within the
    # error tolerance:
    ratio = 1
    n = 0
    while abs(ratio) > tol and n <= nmax:
        n = n + 1
        ratio = F(z, t) / dFdz(z)
        z = z - ratio

    # Report if the maximum number of iterations is exceeded:
    if n >= nmax:
        print("\n\n **Number of iterations exceeds #g in " "lambert" " \n\n ")  # nmax

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


# Subfunctions used in the main body:
# Equation 5.38:
def y(z):
    return r1 + r2 + A * (z * S(z) - 1) / np.sqrt(C(z))


# Equation 5.40:
def F(z, t):
    return (y(z) / C(z)) ** 1.5 * S(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * t


# Equation 5.43:
def dFdz(z):
    if z == 0:
        return np.sqrt(2) / 40 * y(0) ** 1.5 + A / 8 * (
            np.sqrt(y(0)) + A * np.sqrt(1 / (2 * y(0)))
        )
    else:
        return (y(z) / C(z)) ** 1.5 * (
            1 / (2 * z) * (C(z) - 3 * S(z) / (2 * C(z))) + 3 * S(z) ** 2 / (4 * C(z))
        ) + A / 8 * (3 * S(z) / C(z) * np.sqrt(y(z)) + A * np.sqrt(C(z) / y(z)))


"""
    Stumpff functions originated by Karl Stumpff, circa 1947
    Stumpff functions (C(z), S(z)) are part of a universal variable solution,
    which work regardless of eccentricity.
"""


def C(z):  # temporary, until I change calling routines
    return stumpff_C(z)


def S(z):  # temporary, until I change calling routines
    return stumpff_S(z)


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


def sphere_of_influence(sma, mass1, mass2):
    """radius of the SOI (sphere of influence)

    Parameters
    ----------
    sma : float
        semi-major axis of mass1 orbiting mass2
    mass1 : float
        generally the smaller of the 2 mass's (i.e. planet)
    mass2 : float
        generally the larger of the 2 mass's (i.e. sun)

    Returns
    -------
    r_SOI : float
        radius of the SOI (sphere of influence)
    """
    r_SOI = sma * (mass1 / mass2) ** (2 / 5)
    return r_SOI


###############################################
def E_zerosolver(E, args):
    Me = args[0]
    ecc = args[1]
    return E - ecc * np.sin(E) - Me


def solve_for_E(Me: float, ecc: float):
    """
    Solve Keplers equation

    Parameters:
    ----------
    Me : float, mean angle/anomaly
    ecc : float, eccentricity

    Return:
    -------
    sols : float, E [rad]
    """
    # iterative solution process
    sols = scipy.optimize.fsolve(E_zerosolver, x0=Me, args=[Me, ecc])[0]
    return sols


from astro_time import julian_date


def planet_elements_and_sv(planet_id, year, month, day, hour, minute, second, mu):
    """
    Curtis pp.470, section 8.10; p.471-472, algorithm 8.1.; pp.473, example 8.7

    Parameters:
    ----------
    planet_id : _type_
        _description_
    year : _type_
        _description_
    month : _type_
        _description_
    day : _type_
        _description_
    hour : _type_
        _description_
    minute : _type_
        _description_
    second : _type_
        _description_
    mu : _type_
        _description_

    Return:
    -------
    _type_
        _description_
    """
    deg = math.pi / 180  # conversion [rad]->[deg]

    # Vallado equivilent of Curtis p.276, eqn 5.48:
    # parameters of julian_date(yr, mo, d, hr, minute, sec, leap_sec=False)
    jd = julian_date(yr=year, mo=month, d=day, hr=hour, minute=minute, sec=second)

    # commented code, below, obsolete; delete later
    # j0 = j1(year, month, day)
    # ut = (hour + minute / 60 + second / 3600) / 24
    # Curtis p.276 eqn 5.47
    # jd = j0 + ut

    # Planetary ephemeris pp.470, section 8.10; data, p.472, tbl 8.1.
    J2000_coe, rates = planetary_elements(planet_id)

    # Curtis p.471, eqn 8.93a
    t0 = (jd - 2451545) / 36525
    # Curtis p.471, eqn 8.93b
    elements = J2000_coe + rates * t0

    a = elements[0]
    e = elements[1]

    # Curtis p.89, eqn 2.71
    h = math.sqrt(mu * a * (1 - e**2))

    # Reduce the angular elements to range 0 - 360 [deg]
    incl = elements[2]
    RA = elements[3]  # [deg]
    w_hat = elements[4]  # [deg]
    L = elements[5]  # [deg]
    w = w_hat - RA  # [deg]
    M = L - w_hat  # [deg]

    # Curtis, p.163, algorithm 3.1 (M [rad]) in example 3.1
    # E = kepler_E(e, M * deg)  # [rad]
    E = solve_for_E(ecc=e, Me=M * deg)  # [rad]

    # Curtis, p.160, eqn 3.13 (convert to [deg] ???????????????):
    TA = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))  # [deg]?

    coe = [h, e, RA * deg, incl * deg, w * deg, TA * deg]

    # Curtis, p.231, algorithm 4.5; see p. 232, example 4.7
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
    import numpy as np

    # Verify position vectors are collinear. If so, verify the transfer
    # angle is 0 or pi.
    if np.all(np.cross(r1, r2) == 0):
        return 0 if np.all(np.sign(r1) == np.sign(r2)) else np.pi

    # Solve for a unitary vector normal to the vector plane. Its direction and
    # sense the one given by the cross product (right-hand) from r1 to r2.
    h = np.cross(r1, r2) / norm(np.cross(r1, r2))

    # Compute the projection of the normal vector onto the reference plane.
    alpha = dot(np.array([0, 0, 1]), h)

    # Get the minimum angle (0 <= dtheta <= pi) between r1 and r2.
    r1_norm, r2_norm = [norm(vec) for vec in [r1, r2]]
    theta0 = np.arccos(dot(r1, r2) / (r1_norm * r2_norm))

    # Fix theta as needed
    if prograde is True:
        dtheta = theta0 if alpha > 0 else 2 * np.pi - theta0
    else:
        dtheta = theta0 if alpha < 0 else 2 * np.pi - theta0

    return dtheta


def planetary_elements(planet_id):
    J2000_elements = [
        [0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350],
        [0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 181.97909950],
        [1.00000261, 0.01671123, -0.00001531, 0.0, 102.93768193, 100.46457166],
        [1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959, -4.55343205],
        [5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 34.39644501],
        [9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831, 49.95424423],
        [19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630, 313.23810451],
        [30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227, -55.12002969],
        [
            39.48211675,
            0.24882730,
            17.14001206,
            110.30393684,
            224.06891629,
            238.92903833,
        ],
    ]

    cent_rates = [
        [0.00000037, 0.00001906, -0.00594749, -0.12534081, 0.16047689, 149472.67411175],
        [0.00000390, -0.00004107, -0.00078890, -0.27769418, 0.00268329, 58517.81538729],
        [0.00000562, -0.00004392, -0.01294668, 0.0, 0.32327364, 35999.37244981],
        [0.0001847, 0.00007882, -0.00813131, -0.29257343, 0.44441088, 19140.30268499],
        [-0.00011607, -0.00013253, -0.00183714, 0.20469106, 0.21252668, 3034.74612775],
        [-0.00125060, -0.00050991, 0.00193609, -0.28867794, -0.41897216, 1222.49362201],
        [-0.00196176, -0.00004397, -0.00242939, 0.04240589, 0.40805281, 428.48202785],
        [0.00026291, 0.00005105, 0.00035372, -0.00508664, -0.32241464, 218.45945325],
        [-0.00031596, 0.00005170, 0.00004818, -0.01183482, -0.04062942, 145.20780515],
    ]

    J2000_coe = J2000_elements[planet_id - 1]
    rates = cent_rates[planet_id - 1]

    au = 149597871  # [km]
    J2000_coe[0] = J2000_coe[0] * au
    rates[0] = rates[0] * au

    return J2000_coe, rates


################################


def sv_from_coe(h, ecc, RA, incl, w, TA, mu):
    """
    Computes the state vector (r,v) from classical orbital elements (coe).
    2024-August, many edits from MatLab translation!
    Consider using quaternions to avoid the gimbal lock of euler angles.

    mu   - gravitational parameter [km^3 / s^2]
    coe  - orbital elements (h ecc RA incl w TA)
        where
            h    = angular momentum [km^2/s]
            ecc  = eccentricity
            RA   = right ascension of the ascending node [deg]; aka capital W
            incl = inclination of the orbit [deg]
            w    = argument of perigee [deg]
            TA   = true angle/anomaly [deg]
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
    import numpy as np

    RA = RA * math.pi / 180  # [rad] right ascension of the ascending node
    incl = incl * math.pi / 180  # [rad] inclination
    w = w * math.pi / 180  # [rad] argument of periapsis
    TA = TA * math.pi / 180  # [rad] true angle/anomaly

    # ...Equations 4.45 and 4.46 (rp and vp are column vectors):
    rp = (
        (h**2 / mu)
        * (1 / (1 + ecc * math.cos(TA)))
        * (math.cos(TA) * np.array([1, 0, 0]) + math.sin(TA) * np.array([0, 1, 0]))
    )
    rp = rp.reshape(-1, 1)  # convert to column vector
    vp = (mu / h) * (
        -math.sin(TA) * np.array([1, 0, 0]) + (ecc + math.cos(TA)) * np.array([0, 1, 0])
    )
    vp = vp.reshape(-1, 1)  # convert to column vector

    # Create rotation matrices/arrays
    # rotate z-axis thru angle RA , equation 4.34:
    # R3_RA = [ math.cos(RA)  math.sin(RA)  0
    #         -math.sin(RA)  math.cos(RA)  0
    #             0        0     1]
    c_RA, s_RA = math.cos(RA), math.sin(RA)  #
    R3_RA = np.array([[c_RA, s_RA, 0], [-s_RA, c_RA, 0], [0, 0, 1]])

    # rotation about x-axis, inclination, equation 4.32:
    # R1_i = [1       0          0
    #         0   cos(incl)  sin(incl)
    #         0  -sin(incl)  cos(incl)]
    c_in, s_in = np.cos(incl), np.sin(incl)
    R1_i = np.array([[1, 0, 0], [0, c_in, s_in], [0, -s_in, c_in]])
    print(f"incl= {incl*180/np.pi}")
    # print(f"R1_i= {R1_i}")

    # rotation about z-axis, equation 4.34:
    # R3_w = [ cos(w)  sin(w)  0
    #         -sin(w)  cos(w)  0
    #         0       0     1]
    c_w, s_w = np.cos(w), np.sin(w)
    R3_w = np.array([[c_w, s_w, 0], [-s_w, c_w, 0], [0, 0, 1]])
    # print(f"R3_w= {R3_w}")

    # Equation 4.49:
    Q_pX = R3_w @ R1_i @ R3_RA  # matrix multiply
    print(f"Q_px= {Q_pX}")
    Q_Xp = np.transpose(Q_pX)

    # Equations 4.51 (r and v are column vectors):
    r = Q_Xp @ rp
    v = Q_Xp @ vp

    # Convert r and v into row vectors:
    r = np.ravel(r)  # flatten the array
    v = np.ravel(v)  # flatten the array
    return r, v


# ***********************************************


def test_sv_from_coe():
    """
    See Curtis example 4.7.
    h, ecc, RA, incl, w, TA
    """
    print(f"test Curtis function, sv_from_coe()")
    mu_earth_km = 398600  # [km^3/s^2]
    h = 80000  # [km^2/s]
    ecc = 1.4
    RA, incl, w, TA = [40, 30, 60, 30]  # [deg]
    r1_vec, v1_vec = sv_from_coe(
        h=h, ecc=ecc, RA=RA, incl=incl, w=w, TA=TA, mu=mu_earth_km
    )
    print(f"position, r1= {r1_vec}")
    print(f"velocity, v1= {v1_vec}")

    return None


# use the following to test/examine functions
if __name__ == "__main__":

    test_sv_from_coe()  #
