# 2024-03-08, Collection of H.D.Curtis matlab functions converted to python.
# the following is an on-line matlab -> python converter
# https://www.codeconvert.ai/matlab-to-python-converter
import numpy as np


def lambert(R1, R2, t, string):
    global mu

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
        print("\n\n **Number of iterations exceeds #g in " "lambert" " \n\n ") # nmax

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


# Stumpff functions:
def C(z):
    return stumpC(z)


def S(z):
    return stumpS(z)


###############################################
def planet_elements_and_sv(planet_id, year, month, day, hour, minute, second):
    global mu
    deg = math.pi / 180

    # Equation 5.48:
    j0 = J0(year, month, day)

    ut = (hour + minute / 60 + second / 3600) / 24

    # Equation 5.47
    jd = j0 + ut

    # Obtain the data for the selected planet from Table 8.1:
    J2000_coe, rates = planetary_elements(planet_id)

    # Equation 8.93a:
    t0 = (jd - 2451545) / 36525

    # Equation 8.93b:
    elements = J2000_coe + rates * t0

    a = elements[0]
    e = elements[1]

    # Equation 2.71:
    h = math.sqrt(mu * a * (1 - e**2))

    # Reduce the angular elements to within the range 0 - 360 degrees:
    incl = elements[2]
    RA = elements[3] # 360
    w_hat = elements[4] # 360
    L = elements[5] # 360
    w = (w_hat - RA) # 360
    M = (L - w_hat) # 360

    # Algorithm 3.1 (for which M must be in radians)
    E = kepler_E(e, M * deg)  # rad

    # Equation 3.13 (converting the result to degrees):
    TA = (2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))) # 360

    coe = [h, e, RA * deg, incl * deg, w * deg, TA * deg]

    # Algorithm 4.5:
    r, v = sv_from_coe(coe, mu)

    return coe, r, v, jd


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

    au = 149597871
    J2000_coe[0] = J2000_coe[0] * au
    rates[0] = rates[0] * au

    return J2000_coe, rates
################################

def sv_from_coe(coe, mu):
    """
This function computes the state vector (r,v) from the
classical orbital elements (coe).
 
mu   - gravitational parameter (km^3 / s^2)
coe  - orbital elements [h e RA incl w TA]
        where
             h    = angular momentum (km^2/s)
             e    = eccentricity
             RA   = right ascension of the ascending node (rad)
             incl = inclination of the orbit (rad)
             w    = argument of perigee (rad)
             TA   = true anomaly (rad)
    R3_w - Rotation matrix about the z-axis through the angle w
    R1_i - Rotation matrix about the x-axis through the angle i
    R3_W - Rotation matrix about the z-axis through the angle RA
    Q_pX - Matrix of the transformation from perifocal to geocentric 
            equatorial frame
    rp   - position vector in the perifocal frame (km)
    vp   - velocity vector in the perifocal frame (km/s)
    r    - position vector in the geocentric equatorial frame (km)
    v    - velocity vector in the geocentric equatorial frame (km/s)

    User M-functions required: none
"""
    import numpy as np
    import math
    h    = coe(1)
    e    = coe(2)
    RA   = coe(3)
    incl = coe(4)
    w    = coe(5)
    TA   = coe(6)

    #...Equations 4.45 and 4.46 (rp and vp are column vectors):
    rp = (h**2/mu) * (1/(1 + e*math.cos(TA))) * (math.cos(TA)*[1;0;0] + sin(TA)*[0;1;0])
    vp = (mu/h) * (-math.sin(TA)*[1;0;0] + (e + cos(TA))*[0;1;0])

    #...Equation 4.34:
    R3_W = [ math.cos(RA)  sin(RA)  0
            -sin(RA)  cos(RA)  0
                0        0     1]

    #...Equation 4.32:
    R1_i = [1       0          0
            0   cos(incl)  sin(incl)
            0  -sin(incl)  cos(incl)]

    #...Equation 4.34:
    R3_w = [ cos(w)  sin(w)  0 
            -sin(w)  cos(w)  0
            0       0     1]

    #...Equation 4.49:
    Q_pX = (R3_w*R1_i*R3_W)

    #...Equations 4.51 (r and v are column vectors):
    r = Q_pX*rp
    v = Q_pX*vp

    #...Convert r and v into row vectors:
    r = r
    v = v
    return r, v

#testing

