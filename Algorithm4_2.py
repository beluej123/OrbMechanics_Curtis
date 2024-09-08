"""
Convert position & velocity vectors -> COE (classic orbital elements)
Curtis pp.209, algorithm 4.2, & Curtis pp.212, example 4.3.
Orbital elements (coe) -> state vectors (IJK).  Curtis p.232 , example 4.7, algorithm 4.5.

Notes on classic orbital elements angles:
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

Notes:
----------
    2024-03-09, https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html
    Orbital elements -> state vectors (IJK).  Curtis p.232 , example 4.7, algorithm 4.5.

    Uses Curtis, pp.471, algorithm 8.1.  Note Curtis p.277, example 5.4, Sideral time.
    Also see Vallado [2] functions: pp. 296, planetRV() (algotithm 33),
        cov2rv() (algorithm 11), et.al
    Orbital elements tables kept in functionCollection.py
    For my code, generally angles are saved in [rad].

    helpful interplanetary flight http://www.braeunig.us/space/interpl.htm
    
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally angles are saved in [rad], distance [km].
    
References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
    [4] Vallado, David A., (2020, 5th ed.).
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
"""

import numpy as np


# orbit type function helps with printing results
def orbit_type(e):  # returns string, orbit type
    if e > 1:
        orb_type = "hyperbola"
    elif 0 < e < 1:
        orb_type = "ellipse"
    elif e == 1:
        orb_type = "parabola"
    elif e == 0:
        orb_type = "circle"
    else:
        orb_type = "unknown"
    return orb_type


def orbit_rv_COE(r_vec, v_vec, mu):
    # given r & v find COE
    # r_vec = np.array((1000, 5000, 7000))  # km
    # v_vec = np.array((3.0, 4.0, 5.0))  # km/s
    # mu = 3.986e5  # km^3/s^2
    # distance, velocities
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    v_r = np.dot(r_vec / r, v_vec)  # velocity radial
    v_p = np.sqrt(v**2 - v_r**2)  # velocity perpendicular
    print("velocity radial (v_r)=", v_r)
    print("velocity perpendicular (v_p)=", v_p)

    # h=angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    print("angular momentum (h_vec) =", h_vec, "[km^2 / s], and (h) =", h)

    # inclination
    incl = np.arccos(h_vec[2] / h)
    incl_deg = incl * 180 / np.pi
    print("orbit inclination (incl) =", incl_deg)

    # right ascension of ascending node, Omega (or RAAN)
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    Omega = np.arccos(n_vec[0] / n)
    if n_vec[1] < 0:
        Omega = 2 * np.pi - Omega
    Omega_deg = Omega * 180 / np.pi
    print("right ascension of ascending node (Omega_deg)", Omega_deg)

    # eccentricity
    e_vec = (1 / mu) * np.cross(v_vec, h_vec) - (r_vec / r)
    e = np.linalg.norm(e_vec)  # e magnitude
    if e < 0.00005:
        e = 0.0
        theta = 0  # true anomaly undefined here; circular orbit
    else:
        theta = np.arccos(
            np.dot(e_vec, r_vec) / (e * r)
        )  # defined for non-circular orbits
        theta_deg = theta * 180 / np.pi

    print("orbit eccentricity, e =", e)
    print("orbit type =", orbit_type(e))
    if e == 0.0:
        print("true anomaly0, theta0 = not defined; circular")
    else:
        print("true anomaly0, theta0 =", theta_deg, "[deg]")

    # argument of periapsis; defined for non-circular orbits
    if e == 0.0:
        print("argument of periapsis, omega = not defined; circular")
    else:
        omega = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
        omega_deg = omega * 180 / np.pi
        print("argument of periapsis (omega_deg) =", omega_deg, "[deg]")


# below, 2 tests of orbit_rv_COE(r, v, mu)
print("**Example from hyper-link in code**")
r_vec = np.array((1000, 5000, 7000))  # km
v_vec = np.array((3.0, 4.0, 5.0))  # km/s
mu_e = 3.986e5  # earth mu [km^3/s^2]
orbit_rv_COE(r_vec, v_vec, mu_e)

# Example 4.3
print("**Curtis Example 4.3**")
r_vec = np.array((-6045, -3490, 2500))  # km
v_vec = np.array((-3.457, 6.618, 2.533))  # km/s
mu_e = 3.986e5  # earth mu [km^3/s^2]
orbit_rv_COE(r_vec, v_vec, mu_e)
