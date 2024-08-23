"""
Curtis example's 2.12 (p.116), 2.13 (p.123), 2.14 (p.124)
May help development; see https://github.com/jkloser/OrbitalMechanics

References
    ----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""
import numpy as np


def find_r(h, mu_e, r0, vr0, d_theta):
    # Curtis p.121, eqn 2.152
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


def find_f_dot(mu, h, d_theta, r0, r):
    A = mu / h
    B = (1 - np.cos(d_theta)) / (np.sin(d_theta))
    C = mu / (h**2)
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
def r1v1_from_r0v0_dnu(r0_vec, v0_vec, delta_nu, GM):
    """
    Find new position & velocity vectors.
    Curtis, p.123, algorithm 2.3
    Given:
        r0_vec, v0_vec, angle r1->r2 (delta nu)
    Find:
        r1_vec & v1_vec

    Parameters
    ----------
    r0_vec : np.array, initial position vector
    v0_vec : np.array, final velocity vector
    delta_nu : float, angle r0->r1 [rad]
    
    Returns
    ----------
        r1_vec, v1_vec, vr0, h_mag
    """
    # (1a) vector magnitude
    r0_mag = np.linalg.norm(r0_vec)
    
    # (1b) radial velocity of v0_vec
    vr0 = np.dot(v0_vec, (r0_vec / r0_mag))  # radial velocity @ r0
    
    # (1c) find h (constant), Curtis p.117, eqn 2.130
    h_mag = np.linalg.norm(np.cross(r0_vec, v0_vec)) #[km^2/s]
    
    # (1d) for r below, see Curtis p.121, eqn 2.152
    r = find_r(h_mag, GM, r0_mag, vr0, delta_nu)

    # (1e) find f, g, f_dot, g_dot, below, Curtis p.122 eqns 2.158
    f = find_f(GM, r, h_mag, delta_nu)
    g = find_g(r, r0_mag, h_mag, delta_nu)
    f_dot = find_f_dot(mu=GM, h=h_mag, d_theta=delta_nu, r0=r0_mag, r=r)
    g_dot = find_g_dot(GM, r0_mag, h_mag, delta_nu)

    # (2) find r1_vec, v1_vec, below, see Curtis p.118, eqns 2.135, 2.136
    r1_vec = find_position(f, g, r0_vec, v0_vec)
    v1_vec = find_velocity(f_dot, g_dot, r0_vec, v0_vec)
    
    return r1_vec, v1_vec, vr0, h_mag

def test_c_ex2_12():
    """
    Calculate orbital parameters given r0, v0.
    Assume perifocal frame (pqw).
    Given:
        r0_vec, v0_vec
    Find:
        h_vec, h_mag (angular momentum)
        TA (true angle/anomaly)
        ecc (eccentricity)
    Return
    -------
        None
    """
    print(f"\nCurtis Example 2.12 (p.116), Orbit Parameters (PQW Frame):")
    # constants below; mostly from Vallado not Curtis
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    r_earth = 6378.1363  # [km], Vallado p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    mu_sun = GM_sun_km # [km^3/s^2]
    
    r0_vec = np.array([7000, 9000, 0])  #[km] pqw frame
    v0_vec = np.array([-5, 7, 0])  #[km/s] pqw frame

    r0 = np.linalg.norm(r0_vec)
    h_vec = np.cross(r0_vec, v0_vec)
    h_mag = np.linalg.norm(h_vec)
    print(f"angular momentum, h_vec= {h_vec} [km^2/s]")
    print(f"angular momentum, h_mag= {h_mag:.6g} [km^2/s]")

    theta = np.arccos(r0_vec / r0)[0]  # chose p coordinate in pqw frame
    if r0_vec[1] < 0:
        theta = -theta
    theta_deg = theta * 180 / np.pi
    print(f"true anomaly= {theta_deg:.6g} [deg]")

    ecc = (((h_mag**2) / (r0 * GM_earth_km)) - 1) / np.cos(theta)
    print(f"eccentricity, ecc= {ecc:.6g}")
    return None

def test_c_ex2_13():
    """
    Find new position (r1) and new velocity (v1) vectors.
    See Curtis p.123, algorithm 2.3.
    Assume perifocal frame (pqw).
    Given:
        r0_vec, v0_vec, angle (r0->r1) [rad]
    Find:
        r1_vec, v1_vec
    Return
    -------
        None
    """
    print(f"\nCurtis Example 2.13 (p.123), Lagrange Coefficients:")
    # constants; mostly from Vallado not Curtis
    au = 149597870.7  # [km/au] Vallado p.1043, tbl.D-5
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    r_earth = 6378.1363  # [km], Vallado p.1041, tbl.D-3
    GM_sun_km = 1.32712428e11  # [km^3/s^2], Vallado p.1043, tbl.D-5
    # given:    
    r0_vec = np.array([8182.4, -6865.9, 0]) #[km]
    v0_vec = np.array([0.47572, 8.8116, 0]) #[km]
    d_angle_deg = 120  # [deg] angle from r0->r1
    d_angle_rad = d_angle_deg * (np.pi / 180) #[rad]
    
    # (1a) vector magnitude
    r0_mag = np.linalg.norm(r0_vec)
    
    # (1b) radial velocity of v0_vec
    vr0 = np.dot(v0_vec, (r0_vec / r0_mag))  # radial velocity @ r0
    
    # (1c) find h (constant), Curtis p.117, eqn 2.130
    h_mag = np.linalg.norm(np.cross(r0_vec, v0_vec)) #[km^2/s]
    
    # (1d) for r below, see Curtis p.121, eqn 2.152
    r = find_r(h_mag, GM_earth_km, r0_mag, vr0, d_angle_rad)

    # (1e) find f, g, f_dot, g_dot, below, Curtis p.122 eqns 2.158
    f = find_f(GM_earth_km, r, h_mag, d_angle_rad)
    g = find_g(r, r0_mag, h_mag, d_angle_rad)
    f_dot = find_f_dot(GM_earth_km, h_mag, d_angle_rad, r0_mag, r)
    g_dot = find_g_dot(GM_earth_km, r0_mag, h_mag, d_angle_rad)

    # (2) find r1_vec, v1_vec, below, see Curtis p.118, eqns 2.135, 2.136
    r1_vec = find_position(f, g, r0_vec, v0_vec)
    v1_vec = find_velocity(f_dot, g_dot, r0_vec, v0_vec)
    # r0 & v0 are in plane; thus no 3rd vector element
    np.set_printoptions(precision=5) # for np vector printing
    print(f"new position magnitude, r= {r:.6g} [km]")
    print(f"final position, r1_vec= {r1_vec} [km]")
    print(f"final velocity, v1_vec= {v1_vec} [km/s]")
    
    return None

def test_c_ex2_14():
    """
    Calculate orbital parameters. Example 2.14 (p.124), follow up to example 2.13
    Given:
        r0_mag, v0_mag, h_mag
    Find:
        eccentricity, true angle/anomaly at r0, location of periapsis
    Return
    -------
        None
    """
    print("\nCurtis Example 2.14 (p.124), Orbital Parameters:")
    # taken from example 2_14:
    GM_earth_km = 3.986004415e5  # [km^3/s^2], Vallado p.1041, tbl.D-3
    GM=GM_earth_km
    
    r0_vec = np.array([8182.4, -6865.9, 0]) #[km]
    v0_vec = np.array([0.47572, 8.8116, 0]) #[km]
    d_angle_deg = 120  # [deg] angle from r0->r1
    delta_nu = d_angle_deg * (np.pi / 180) #[rad]
    r0_mag = np.linalg.norm(r0_vec)
    # from problem statement, do not need r1_vec or v1_vec
    r1_vec, v1_vec, vr0, h_mag=r1v1_from_r0v0_dnu(r0_vec, v0_vec, delta_nu, GM)
    
    print(f"r0_mag= {r0_mag:.6g} [km]")
    if vr0 < 0:
        print(f"vr0<0, approaching periapsis: vr0= {vr0:.6g} [km/s]")
    else:
        print(f"vr0>=0, leaving periapsis: vr0= {vr0:.6g} [km/s]")
    print(f"angular momentum, h_mag= {h_mag:.6g} [km^2/s]")

    # used Maple to verify algebra for ecc expression
    ecc_sq = (h_mag**2 / (r0_mag * GM) - 1) ** 2 + (vr0**2) * h_mag**2 / GM**2
    ecc = np.sqrt(ecc_sq)
    print(f"eccentricity, ecc= {ecc:.6g}")

    theta = np.arccos((h_mag**2 - GM * r0_mag) / (r0_mag * GM * ecc))
    theta_deg = theta * 180 / np.pi
    if vr0 < 0:
        theta = 2 * np.pi - theta
        theta_deg = theta * 180 / np.pi
        print(f"vr0<0, approaching, theta_deg= {theta_deg:.6g} [deg]")
    else:
        print(f"vr0>=0, leaving, theta_deg= {theta_deg:.6g} [deg]")
    
    return None


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

# Guides tests & functions.
if __name__ == "__main__":
    # test_c_ex2_12()  # Curtis example 2.12
    # test_c_ex2_13()  # Curtis example 2.13
    test_c_ex2_14()  # Curtis example 2.14