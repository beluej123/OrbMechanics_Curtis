"""
Exploring methods to define celestial body parameters that
    may be retrieved or written to by other functions.
    Began with ideas in my valladopy directory, solarsys.py.
2024-11-10, older data organization is just below these notes; dataclass follows...
2024-11-08, not sure what makes sense for parameter data storage and retrieval.
    dataclass or class or something else (i.e. import txt file with formatted data?)
Solar system parameters/constants:
    dataclass's organized; orbit data & body data.

References:
----------
    See references.py for references list.

Notes:
----------
    2024-11-07, exploring schemes to capture celestial body data.
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally, angles are saved in [rad], distance [km].

    https://www.dataquest.io/blog/how-to-use-python-data-classes/
    Search, ArjanCodes and dataclass, https://www.youtube.com/watch?v=lX9UQp2NwTk

Orbital Elements Naming Collection:
Start with Kepler coe (classic orbital elements).
    https://ssd.jpl.nasa.gov/planets/approx_pos.html
    Horizons web-ephemeris, https://ssd.jpl.nasa.gov/horizons/app.html#/

    o_type : int  , [-] orbit type (python dictionary list)
                    0:"circular", 1:"circular inclined", 2:"circular equatorial"
                    3:"elliptical", 4:"elliptical equatorial"
                    5:"parabolic", 6:"parabolic equatorial"
                    7:"hyperbolic", 8:"hyperbolic equatorial"
    sp     : float, [km or au] semi-parameter (aka p)
    sma    : float, [km or au] semi-major axis (aka a)
    ecc    : float, [--] eccentricity
    incl   : float, [rad] inclination
    raan   : float, [rad] right ascension of ascending node,
                    also called Longitude of Ascending Node (Omega, or capital W)
    w_     : float, [rad] argument of periapsis (aka aop, or arg_p)
    TA     : float, [rad] true angle/anomaly (aka t_anom, or theta)

    alternative coe's including circular & equatorial:
    Lt0    : float, [rad] true longitude at epoch, circular equatorial
                    Position on the ecliptic, accounting for its inclination.
                    when incl=0, ecc=0
    w_bar  : float, [rad] longitude of periapsis (aka II), equatorial
                NOTE ** NOT argument of periapsis, w_ ??????????????????? **
                Note, w_bar = w_ + raan, measured in 2 planes (Vallado [4] p.1015)
    u_     : float, [rad] argument of lattitude (aka ), circular inclined

    Other orbital elements:
        w_p    : float [rad] longitude of periapsis (aka w_bar) ??
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
        use longitude of periapsis, II (aka w_bar); II=acos(e_vec[0]/e_mag
        If e_vec[1] < 0 then 180 < II < 360 degree

    circular & equatorial, e=0 and i=0 or i=180: w_ and raan and TA = undefined;
        use true longitude, Lt0 = angle between r0 & I-axis; Lt0=acos(r_vec[1]/r_mag)
        If r_mag[1] < 0 then 180 < Lt0 < 360 degree

From JPL Horizizons, osculating elements:
    Symbol & meaning [1 au= 149597870.700 km, 1 day= 86400.0 s]:
    JDTDB  Julian Day Number, Barycentric Dynamical Time
    EC     Eccentricity, e
    QR     Periapsis distance, q (au)
    IN     Inclination w.r.t X-Y plane, i (degrees)
    OM     Longitude of Ascending Node, OMEGA, (degrees)
    W      Argument of Perifocus, w (degrees)
    Tp     Time of periapsis (Julian Day Number)
    N      Mean motion, n (degrees/day)
    MA     Mean anomaly, M (degrees)
    TA     True anomaly, nu (degrees)
    A      Semi-major axis, a (au)
    AD     Apoapsis distance (au)
    PR     Sidereal orbit period (day)

From my version (elemLib.py) of skyfield'sd osculating elements attributes:
    01) sma = semi-major axis (aka a)
    02) b = semi-minor axis
    03) ecc_mag = eccentricity
    04) E = eccentric anomaly
    05) h_mag = specific angular momentum magnitude
    06) incl = inclination
    07) l = true longitude
    08) L = mean longitude
    09) M = mean anomaly
    10) n = mean motion
    11) Om = longitude of ascending node (aka RAAN)
    12) p = semi-latus rectum
    13) P = period (capital P)
    14) q = periapsis distance
    15) Q = apoapsis distance
    16) t = time (given t0)
    17) tp = time since periapsis
    18) u = argument of latitude
    19) nu = true anomaly (aka ta, sometimes v)
    20) w = argument of periapsis
    21) lp = longitude of periapsis
"""

import numpy as np
from scipy.integrate import solve_ivp

import constants  # includes general unuts conversions


class Body:
    """one line description"""

    def __init__(self, name, mass, radius, parent_body, position, velocity, mu):
        self.name = name
        self.mass = mass
        self.radius = radius  # body radius
        self.mu = constants.G_ * mass

    def __repr__(self):
        return f"Body(name={self.name}, mass={self.mass}, radius={self.radius}, mu={self.mu})"


class Orbit:
    """one line description"""

    def __init__(
        self,
        semi_major_axis,
        eccentricity,
        inclination,
        argument_of_periapsis,
        longitude_of_ascending_node,
        true_anomaly,
        orbiting_body,
    ):
        self.semi_major_axis = semi_major_axis  # aka a
        self.eccentricity = eccentricity  # aka ecc
        self.inclination = inclination  # aka inc
        self.argument_of_periapsis = argument_of_periapsis
        self.longitude_of_ascending_node = longitude_of_ascending_node
        self.true_anomaly = true_anomaly
        self.orbiting_body = orbiting_body


class Spacecraft:
    """one line description"""

    def __init__(
        self,
        name,
        mass,
        t0,
        r0=0.0,
        v0=0.0,
        engine_type="",
        fuel_capacity=0,
        current_fuel=0,
    ):
        """
        Initialize Spacecraft object.
        Input:
        ----------
            name (str): Spacecraft name
            t0 : [] time initial state (r0, v0), astropy or python time not sure which one
            r0 (np.ndarray): Initial position vector in Cartesian coordinates (km)
            v0 (np.ndarray): Initial velocity vector in Cartesian coordinates (km/s)
        """
        self.name = name
        self.mass = mass
        self.t0 = t0  # state epoch time
        self.state0 = np.concatenate((r0, v0))
        self.r0 = np.array(position)  #
        self.v0 = np.array(velocity)
        self.engine_type = engine_type
        self.fuel_capacity = fuel_capacity
        self.current_fuel = current_fuel

    def calculate_delta_v(self, target_orbit):
        # placeholder for delta-v calculation
        pass

    def apply_thrust(self, delta_v, duration):
        # placeholder for thrust application
        pass


# 2-Body Propagation
class B2_Prop:
    """one line description"""

    def __init__(self, central_body, initial_state, time_span):
        self.central_body = central_body
        self.initial_state = np.array(
            initial_state
        )  # Position and velocity (x, y, z, vx, vy, vz)
        self.time_span = time_span  # Time span for orbit propagation
        self.trajectory = None  # Store the trajectory here

    def two_body_equations(self, time, state):
        # Extract position and velocity from state
        position = state[:3]
        velocity = state[3:]

        # Calculate distance from central body
        r = np.linalg.norm(position)

        # Calculate gravitational acceleration
        mu = (
            self.central_body.mass * 6.6743e-11
        )  # Gravitational constant * mass of central body
        acceleration = -mu * position / (r**3)

        # Return the derivatives of the state
        dstate_dt = np.concatenate((velocity, acceleration))
        return dstate_dt

    def propagate(self):
        # Solve the differential equation using solve_ivp
        solution = solve_ivp(
            self.two_body_equations,
            self.time_span,
            self.initial_state,
            method="RK45",
            dense_output=True,
        )

        # Store the trajectory
        self.trajectory = solution.y.T
        return self.trajectory

    def set_orbit(self, orbit):
        self.orbit = orbit

    # propagate; put these def inside B2_Propagate
    def equations_of_motion(self, t, state):
        """
        Defines the equations of motion for the spacecraft.

        Args:
            t (float): Time (s).
            state (np.ndarray): State vector [rx, ry, rz, vx, vy, vz].

        Returns:
                np.ndarray: Derivative of the state vector [vx, vy, vz, ax, ay, az].
        """
        r = state[0:3]
        v = state[3:6]
        norm_r = np.linalg.norm(r)
        a = -self.mu * r / norm_r**3
        return np.concatenate((v, a))

    def propagate_orbit(self, t_span, t_eval):
        """
        Propagates the orbit of the spacecraft.

        Args:
            t_span (tuple): Time span for the propagation (t_start, t_end) in seconds.
            t_eval (np.ndarray): Array of time values at which to evaluate the solution.

        Returns:
            scipy.integrate._ivp.OdeResult: Solution of the ODE integration.
        """
        solution = solve_ivp(
            self.equations_of_motion,
            t_span,
            self.state0,
            t_eval=t_eval,
            dense_output=True,
        )
        return solution


# *****************

# Example Usage
# earth = Body("Earth", 5.972e24, 6371000)  # Mass in kg, radius in meters
# iss_orbit = Orbit(
#     400000 + earth.radius, 0.0005, 51.6, 170, 235, 45, earth
# )  # Example ISS orbit parameters
# iss = Spacecraft("International Space Station", 420000, iss_orbit, "Chemical", 7000)

# print(f"{iss.name} is orbiting {iss.current_orbit.orbiting_body.name}")
# print(f"Semi-major axis: {iss.current_orbit.semi_major_axis} m")


# ************************************************


class Spacecraft:
    """
    A class representing a spacecraft and its orbital motion.
    """

    MU_EARTH = 3.986004418e14  # Gravitational parameter of Earth (m^3/s^2)

    def __init__(self, a, e, i, raan, argp, nu, epoch=0):
        """
        Initializes a Spacecraft object with orbital elements.

        Args:
            a (float): Semi-major axis (m)
            e (float): Eccentricity
            i (float): Inclination (rad)
            raan (float): Right ascension of the ascending node (rad)
            argp (float): Argument of periapsis (rad)
            nu (float): True anomaly (rad)
            epoch (float): Epoch time (s), default is 0
        """
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.argp = argp
        self.nu = nu
        self.epoch = epoch
        self._update_cartesian_state()

    def _update_cartesian_state(self):
        """
        Updates the Cartesian state (position and velocity) based on current orbital elements.
        """
        E = self._true_to_eccentric_anomaly(self.nu, self.e)
        x_perifocal = self.a * (np.cos(E) - self.e)
        y_perifocal = self.a * np.sqrt(1 - self.e**2) * np.sin(E)
        x_dot_perifocal = (
            -np.sqrt(self.MU_EARTH / self.a) * np.sin(E) / (1 - self.e * np.cos(E))
        )
        y_dot_perifocal = (
            np.sqrt(self.MU_EARTH / self.a)
            * np.sqrt(1 - self.e**2)
            * np.cos(E)
            / (1 - self.e * np.cos(E))
        )

        rotation_matrix = self._perifocal_to_eci_matrix(self.i, self.raan, self.argp)

        r_perifocal = np.array([x_perifocal, y_perifocal, 0])
        v_perifocal = np.array([x_dot_perifocal, y_dot_perifocal, 0])

        self.r = np.dot(rotation_matrix, r_perifocal)
        self.v = np.dot(rotation_matrix, v_perifocal)

    def _perifocal_to_eci_matrix(self, i, raan, argp):
        """
        Creates the rotation matrix from perifocal to ECI frame.
        """
        r_z_raan = np.array(
            [
                [np.cos(raan), -np.sin(raan), 0],
                [np.sin(raan), np.cos(raan), 0],
                [0, 0, 1],
            ]
        )
        r_x_i = np.array(
            [[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]]
        )
        r_z_argp = np.array(
            [
                [np.cos(argp), -np.sin(argp), 0],
                [np.sin(argp), np.cos(argp), 0],
                [0, 0, 1],
            ]
        )

        return np.dot(np.dot(r_z_raan, r_x_i), r_z_argp)

    def _true_to_eccentric_anomaly(self, nu, e):
        """
        Converts true anomaly to eccentric anomaly.
        """
        return np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), np.cos(nu) + e)

    def propagate(self, time_of_flight, dt=10):
        """
        Propagates the spacecraft's orbit using the two-body equation of motion.

        Args:
            time_of_flight (float): Time to propagate (s)
                dt (float): Time step for propagation (s)
        """
        t_eval = np.arange(0, time_of_flight + dt, dt)
        sol = solve_ivp(
            self._two_body_equation,
            (0, time_of_flight),
            np.concatenate((self.r, self.v)),
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10,
        )

        self.r = sol.y[:3, -1]
        self.v = sol.y[3:, -1]
        self.epoch += time_of_flight
        self._update_orbital_elements()

    def _two_body_equation(self, t, y):
        """
        Defines the two-body equation of motion.
        """
        r_vec = y[:3]
        v_vec = y[3:]
        norm_r = np.linalg.norm(r_vec)
        a_vec = -self.MU_EARTH * r_vec / norm_r**3
        return np.concatenate((v_vec, a_vec))

    def _update_orbital_elements(self):
        """
        Updates the orbital elements based on the current Cartesian state.
        """
        h_vec = np.cross(self.r, self.v)
        h = np.linalg.norm(h_vec)
        e_vec = (np.cross(self.v, h_vec) / self.MU_EARTH) - (
            self.r / np.linalg.norm(self.r)
        )
        self.e = np.linalg.norm(e_vec)

        self.a = 1 / (
            (2 / np.linalg.norm(self.r)) - (np.linalg.norm(self.v) ** 2 / self.MU_EARTH)
        )

        self.i = np.arccos(h_vec[2] / h)

        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        if n != 0:
            self.raan = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                self.raan = 2 * np.pi - self.raan
        else:
            self.raan = 0

        if self.e != 0:
            self.argp = np.arccos(np.dot(n_vec, e_vec) / (n * self.e))
            if e_vec[2] < 0:
                self.argp = 2 * np.pi - self.argp
        else:
            self.argp = 0

        if self.e != 0:
            self.nu = np.arccos(
                np.dot(e_vec, self.r) / (self.e * np.linalg.norm(self.r))
            )
            if np.dot(self.r, self.v) < 0:
                self.nu = 2 * np.pi - self.nu
        else:
            self.nu = np.arccos(np.dot(n_vec, self.r) / (n * np.linalg.norm(self.r)))
            if self.r[2] < 0:
                self.nu = 2 * np.pi - self.nu


# ************************************************


class Orbit(Body):
    """one line description"""
    def __init__(self, central_body, orbiting_body, G):
        self.central_body = central_body
        self.orbiting_body = orbiting_body
        self.G = G
        self.calc_elements

    def calc_elements(self):
        r = self.orbiting_body.position - self.central_body.position
        v = self.orbiting_body.velocity
        mu = self.G * (self.central_body.mass + self.orbiting_body.mass)

        h = np.cross(r, v)
        ecc_vec = np.cross(v, h) / mu - (r / np.linalg.norm(r))
        ecc_mag = np.linalg.norm(ecc_vec)
        if ecc_mag < 1:
            sma = -mu / np.linalg.norm(v) ** 2  # aka 'a'
            self.period = 2 * np.pi * np.sqrt(sma**3 / mu)
        else:
            self.period = float("inf")

        self.semi_major_axis = sma if ecc_mag < 1 else float("inf")
        self.ecc_mag = ecc_mag
        self.inc = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))

    def __repr__(self):
        return f"Orbit(central_body={self.central_body.name}, \
                        orbiting_body={self.orbiting_body.name}, \
                        semi_major_axis={self.semi_major_axis:.4g}, \
                        ecc_mag={self.ecc_mag}, \
                        inc={self.inc:.4g}, \
                        period={self.period:.4g})"


class Moon(Body):
    # Orbit Params
    sma_ER = 60.27
    sma_km = 384400.0
    ecc = 0.05490
    incl_deg = 5.145396
    raan_deg = None
    long_perihelion = None
    true_long = None
    period_yrs = 0.0748
    tropical_days = 27.321582
    orb_vel = 1.0232  # km/s

    # Body Params
    eq_radius_km = 1738.0
    flat = None
    mu = 4902.799
    mass_norm = 0.01230
    mass_kg = 7.3483e22
    rot_days = 27.32166
    eq_inc = 6.68
    j2 = 0.0002027
    j3 = None
    j4 = None
    density = 3.34


class Earth(Body):
    # Orbit Params
    sma_au = 1.00000100178
    sma_km = 149598023.0
    ecc = 0.016708617
    incl = 0.0
    raan_deg = 0.0
    long_perihelion = 102.93734808
    true_long = 100.46644851
    period_yrs = 0.99997862

    # Body Params
    eq_radius_km = 6378.1363
    flag = 0.0033528131
    mu = 3.986004415e5
    mass_norm = 1.0
    mass_kg = 5.9742e24
    rot_days = 0.99726968
    eq_inc = 23.45
    j2 = 0.0010826269
    j3 = -0.0000025323
    j4 = -0.0000016204
    density = 5.515


class OrbitParameters:

    # Constructor method initializes an instance of the class with the given values.
    def __init__(self, sma, ecc, incl, raan, w_, nu, mu=398600.4418):
        """
        Initialize an OrbitalParameters object.
        Args:
            a (float)   : [km] Semi-major axis
            e (float)   : Eccentricity
            i (float)   : [rad] Inclination
            raan (float): [rad] Right ascension of ascending node
            w_ (float)  : [rad] Argument of perigee, aka argp
            nu (float)  : [rad] True anomaly
            mu (float)  : [km^3/s^2] Gravitational parameter
                            Defaults to Earth's value
        """
        self.sma = sma  # [km] Semi-major axis
        self.ecc = ecc  # Eccentricity
        self.incl = incl  # [rad] Inclination
        self.raan = raan  # [rad] Right Ascension of the Ascending Node (omega)
        self.w_ = w_  # [rad] Argument of Perigee
        self.nu = nu  # [rad] True Anomaly
        self.mu = mu  # [km^3/s^2] Gravitational parameter

    def __str__(self):
        """
        The string representation method prints orbital parameters
            in a human-readable format.
        """
        return (
            f"Orbital Parameters:\n"
            f"Semi-major axis (sma): {self.sma} km\n"
            f"Eccentricity (ecc): {self.ecc}\n"
            f"Inclination (incl): {self.incl} rad\n"
            f"RAAN (omega): {self.raan} rad\n"
            f"Argument of Perigee (w_): {self.w_} rad\n"
            f"True Anomaly (nu): {self.nu} rad\n"
            f"Gravitational Parameter (mu): {self.mu} km^3/s^2"
        )


# orbit = OrbitParameters(sma=7000, ecc=0.05, incl=0.5, raan=1.2, w_=2.5, nu=0.8)
# print(orbit)


@dataclass(frozen=False, kw_only=True, slots=True)
class StarParms:  # includes solar constants
    au_: float  # derrived from earth orbit
    mu: float  # [km^3/s^2] Gravitational parameter
    mass_kg: float  # [kg] star mass


sun_prms = StarParms(
    au_=149598023.0,  # [km], Vallado [4] p.1059, tbl.D-5
    mu=1.32712428e11,  # [km^3/s^2], Vallado [4] p.1059, tbl.D-5
    mass_kg=1.9891e30,  # [kg], Vallado [4] p.1059, tbl.D-5
)


@dataclass(frozen=False, kw_only=True, slots=True)
class OrbitParms:
    ref_plane: str  # data reference; i.e. ecliptic (ecl), equatorial (equ)
    ref_jd: float  # data associated with Julian date
    sma: float  # [km] Semi-major axis
    ecc: float  # Eccentricity
    incl: float  # [rad] Inclination
    raan: float  # [rad] Right Ascension of the Ascending Node (omega)
    # w_bar, longitude of periapsis (aka II), equatorial
    w_bar: float  # [rad] longitude of periapsis; w_ + raan

    # Lt0=true longitude at epoch = TA + w_bar
    Lt0: float  # [rad] true longitude at epoch


@dataclass(frozen=False, kw_only=True, slots=True)
class BodyParams:
    eq_radius_km: float
    flatten: float  # reciprocal falttening
    mu: float  # [km^3/s^2] Gravitational parameter
    mass_norm: float
    mass_kg: float
    rot_days: float  # [days] day rotation
    eq_inc: float  # equator inclination to orbit
    j2: float
    j3: float
    j4: float
    density: float


# @dataclass(frozen=False, kw_only=True, slots=True)
# class Body_Data:
#     b_name: str  # body name
#     OrbitParms: OrbitParms
#     BodyParams: BodyParams


mercury_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or ICRF equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=57909083,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.205631752,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=7.00498625 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=48.33089304 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=77.45611904 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=252.25090551 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
mercury_b_prms = BodyParams(
    eq_radius_km=2439.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    mu=2.2032e4,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=0.0552743,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=3.3022e23,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=58.6462,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=0 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.00006,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    density=5.43,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

venus_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or ICRF equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=108208601,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.006771882,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=3.39446619 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=76.67992016 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=131.56370724 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=181.97980084 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
venus_b_prms = BodyParams(
    eq_radius_km=6052.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    mu=3.257e5,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=0.815,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=4.869e24,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=-243.01,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=177.3 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.000027,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    density=5.24,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

earth_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or ICRF equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=149598023.0,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.016708617,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=0.0,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=0.0,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=102.93734808 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=100.46644851 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
earth_b_prms = BodyParams(
    eq_radius_km=6378.1363,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0033528131,  # [], Vallado [4] p.1057, tbl.D-3
    mu=3.986004415e5,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=1.0,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=5.9742e24,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=0.99726968,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=23.45 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.0010826269,  # [], Vallado [4] p.1057, tbl.D-3
    j3=-0.0000025323,  # [], Vallado [4] p.1057, tbl.D-3
    j4=-0.0000016204,  # [], Vallado [4] p.1057, tbl.D-3
    density=5.515,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

mars_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=227939186,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.09340062,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=1.84972648 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=49.55809321 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=336.06023398 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=355.43327463 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
mars_b_prms = BodyParams(
    eq_radius_km=3397.2,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0064763,  # [], Vallado [4] p.1057, tbl.D-3
    mu=4.305e4,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=0.10744,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=6.4191e23,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=1.02595675,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=25.19 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.001964,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.00036,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0,  # [], Vallado [4] p.1057, tbl.D-3
    density=3.94,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

jupiter_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=778298361,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.048494851,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=1.30326966 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=100.46444064 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=14.33130924 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=34.35148392 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
jupiter_b_prms = BodyParams(
    eq_radius_km=71492.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0648744,  # [], Vallado [4] p.1057, tbl.D-3
    mu=1.268e8,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=318.0,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=1.8988e27,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=0.41354,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=3.12 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.01475,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=-0.00058,  # [], Vallado [4] p.1057, tbl.D-3
    density=1.33,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

saturn_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=1429394133,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.055508622,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=2.4888781 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=113.6655237 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=93.05678728 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=50.07747138 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
saturn_b_prms = BodyParams(
    eq_radius_km=60268.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0979624,  # [], Vallado [4] p.1057, tbl.D-3
    mu=3.794e7,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=95.159,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=5.685e26,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=0.4375,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=26.73 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.01645,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=-0.001,  # [], Vallado [4] p.1057, tbl.D-3
    density=1.33,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

uranus_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=2875038615,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.046295898,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=0.77319617 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=74.00594723 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=173.00515922 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=314.05500511 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
uranus_b_prms = BodyParams(
    eq_radius_km=25559.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0229273,  # [], Vallado [4] p.1057, tbl.D-3
    mu=5.794e6,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=14.4998,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=8.6625e25,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=-0.65,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=97.86 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.012,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    density=1.30,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

neptune_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=4504449769,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.008988095,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=1.76995221 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=131.78405702 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=48.12369050 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=304.34866548 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
neptune_b_prms = BodyParams(
    eq_radius_km=24764.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0171,  # [], Vallado [4] p.1057, tbl.D-3
    mu=6.809e6,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=17.203,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=1.0278e26,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=0.768,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=29.56 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.004,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    density=1.76,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)

pluto_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=5915799000,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.24905,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=17.14216667 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=110.29713889 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis; w_ + raan
    w_bar=224.134861 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    # Lt0=true longitude at epoch = TA + w_bar
    Lt0=238.74394444 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
pluto_b_prms = BodyParams(
    eq_radius_km=1151.0,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    mu=9.0e2,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=0.00251,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=1.5e22,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=-6.3867,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=118 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0.0,  # [], Vallado [4] p.1057, tbl.D-3
    density=1.1,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)


# print(f"{earth_o_prms.sma} [km]")  # [km] earth orbit parameters
# print(f"{earth_b_prms.eq_inc} [rad]")  # [rad] earth body parameters

# add new attribute to class
# earth_params.b_name = "earth" # linter has a problem with this...
# print(f"{earth_params.b_name}")

# field_list = fields(earth_params)
# for field in field_list:
#     print(f"{field.name}, {getattr(earth_params, field.name)}")


def test_o_mech_classes():
    # define orbit at t0
    name = "earth"
    earth_mass = 1.0
    earth_p_t0 = np.array([1.0, 1.0, 1.0])
    earth_v_t0 = np.array([1.0, 1.0, 1.0])
    earth_body = Body(
        name=name, mass=earth_mass, position=earth_p_t0, velocity=earth_v_t0
    )
    print(f"Earth body attributes: {earth_body}")

    earth_ele_t0 = Orbit.calc_elements()


def main():
    # just a placeholder to help with editor navigation:--)
    return


# use the following to test/examine functions
if __name__ == "__main__":
    test_o_mech_classes()  # exploring creation of orbital mech classes

    # Example Usage propagate
    # Earth parameters
    mu_earth = 398600.0  # km^3/s^2
    # Initial conditions (example: LEO)
    r0 = np.array([6500, 0, 0])  # km
    v0 = np.array([0, 8, 0])  # km/s

    # Create spacecraft object
    spacecraft = Spacecraft("MySatellite", mu_earth, r0, v0)

    # Propagation time
    t_span = (0, 3600 * 2)  # Propagate for 2 hours
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # Evaluate at 100 points

    # Propagate the orbit
    solution = spacecraft.propagate_orbit(t_span, t_eval)

    # Extract results
    time = solution.t
    position = solution.y[0:3, :]
    velocity = solution.y[3:6, :]

    # Print some results
    print(f"Spacecraft: {spacecraft.name}")
    print("Time (s) | Position (km) | Velocity (km/s)")
    for i in range(len(time)):
        print(f"{time[i]:.2f} | {position[:, i]} | {velocity[:, i]}")
