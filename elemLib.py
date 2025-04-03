"""
Goal produce orbital elements and convert between elements.
Osculating elements calculations, matches NASA HORIZONS.
2025-03-04, JBelue edited from skyfield repo, elementslib.py
2025-04-02, still figuring out how best to manage units without adding a lot
    of computational and memory overhead.
NOTE: may speedup execution, perhapse using @reify like in the skyfield repo.

Produce the osculating orbital elements for a position and velocity at t0.
    Not yet figured how to manage coordinate frames; ICRF (equatorial, ecliptic).
    Skyfield.positionlib helps manage coordinate frames that is integrated
        with solar system body references; generally the reference_frame is
        an optional argument as a 3x3 numpy array.
        The reference frame by default is the ICRF. Commonly used reference
        frames are found in skyfield.data.spice.inertial_frames.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
from astropy import units as u  # manage variable units
from numpy import (
    arccos,
    arctan,
    arctan2,
    arctanh,
    array,
    cross,
    divide,
    float64,
    inf,
    ones_like,
    pi,
    repeat,
    sin,
    sinh,
    sqrt,
    tan,
    where,
    zeros_like,
)
from pint import UnitRegistry  # manage variable units

from constants import AU_, DAY_S, GM_SUN, tau
from functions import angle_between, dots, length_of, reify
from units import Angle, Distance, Velocity

ureg = UnitRegistry()
Q_= ureg.Quantity

class OsculatingElements(object):
    """
    Osculating orbital elements.

    An ``OsculatingElements`` object initialized with:
    position_vec : assume ecliptic frame
        Position vector with shape (3,) or (3, n)
    velocity_vec : velocity object
        Velocity vector with shape (3,) or (3, n)
    position.t   : time, time object
        time of the position and velocity vectors
    mu_km_s: float
        Gravitational parameter (G*M) in units of km^3/s^2

    """

    def __init__(self, position, velocity, time, mu_km_s):
        if mu_km_s <= 0:
            raise ValueError(
                "mu_km_s (standard gravitational parameter [km^3/s^2]"
                "must be positive and non-zero."
            )

        self._pos_vec = position
        self._vel_vec = velocity
        # self._pos_vec = position.km
        # self._vel_vec = velocity.km_per_s
        self.time = time
        self._mu = mu_km_s
        self._mu_km_d = mu_km_s * (DAY_S * DAY_S)

        self._h_vec = cross(self._pos_vec, self._vel_vec, 0, 0).T
        self._e_vec = eccentricity_vector(self._pos_vec, self._vel_vec, self._mu)
        self._n_vec = node_vector(self._h_vec)

    @reify
    def apoapsis_distance(self):
        Q = apoapsis_distance(self.semi_latus_rectum.km, self.eccentricity)
        return Distance(km=Q)

    @reify
    def argument_of_latitude(self):
        w = self.argument_of_periapsis.radians
        v = self.true_anomaly.radians
        u = (w + v) % tau  # modulo 2pi
        return Angle(radians=u)

    @reify
    def argument_of_periapsis(self):
        w = argument_of_periapsis(
            self._n_vec, self._e_vec, self._pos_vec, self._vel_vec
        )
        return Angle(radians=w)

    @reify
    def eccentric_anomaly(self):
        E = eccentric_anomaly(
            self.true_anomaly.radians, self.eccentricity, self.semi_latus_rectum.km
        )
        return Angle(radians=E)

    @reify
    def eccentricity(self):
        return length_of(self._e_vec)

    @reify
    def inclination(self):
        i = inclination(self._h_vec)
        return Angle(radians=i)

    @reify
    def longitude_of_ascending_node(self):
        Om = longitude_of_ascending_node(self.inclination.radians, self._h_vec)
        return Angle(radians=Om)

    @reify
    def longitude_of_periapsis(self):
        Om = self.longitude_of_ascending_node.radians
        w = self.argument_of_periapsis.radians
        lp = (Om + w) % tau
        return Angle(radians=lp)

    @reify
    def mean_anomaly(self):
        M = mean_anomaly(self.eccentric_anomaly.radians, self.eccentricity)
        return Angle(radians=M)

    @reify
    def mean_longitude(self):
        L = (
            self.longitude_of_ascending_node.radians
            + self.argument_of_periapsis.radians
            + self.mean_anomaly.radians
        ) % tau
        return Angle(radians=L)

    @reify
    def mean_motion_per_day(self):
        n = mean_motion(self.semi_major_axis.km, self._mu)
        return Angle(radians=n * DAY_S)

    @reify
    def periapsis_distance(self):
        q = periapsis_distance(self.semi_latus_rectum.km, self.eccentricity)
        return Distance(km=q)

    @reify
    def periapsis_time(self):
        M = mean_anomaly(
            self.eccentric_anomaly.radians,
            self.eccentricity,
            shift=False,
        )
        tp = time_since_periapsis(
            M,
            self.mean_motion_per_day.radians,
            self.true_anomaly.radians,
            self.semi_latus_rectum.km,
            self._mu_km_d,
        )
        # 2025-03-14; pi-day. Not sure about time conversion here
        # ts = self.time.ts
        # times = self.time.tdb - tp
        # return ts.tdb(jd=times)
        return self.time - timedelta(days=tp)

    @reify
    def period_in_days(self):
        P = period(self.semi_major_axis.km, self._mu)
        return P / DAY_S

    @reify
    def semi_latus_rectum(self):
        p = semi_latus_rectum(self._h_vec, self._mu)
        return Distance(km=p)

    @reify
    def semi_major_axis(self):
        a = semi_major_axis(self.semi_latus_rectum.km, self.eccentricity)
        return Distance(km=a)

    @reify
    def semi_minor_axis(self):
        b = semi_minor_axis(self.semi_latus_rectum.km, self.eccentricity)
        return Distance(km=b)

    @reify
    def true_anomaly(self):
        v = true_anomaly(self._e_vec, self._pos_vec, self._vel_vec, self._n_vec)
        return Angle(radians=v)

    @reify
    def true_longitude(self):
        Om = self.longitude_of_ascending_node.radians
        w = self.argument_of_periapsis.radians
        v = self.true_anomaly.radians
        l = (Om + w + v) % tau
        return Angle(radians=l)

    def __repr__(self):
        return "<Elements {0} sets>".format(self.time.tt.size)


# Also these sites:
# https://web.archive.org/web/*/http://ccar.colorado.edu/asen5070/handouts/cart2kep2002.pdf
# https://web.archive.org/web/*/http://ccar.colorado.edu/asen5070/handouts/kep2cart_2002.doc
# https://en.wikipedia.org/wiki/Orbital_elements
# http://www.bogan.ca/orbits/kepler/orbteqtn.html


def normpi(num):
    return (num + pi) % tau - pi


def eccentricity(h, a, mu):
    condition = h**2 / (a * mu) <= 1
    if h.ndim == 0:
        return sqrt(1 - h**2 / (a * mu)) if condition else float64(0)
    else:
        return sqrt(1 - h**2 / (a * mu), out=zeros_like(h), where=condition)


def eccentricity_vector(pos_vec, vel_vec, mu):
    r = length_of(pos_vec)
    v = length_of(vel_vec)
    return ((v**2 - mu / r) * pos_vec - dots(pos_vec, vel_vec) * vel_vec) / mu


def node_vector(h_vec):
    n_vec = array([-h_vec[1], h_vec[0], zeros_like(h_vec[0])])  # h_vec cross [0, 0, 1]
    n = length_of(n_vec)

    if h_vec.ndim == 1:
        return n_vec / n if n != 0 else n_vec
    else:
        return divide(n_vec, n, out=n_vec, where=n != 0)


class OscuElem(object):
    """
    Osculating orbital elements edited from skyfield library.
    Designed to accomodate multiple array inputs for r0, v0, t0, mu0.
    Note the companion class (rv2cos()) for single r0, v0, t0 arrays.

    May use units aware r0 and v0.

    Input Parameters:
    ----------
        r0_vec  : [km] ndarray, units aware, ecliptic frame, shape (3,) or (3, n)
        v0_vec  : [km/s] ndarray, units aware, ecliptic frame, shape (3,) or (3, n)
        t0      : time object of r0 and v0
        mu_km_s : [km^3/s^2] float, units aware, gravitational parameter (G*M)
    Returns:
    ----------
        Osculating elements attributes:
        01) sma = semi-major axis (aka a)
        02) b = semi-minor axis
        03) ecc_mag = eccentricity
        04) E = eccentric anomaly
        05) h_mag = specific angular momentum
        06) incl = inclination
        07) l = true longitude
        08) L = mean longitude
        09) M = mean anomaly
        10) n = mean motion
        11) Om = longitude of ascending node (aka RAAN)
        12) p = semi-latus rectum
        13) P = period
        14) q = periapsis distance
        15) Q = apoapsis distance
        16) t = time (given t0)
        17) tp = time since periapsis
        18) u = argument of latitude
        19) nu = true anomaly (aka ta, sometimes v)
        20) w = argument of periapsis
        21) lp = longitude of periapsis

        Sources:
        Mostly Bate, Mueller, & White,
          Fundamentals of Astrodynamics (1971), Section 2.4, pgs. 61-64
    """

    def __init__(self, r0_vec, v0_vec, t0, mu_km_s):
        if mu_km_s <= 0:
            raise ValueError(
                "mu_km_s (standard gravitational parameter [km^3/s^2]"
                "must be positive and non-zero."
            )
        # verify units aware variables; unit=astropy, units=pint
        u_aware = False
        if hasattr(r0_vec, "unit"): # astropy units management
            # verify v0_vec compatable units
            u_aware = True
            pos_vec = r0_vec.to(u.km)
            vel_vec = v0_vec.to(u.km / u.s)
            self._pos_vec = pos_vec.value
            self._vel_vec = vel_vec.value
        elif hasattr(r0_vec, "units"): # pint units management
            # verify input vectors compatable units
            u_aware = True
            pos_vec = r0_vec.to(ureg.km)
            vel_vec = v0_vec.to(ureg.km / ureg.s)
            self._pos_vec = pos_vec.magnitude
            self._vel_vec = vel_vec.magnitude
        else:
            print(f"Units are NOT assigned to r0_vec and v0_vec, but SHOULD be.")
            self._pos_vec = r0_vec  # NOT units aware
            self._vel_vec = v0_vec  # NOT units aware

        self.time = t0
        self._mu = mu_km_s
        self._mu_km_d = mu_km_s * (DAY_S * DAY_S)
        # self._h_vec = cross(self._pos_vec, self._vel_vec, 0, 0).T
        self._h_vec = np.cross(self._pos_vec, self._vel_vec)
        self._ecc_vec = ecc_vec(self._pos_vec, self._vel_vec, self._mu)
        self._n_vec = node_vector(self._h_vec)

        self.h_mag = np.linalg.norm(self._h_vec)
        self.incl = incl(self._h_vec)
        # note, ecc_mag_v() when given ecc_vec; else ecc_mag(h_mag, sma, mu)
        self.ecc_mag = ecc_mag_v(self._ecc_vec)
        self.true_anomaly = true_anomaly(
            e_vec=self._ecc_vec,
            pos_vec=self._pos_vec,
            vel_vec=self._vel_vec,
            n_vec=self._n_vec,
        )
        self.eccentric_anomaly = eccentric_anomaly(
            nu=self.true_anomaly, ecc_mag=self.ecc_mag
        )
        p = semi_latus_rectum(h_vec=self._h_vec, mu=self._mu)
        self.semi_latus_rectum = p
        self.semi_major_axis = semi_major_axis(p=p, ecc_mag=self.ecc_mag)
        self.semi_minor_axis = semi_minor_axis(p=p, ecc_mag=self.ecc_mag)
        self.periapsis_distance = periapsis_distance(p=p, ecc_mag=self.ecc_mag)
        self.apoapsis_distance = apoapsis_distance(p=p, ecc_mag=self.ecc_mag)

        self.mean_anomaly = mean_anomaly(self.eccentric_anomaly, ecc_mag=self.ecc_mag)
        self.mean_motion = mean_motion(sma=self.semi_major_axis, mu=self._mu)
        self.longitude_of_ascending_node = longitude_of_ascending_node(
            self.incl, self._h_vec
        )
        self.period = period(sma=self.semi_major_axis, mu=self._mu)
        self.argument_of_periapsis = argument_of_periapsis(
            n_vec=self._n_vec,
            e_vec=self._ecc_vec,
            pos_vec=self._pos_vec,
            vel_vec=self._vel_vec,
        )
        self.time_since_periapsis = time_since_periapsis(
            M=self.mean_anomaly,
            n=self.mean_motion,
            nu=self.true_anomaly,
            p=self.semi_latus_rectum,
            mu=self._mu,
        )

        self.true_longitude = true_longitude(
            Om=self.longitude_of_ascending_node,
            w=self.argument_of_periapsis,
            nu=self.true_anomaly,
        )
        self.longitude_of_periapsis = longitude_of_periapsis(
            Om=self.longitude_of_ascending_node, w=self.argument_of_periapsis
        )
        self.argument_of_latitude = argument_of_latitude(
            w=self.argument_of_periapsis, nu=self.true_anomaly
        )


def ecc_vec(pos_vec, vel_vec, mu):
    r = np.linalg.norm(pos_vec)
    v = np.linalg.norm(vel_vec)
    return ((v**2 - mu / r) * pos_vec - np.dot(pos_vec, vel_vec) * vel_vec) / mu


def ecc_mag_v(ecc_vec):  # use this when u have the ecc_vector
    return np.linalg.norm(ecc_vec)


def ecc_mag(h_mag, sma, mu):  # use when NOT given ecc_vector
    condition = h_mag**2 / (sma * mu) <= 1
    if h_mag.ndim == 0:
        return sqrt(1 - h_mag**2 / (sma * mu)) if condition else float64(0)
    else:
        return sqrt(1 - h_mag**2 / (sma * mu), out=zeros_like(h_mag), where=condition)


def semi_latus_rectum(h_vec, mu):  # aka p
    return np.linalg.norm(h_vec) ** 2 / mu


def incl(h_vec):
    k_vec = array([zeros_like(h_vec[0]), zeros_like(h_vec[0]), ones_like(h_vec[0])])
    return angle_between(h_vec, k_vec)


def mean_anomaly(E, ecc_mag, shift=True):
    if ecc_mag.ndim == 0:
        if ecc_mag < 1:
            return (E - ecc_mag * sin(E)) % tau
        elif ecc_mag > 1:
            M = ecc_mag * sinh(E) - E
            return normpi(M) if shift else M
        else:
            return float64(0)
    else:
        M = zeros_like(ecc_mag)  # defaults to 0 for parabolic

        inds = ecc_mag < 1  # elliptical
        M[inds] = (E[inds] - ecc_mag[inds] * sin(E[inds])) % tau

        inds = ecc_mag > 1  # hyperbolic
        if shift:
            M[inds] = normpi(ecc_mag[inds] * sinh(E[inds]) - E[inds])
        else:
            M[inds] = ecc_mag[inds] * sinh(E[inds]) - E[inds]

        return M


def mean_motion(sma, mu):
    return sqrt(mu / abs(sma) ** 3)


def longitude_of_ascending_node(incl, h_vec):
    if incl.ndim == 0:
        return arctan2(h_vec[0], -h_vec[1]) % tau if incl != 0 else float64(0)
    else:
        return arctan2(h_vec[0], -h_vec[1], out=zeros_like(incl), where=incl != 0) % tau


def true_anomaly(e_vec, pos_vec, vel_vec, n_vec):
    """return variable nu"""
    if pos_vec.ndim == 1:
        if np.linalg.norm(e_vec) > 1e-15:  # not circular
            angle = angle_between(e_vec, pos_vec)
            nu = angle if dots(pos_vec, vel_vec) > 0 else -angle % tau

        elif np.linalg.norm(n_vec) < 1e-15:  # circular and equatorial
            angle = arccos(pos_vec[0] / np.linalg.norm(pos_vec))
            nu = angle if vel_vec[0] < 0 else -angle % tau

        else:  # circular and not equatorial
            angle = angle_between(n_vec, pos_vec)
            nu = angle if pos_vec[2] >= 0 else -angle % tau

        return nu if np.linalg.norm(e_vec) < (1 - 1e-15) else normpi(nu)
    else:
        nu = zeros_like(pos_vec[0])
        circular = np.linalg.norm(e_vec) < 1e-15
        equatorial = np.linalg.norm(n_vec) < 1e-15

        inds = ~circular
        angle = angle_between(e_vec[:, inds], pos_vec[:, inds])
        condition = dots(pos_vec[:, inds], vel_vec[:, inds]) > 0
        nu[inds] = where(condition, angle, -angle % tau)

        inds = circular * equatorial
        angle = arccos(pos_vec[0][inds] / length_of(pos_vec)[inds])
        condition = vel_vec[0][inds] < 0
        nu[inds] = where(condition, angle, -angle % tau)

        inds = circular * ~equatorial
        angle = angle_between(n_vec[:, inds], pos_vec[:, inds])
        condition = pos_vec[2][inds] >= 0
        nu[inds] = where(condition, angle, -angle % tau)

        inds = np.linalg.norm(e_vec) > (1 - 1e-15)
        nu[inds] = normpi(nu[inds])

        return nu


def argument_of_periapsis(n_vec, e_vec, pos_vec, vel_vec):
    if n_vec.ndim == 1:
        if np.linalg.norm(e_vec) < 1e-15:  # circular
            return 0

        elif np.linalg.norm(n_vec) < 1e-15:  # equatorial and not circular
            angle = arctan2(e_vec[1], e_vec[0]) % tau
            return angle if cross(pos_vec, vel_vec, 0, 0).T[2] >= 0 else -angle % tau

        else:  # not circular and not equatorial
            angle = angle_between(n_vec, e_vec)
            return angle if e_vec[2] > 0 else -angle % tau
    else:
        w = zeros_like(pos_vec[0])  # defaults to 0 for circular orbits

        equatorial = np.linalg.norm(n_vec) < 1e-15
        circular = np.linalg.norm(e_vec) < 1e-15

        inds = ~circular * equatorial
        angle = arctan2(e_vec[1][inds], e_vec[0][inds]) % tau
        condition = cross(pos_vec[:, inds], vel_vec[:, inds], 0, 0).T[2] >= 0
        w[inds] = where(condition, angle, -angle % tau)

        inds = ~circular * ~equatorial
        angle = angle_between(n_vec[:, inds], e_vec[:, inds])
        condition = e_vec[2][inds] > 0
        w[inds] = where(condition, angle, -angle % tau)
        return w


def eccentric_anomaly(nu, ecc_mag):
    if ecc_mag.ndim == 0:
        if ecc_mag < 1:
            return 2 * arctan(sqrt((1 - ecc_mag) / (1 + ecc_mag)) * tan(nu / 2))
        elif ecc_mag > 1:
            return normpi(
                2 * arctanh(tan(nu / 2) / sqrt((ecc_mag + 1) / (ecc_mag - 1)))
            )
        else:
            return 0
    else:
        E = zeros_like(ecc_mag)  # defaults to 0 for parabolic

        inds = ecc_mag < 1  # elliptical
        E[inds] = 2 * arctan(
            sqrt((1 - ecc_mag[inds]) / (1 + ecc_mag[inds])) * tan(nu[inds] / 2)
        )

        inds = ecc_mag > 1  # hyperbolic
        E[inds] = normpi(
            2
            * arctanh(
                tan(nu[inds] / 2) / sqrt((ecc_mag[inds] + 1) / (ecc_mag[inds] - 1))
            )
        )
        return E


def semi_major_axis(p, ecc_mag):
    if p.ndim == 0:
        return p / (1 - ecc_mag**2) if ecc_mag != 1 else float64(inf)
    else:
        return divide(p, 1 - ecc_mag**2, out=repeat(inf, p.shape), where=ecc_mag != 1)


def semi_minor_axis(p, ecc_mag):
    if ecc_mag.ndim == 0:
        if ecc_mag < 1:
            return p / sqrt(1 - ecc_mag**2)
        elif ecc_mag > 1:
            return p * sqrt(ecc_mag**2 - 1) / (1 - ecc_mag**2)
        else:
            return float64(0)
    else:
        b = zeros_like(ecc_mag)  # 0 default for parabolic

        inds = ecc_mag < 1  # elliptical
        b[inds] = p[inds] / sqrt(1 - ecc_mag[inds] ** 2)

        inds = ecc_mag > 1  # hyperbolic
        b[inds] = p[inds] * sqrt(ecc_mag[inds] ** 2 - 1) / (1 - ecc_mag[inds] ** 2)

        return b


def period(sma, mu):
    if sma.ndim == 0:
        return tau * sqrt(sma**3 / mu) if sma > 0 else float64(inf)
    else:
        return tau * sqrt(sma**3 / mu, out=repeat(inf, sma.shape), where=sma > 0)


def periapsis_distance(p, ecc_mag):
    if p.ndim == 0:
        return p * (1 - ecc_mag) / (1 - ecc_mag**2) if ecc_mag != 1 else p / 2
    else:
        return divide(
            p * (1 - ecc_mag), (1 - ecc_mag**2), out=p / 2, where=ecc_mag != 1
        )


def apoapsis_distance(p, ecc_mag):
    if p.ndim == 0:
        return p * (1 + ecc_mag) / (1 - ecc_mag**2) if ecc_mag < 1 else float64(inf)
    else:
        return divide(
            p * (1 + ecc_mag),
            1 - ecc_mag**2,
            out=repeat(inf, p.shape),
            where=ecc_mag < (1 - 1e-15),
        )


def time_since_periapsis(M, n, nu, p, mu):
    # Problem: this `too_small` tuning parameter is sensitive to the
    # units used for time, even though this routine should be unit-
    # agnostic.  It was originally 1e-19 but now is 1e-19 * DAY_S.
    too_small = 8.64e-15
    if p.ndim == 0:
        if n >= too_small:  # non-parabolic
            return M / n
        else:  # parabolic
            D = tan(nu / 2)
            return sqrt(2 * (p / 2) ** 3 / mu) * (D + D**3 / 3)
    else:
        parabolic = n < too_small
        t = divide(M, n, out=zeros_like(p), where=~parabolic)

        D = tan(nu[parabolic] / 2)
        t[parabolic] = sqrt(2 * (p[parabolic] / 2) ** 3 / mu) * (D + D**3 / 3)

        return t


def argument_of_latitude(w, nu):
    u = (w + nu) % tau  # modulo 2pi
    return u


def true_longitude(Om, w, nu):
    l = (Om + w + nu) % tau
    return l


def longitude_of_periapsis(Om, w):
    lp = (Om + w) % tau
    return lp


def test_OscuElem():
    """
    Edited version of skyfield's OsculatingElements().
    Earth->Venus transfer orbit, 1988-04-08.
    r1_vec, v1_vec taken from skyfield.
    """
    # ICRF equatorial frame
    r0_vec = np.array([-1.4256732793e08, -4.3413585068e07, -1.8821120387e07])*ureg.km
    v0_vec = np.array([10.1305671502, -22.2030303357, -11.8934643152])*(ureg.km/ureg.s)
    position = r0_vec
    velocity = v0_vec
    time = datetime(1988, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
    mu_km_s = GM_SUN

    elem1 = OscuElem(r0_vec=position, v0_vec=velocity, t0=time, mu_km_s=mu_km_s)
    print(f"\nTransfer orbital elements:")
    # print(f"   Periapsis time: {elem1.time_since_periapsis.utc_strftime()}")
    # print(f"   Periapsis time: {elem1.time_since_periapsis.strftime("%Y-%m-%d %H:%M:%S UTC")}")
    for attr in dir(elem1):
        if not attr.startswith("__"):
            print(f"   {attr}, {getattr(elem1, attr)}")

    return


def main():
    # just a placeholder to help with editor navigation:--)
    pass
    return


# use the following to test/examine functions
if __name__ == "__main__":

    test_OscuElem()  # edited version of skyfield's OsculatingElements()
    main()
