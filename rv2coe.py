"""
produce coe (classical orbital elements) from r0, v0, t0, and gm.
    rv2coe class covers a broader range of coe then Curtis [3] or Vallado [4].
    This rv2coe class is an edited version of JBelue elemLib.py, class OscuElem().
        OscuElem() accomodates multiple array vectors; r0,v0; r1,v1; r2,v2, while
        rv2coe() class expects a single vector set r0, v0, t0, and gm.
    Internal calculation units are kilometer, seconds, radians.
Notes:
    1) some subtle non-obvious efficient calculations; i.e. see length_of().
        length_of() tends to be faster than np.linalg.norm() for small arrays.
    2) TODO: may speedup execution by limiting redundant calculations, perhapse
        using @reify, as in the skyfield repo.

References:
----------
    See references.py for references list.  Note additional links below:
    https://en.wikipedia.org/wiki/Orbital_elements
    http://www.bogan.ca/orbits/kepler/orbteqtn.html
"""

from datetime import datetime, timedelta, timezone

import numpy as np
from astropy import units as u  # astropy units management
from numpy import (
    arccos,
    arctan,
    arctan2,
    arctanh,
    array,
    cross,
    float64,
    inf,
    ones_like,
    sin,
    sinh,
    sqrt,
    tan,
    zeros_like,
)
from pint import UnitRegistry  # manage variable units

from constants import DAY_S, GM_SUN, PI, RAD2DEG, TAU
from func_gen import angle_between, length_of

ureg = UnitRegistry()  # pint units management
Q_ = ureg.Quantity


class RV2coe(object):
    """
    Edited skyfield's osculating orbital elements class library, OsculatingElements().
    Accomodates multiple array inputs for r0, v0, t0, mu0.
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
        13) P = period (capital P)
        14) q = periapsis distance
        15) Q = apoapsis distance
        16) t = time (given t0)
        17) tp = time since periapsis
        18) u = argument of latitude
        19) nu = true anomaly (aka ta, or sometimes v)
        20) w = argument of periapsis
        21) lp = longitude of periapsis
    """

    def __init__(self, r0_vec, v0_vec, t0, mu_km_s):
        if mu_km_s <= 0:
            raise ValueError(
                "mu_km_s (standard gravitational parameter [km^3/s^2]"
                "must be positive and non-zero."
            )
        # check for units aware variables; unit=astropy, units=pint
        if hasattr(r0_vec, "unit"):  # astropy units management
            # set input variable units to compatable units
            pos_vec = r0_vec.to(u.km)
            vel_vec = v0_vec.to(u.km / u.s)
            self._pos_vec = pos_vec.value
            self._vel_vec = vel_vec.value
        elif hasattr(r0_vec, "units"):  # pint units management
            # set input variable units to compatable units
            pos_vec = r0_vec.to(ureg.km)
            vel_vec = v0_vec.to(ureg.km / ureg.s)
            self._pos_vec = pos_vec.magnitude
            self._vel_vec = vel_vec.magnitude
        else:  # no units assigned to r0
            print("Units are NOT assigned to r0_vec and v0_vec, but SHOULD be.")
            self._pos_vec = r0_vec  # NOT units aware
            self._vel_vec = v0_vec  # NOT units aware

        self.time = t0
        self._mu = mu_km_s
        # self._h_vec = cross(self._pos_vec, self._vel_vec, 0, 0).T
        self._h_vec = np.cross(self._pos_vec, self._vel_vec)
        self._ecc_vec = ecc_vec(self._pos_vec, self._vel_vec, self._mu)
        self._n_vec = node_vector(self._h_vec)

        self.h_mag = length_of(self._h_vec)
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


def normpi(num):
    """normalize to values <= 2pi"""
    return (num + PI) % TAU - PI


def node_vector(h_vec):
    """one line description"""
    n_vec = array([-h_vec[1], h_vec[0], zeros_like(h_vec[0])])  # h_vec cross [0, 0, 1]
    n = length_of(n_vec)
    return n_vec / n if n != 0 else n_vec


def ecc_vec(pos_vec, vel_vec, mu):
    """one line description"""
    r = length_of(pos_vec)
    v = length_of(vel_vec)
    return ((v**2 - mu / r) * pos_vec - np.dot(pos_vec, vel_vec) * vel_vec) / mu


def ecc_mag_v(ecc_vec):  # use this when u have the ecc_vector
    """one line description"""
    return length_of(ecc_vec)


def ecc_mag(h_mag, sma, mu):  # use when NOT given ecc_vector
    """one line description"""
    condition = h_mag**2 / (sma * mu) <= 1
    return sqrt(1 - h_mag**2 / (sma * mu)) if condition else float64(0)


def semi_latus_rectum(h_vec, mu):
    """aka p"""
    return length_of(h_vec) ** 2 / mu


def incl(h_vec):
    """one line description"""
    k_vec = array([zeros_like(h_vec[0]), zeros_like(h_vec[0]), ones_like(h_vec[0])])
    return angle_between(h_vec, k_vec)


def mean_anomaly(E, ecc_mag, shift=True):
    """one line description"""
    if ecc_mag < 1:
        return (E - ecc_mag * sin(E)) % TAU
    elif ecc_mag > 1:
        M = ecc_mag * sinh(E) - E
        return normpi(M) if shift else M
    else:
        return float64(0)


def mean_motion(sma, mu):
    """one line description"""
    return sqrt(mu / abs(sma) ** 3)


def longitude_of_ascending_node(incl, h_vec):
    """one line description"""
    return arctan2(h_vec[0], -h_vec[1]) % TAU if incl != 0 else float64(0)


def true_anomaly(e_vec, pos_vec, vel_vec, n_vec):
    """return variable nu"""
    if pos_vec.ndim == 1:
        if length_of(e_vec) > 1e-15:  # not circular
            angle = angle_between(e_vec, pos_vec)
            nu = angle if np.dot(pos_vec, vel_vec) > 0 else -angle % TAU

        elif length_of(n_vec) < 1e-15:  # circular and equatorial
            angle = arccos(pos_vec[0] / length_of(pos_vec))
            nu = angle if vel_vec[0] < 0 else -angle % TAU

        else:  # circular and not equatorial
            angle = angle_between(n_vec, pos_vec)
            nu = angle if pos_vec[2] >= 0 else -angle % TAU

        return nu if length_of(e_vec) < (1 - 1e-15) else normpi(nu)


def argument_of_periapsis(n_vec, e_vec, pos_vec, vel_vec):
    """one line description"""
    # length_of() tends to be faster than np.linalg.norm() for small arrays
    if length_of(e_vec) < 1e-15:  # circular
        return 0

    elif length_of(n_vec) < 1e-15:  # equatorial and not circular
        angle = arctan2(e_vec[1], e_vec[0]) % TAU
        return angle if cross(pos_vec, vel_vec, 0, 0).T[2] >= 0 else -angle % TAU

    else:  # not circular and not equatorial
        angle = angle_between(n_vec, e_vec)
        return angle if e_vec[2] > 0 else -angle % TAU


def eccentric_anomaly(nu, ecc_mag):
    """one line description"""
    if ecc_mag < 1:
        return 2 * arctan(sqrt((1 - ecc_mag) / (1 + ecc_mag)) * tan(nu / 2))
    elif ecc_mag > 1:
        return normpi(2 * arctanh(tan(nu / 2) / sqrt((ecc_mag + 1) / (ecc_mag - 1))))
    else:
        return 0


def semi_major_axis(p, ecc_mag):
    """one line description"""
    return p / (1 - ecc_mag**2) if ecc_mag != 1 else float64(inf)


def semi_minor_axis(p, ecc_mag):
    """one line description"""
    if ecc_mag < 1:
        return p / sqrt(1 - ecc_mag**2)
    elif ecc_mag > 1:
        return p * sqrt(ecc_mag**2 - 1) / (1 - ecc_mag**2)
    else:
        return float64(0)


def period(sma, mu):
    """one line description"""
    return TAU * sqrt(sma**3 / mu) if sma > 0 else float64(inf)


def periapsis_distance(p, ecc_mag):
    """one line description"""
    return p * (1 - ecc_mag) / (1 - ecc_mag**2) if ecc_mag != 1 else p / 2


def apoapsis_distance(p, ecc_mag):
    """one line description"""
    return p * (1 + ecc_mag) / (1 - ecc_mag**2) if ecc_mag < 1 else float64(inf)


def time_since_periapsis(M, n, nu, p, mu):
    """one line description"""
    # Problem: this `too_small` tuning parameter is sensitive to the
    # units used for time, even though this routine should be unit-
    # agnostic.  It was originally 1e-19 but now is 1e-19 * DAY_S.
    too_small = 8.64e-15
    if n >= too_small:  # non-parabolic
        return M / n
    else:  # parabolic
        D = tan(nu / 2)
        return sqrt(2 * (p / 2) ** 3 / mu) * (D + D**3 / 3)


def argument_of_latitude(w, nu):
    """one line description"""
    u = (w + nu) % TAU  # modulo 2pi
    return u


def true_longitude(Om, w, nu):
    """one line description"""
    l = (Om + w + nu) % TAU  # modulo 2pi
    return l


def longitude_of_periapsis(Om, w):
    """one line description"""
    lp = (Om + w) % TAU  # modulo 2pi
    return lp


def test_RV2coe():
    """
    Earth->Venus transfer orbit, 1988-04-08.
    r1_vec, v1_vec taken from skyfield.
    """
    # ICRF equatorial frame, from skyfield; pint units management
    r0_vec = np.array([-1.4256732793e08, -4.3413585068e07, -1.8821120387e07]) * ureg.km
    v0_vec = np.array([10.1305671502, -22.2030303357, -11.8934643152]) * (
        ureg.km / ureg.s
    )
    position = r0_vec
    velocity = v0_vec
    mu_km_s = GM_SUN
    time = datetime(1988, 4, 8, 0, 0, 0, tzinfo=timezone.utc)

    # OscuElem() expects units aware r0, v0, mu, & python time
    elem1 = RV2coe(r0_vec=position, v0_vec=velocity, t0=time, mu_km_s=mu_km_s)
    print("\nTransfer orbital elements:")

    # next, print orbital elements
    for attr in dir(elem1):
        if not attr.startswith("__"):
            print(f"   {attr}, {getattr(elem1, attr)}")

    print(f"  epoch time: {elem1.time}")
    print(
        f"  last periapsis time: {elem1.time - timedelta(seconds=elem1.time_since_periapsis)}"
    )
    print(f"  time since periapsis: {elem1.time_since_periapsis} [seconds]")
    print(f"  time since periapsis: {elem1.time_since_periapsis/DAY_S} [days]")
    print(f"  orbit inclination: {elem1.incl*RAD2DEG} [deg]")
    print(f"  orbit eccentricity: {elem1.ecc_mag}")

    print("\nExplore pint units and conversions:")
    print("   Transfer distance unit from r0 to sma (semi-major axis).")
    sma = elem1.semi_major_axis = elem1.semi_major_axis * r0_vec.units
    print(f"   r0_vec units: {r0_vec:~}")  # ~ = short unit form (pint)
    print(f"   calculated sma with units: {sma:~}")  # ~ = short unit form (pint)

    # angle's, but note they are dimensionless :--)
    incl = Q_(elem1.incl, "rad")
    print(f"   incl: {incl:~}")  # ~ = short unit form (pint)
    print(f"   incl: {incl.to("deg"):~}")  # ~ = short unit form (pint)
    return


def main():
    """just a placeholder to help with editor navigation:--)"""
    return


# use the following to test/examine functions
if __name__ == "__main__":

    test_RV2coe()  # edited version of skyfield's OsculatingElements()
    main()
