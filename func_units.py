# -*- coding: utf-8 -*-
"""
Function collection related to units management and conversions.
    2025-5 have not figured out what to do with this file...
2025, JBelue edits Skyfield repo; I do not understand many Skyfield techniques.
    Generally the functions are units-aware; either pint or astropy.
Distance, velocity, and angle support from Skyfield.

"""
import astropy.units as ap_u
import numpy as np
import pint
from numpy import copysign, isnan

from constants import AU_KM, AU_M, DAY_S, TAU, C
from func_gen import _to_array, length_of, reify

ureg = pint.UnitRegistry()  # pint units management

_dfmt = "{0}{1:02}deg {2:02}' {3:02}.{4:0{5}}\""
_dsgn = "{0:+>1}{1:02}deg {2:02}' {3:02}.{4:0{5}}\""
_hfmt = "{0}{1:02}h {2:02}m {3:02}.{4:0{5}}s"


class UnpackingError(Exception):
    """You cannot iterate directly over a Skyfield measurement object."""


class Unit(object):
    """A measurement that can be expressed in several choices of unit."""

    def __getitem__(self, *args):
        """Tell users to ask for a specific unit before indexing or slicing."""
        cls = self.__class__
        name = cls.__name__
        s = "to use this {0}, ask for its value in a particular unit:\n\n{1}"
        attrs = sorted(
            k
            for k, v in cls.__dict__.items()
            if k[0].islower() and isinstance(v, (getset, reify))
        )
        examples = ["    {0}.{1}".format(name.lower(), attr) for attr in attrs]
        raise UnpackingError(s.format(name, "\n".join(examples)))

    __iter__ = __getitem__  # give advice about both foo[i] and "x,y,z = foo"


class getset(object):
    """Unit name that serves as both a class constructor and instance attribute.

    This supports two use cases:

    * When called as a class method like ``Distance.km(5.0)``, we build
      and return an instance of ``Distance`` whose ``km`` has been set
      to 5.0 and whose base unit ``m``, using the appropriate conversion
      factor, has been set to 5000.0.

    * When invoked like ``d.km`` on a particular ``Distance`` that
      doesn't yet have a ``km`` attribute (which otherwise Python itself
      would have returned), we apply the conversion factor to ``d.m``
      and return the result.

    """

    def __init__(self, name, docstring, conversion_factor=None, core_unit=None):
        self.name = name
        self.__doc__ = docstring
        self.conversion_factor = conversion_factor
        self.core_unit = core_unit

    def __get__(self, instance, objtype=None):
        if instance is None:  # the class itself has been asked for this name

            def constructor(value):
                value = _to_array(value)
                obj = objtype.__new__(objtype)
                setattr(obj, self.name, value)
                conversion_factor = self.conversion_factor
                if conversion_factor is not None:
                    setattr(obj, self.core_unit, value / conversion_factor)
                return obj

            constructor.__doc__ = self.__doc__
            return constructor
        value = getattr(instance, self.core_unit) * self.conversion_factor
        instance.__dict__[self.name] = value
        return value


class Distance(Unit):
    """A distance, stored internally as au and available in other units.

    You can initialize a ``Distance`` by providing a single float or a
    float array as either an ``au=``, ``km=``, or ``m=`` parameter.

    You can access the magnitude of the distance with its three
    attributes ``.au``, ``.km``, and ``.m``.  By default a distance
    prints itself in astronomical units (au), but you can take control
    of the formatting and choice of units yourself using standard Python
    numeric formatting:

    >>> d = Distance(au=1)
    >>> print(d)
    1.0 au
    >>> print('{:.2f} km'.format(d.km))
    149597870.70 km

    """

    _warned = False

    def __init__(self, au=None, km=None, m=None):
        if au is not None:
            self.au = _to_array(au)
            """Astronomical units."""
        elif km is not None:
            self.km = km = _to_array(km)
            self.au = km / AU_KM
        elif m is not None:
            self.m = m = _to_array(m)
            self.au = m / AU_M
        else:
            raise ValueError("To construct a Distance provide au, km, or m.")

    au = getset(
        "au", "Astronomical units" " (the Earth-Sun distance of 149,597,870,700 m)."
    )
    km = getset("km", "Kilometers (1,000 meters).", AU_KM, "au")
    m = getset("m", "Meters.", AU_M, "au")

    def __str__(self):
        n = self.au
        return ("{0} au" if getattr(n, "shape", 0) else "{0:.6} au").format(n)

    def __repr__(self):
        return "<{0} {1}>".format(type(self).__name__, self)

    def length(self):
        """Compute the length when this is an |xyz| vector.

        The Euclidean vector length of this vector is returned as a new
        :class:`~skyfield.units.Distance` object.

        >>> from skyfield.api import Distance
        >>> d = Distance(au=[1, 1, 0])
        >>> d.length()
        <Distance 1.41421 au>

        """
        return Distance(au=length_of(self.au))

    def light_seconds(self):
        """Return the length of this vector in light seconds."""
        return self.m / C

    def to(self, unit):
        """Convert this distance to the given AstroPy unit."""
        from astropy.units import au

        return (self.au * au).to(unit)


class Velocity(Unit):
    """A velocity, stored internally as au/day and available in other units.

    You can initialize a ``Velocity`` by providing a float or float
    array to its ``au_per_d=`` parameter.

    """

    _warned = False

    # TODO: consider reworking this class to return a Rate object.

    def __init__(self, au_per_d=None, km_per_s=None):
        if km_per_s is not None:
            self.km_per_s = km_per_s = _to_array(km_per_s)
            self.au_per_d = km_per_s * DAY_S / AU_KM
        elif au_per_d is not None:
            self.au_per_d = _to_array(au_per_d)
        else:
            raise ValueError("to construct a Velocity provide" " au_per_d or km_per_s")

    au_per_d = getset("au_per_d", "Astronomical units per day.")
    km_per_s = getset("km_per_s", "Kilometers per second.", AU_KM / DAY_S, "au_per_d")
    m_per_s = getset("m_per_s", "Meters per second.", AU_M / DAY_S, "au_per_d")

    def __str__(self):
        n = self.au_per_d
        fmt = "{0} au/day" if getattr(n, "shape", 0) else "{0:.6} au/day"
        return fmt.format(n)

    def __repr__(self):
        return "<{0} {1}>".format(type(self).__name__, self)

    def to(self, unit):
        """Convert this velocity to the given AstroPy unit."""
        from astropy.units import au, d

        return (self.au_per_d * au / d).to(unit)


class AngleRate(object):
    """The rate at which an angle is changing."""

    # TODO: design and implement public constructor.

    @classmethod
    def _from_radians_per_day(cls, radians_per_day):
        ar = cls()
        ar._radians_per_day = radians_per_day
        return ar

    @reify
    def radians(self):
        """:class:`Rate` of change in radians."""
        return Rate._from_per_day(self._radians_per_day)

    @reify
    def degrees(self):
        """:class:`Rate` of change in degrees."""
        return Rate._from_per_day(self._radians_per_day / TAU * 360.0)

    @reify
    def arcminutes(self):
        """:class:`Rate` of change in arcminutes."""
        return Rate._from_per_day(self._radians_per_day / TAU * 21600.0)

    @reify
    def arcseconds(self):
        """:class:`Rate` of change in arcseconds."""
        return Rate._from_per_day(self._radians_per_day / TAU * 1296000.0)

    @reify
    def mas(self):
        """:class:`Rate` of change in milliarcseconds."""
        return Rate._from_per_day(self._radians_per_day / TAU * 1.296e9)

    # TODO: str; repr; conversion to AstroPy units


class Rate(object):
    """Measurement whose denominator is time."""

    # TODO: design and implement public constructor.

    @classmethod
    def _from_per_day(cls, per_day):
        r = cls()
        r._per_day = per_day
        return r

    @reify
    def per_day(self):
        """Units per day of Terrestrial Time."""
        return self._per_day

    @reify
    def per_hour(self):
        """Units per hour of Terrestrial Time."""
        return self._per_day / 24.0

    @reify
    def per_minute(self):
        """Units per minute of Terrestrial Time."""
        return self._per_day / 1440.0

    @reify
    def per_second(self):
        """Units per second of Terrestrial Time."""
        return self._per_day / 86400.0


# Angle units.

_instantiation_instructions = """to instantiate an Angle, try one of:

Angle(angle=another_angle)
Angle(radians=value)
Angle(degrees=value)
Angle(hours=value)

where `value` can be either a Python float, a list of Python floats,
or a NumPy array of floats"""


class Angle(Unit):

    def __init__(
        self,
        angle=None,
        radians=None,
        degrees=None,
        hours=None,
        preference=None,
        signed=False,
    ):

        if angle is not None:
            if not isinstance(angle, Angle):
                raise ValueError(_instantiation_instructions)
            self.radians = angle.radians
        elif radians is not None:
            self.radians = _to_array(radians)
        elif degrees is not None:
            self._degrees = degrees = _to_array(_unsexagesimalize(degrees))
            self.radians = degrees / 360.0 * TAU
        elif hours is not None:
            self._hours = hours = _to_array(_unsexagesimalize(hours))
            self.radians = hours / 24.0 * TAU

        self.preference = (
            preference
            if preference is not None
            else "hours" if hours is not None else "degrees"
        )
        self.signed = signed

    @classmethod
    def from_degrees(cls, degrees, signed=False):
        degrees = _to_array(_unsexagesimalize(degrees))
        self = cls.__new__(cls)
        self.degrees = degrees
        self.radians = degrees / 360.0 * TAU
        self.preference = "degrees"
        self.signed = signed
        return self

    radians = getset("radians", "Radians (𝜏 = 2𝜋 in a circle).")

    @reify
    def _hours(self):
        return self.radians * 24.0 / TAU

    @reify
    def _degrees(self):
        return self.radians * 360.0 / TAU

    @reify
    def hours(self):
        r"""Hours (24\ |h| in a circle)."""
        if self.preference != "hours":
            raise WrongUnitError("hours")
        return self._hours

    @reify
    def degrees(self):
        """Degrees (360° in a circle)."""
        if self.preference != "degrees":
            raise WrongUnitError("degrees")
        return self._degrees

    def arcminutes(self):
        """Return the angle in arcminutes."""
        return self._degrees * 60.0

    def arcseconds(self):
        """Return the angle in arcseconds."""
        return self._degrees * 3600.0

    def mas(self):
        """Return the angle in milliarcseconds."""
        return self._degrees * 3600000.0

    def __str__(self):
        size = self.radians.size
        if size == 0:
            return "Angle []"
        if self.preference == "degrees":
            v = self._degrees
            fmt = _dsgn.format if self.signed else _dfmt.format
            places = 1
        else:
            v = self._hours
            fmt = _hfmt.format
            places = 2
        if size >= 2:
            return "{0} values from {1} to {2}".format(
                len(v), _sfmt(fmt, v[0], places), _sfmt(fmt, v[-1], places)
            )
        return _sfmt(fmt, v, places)

    def __repr__(self):
        if self.radians.size == 0:
            return "<{0} []>".format(type(self).__name__)
        else:
            return "<{0} {1}>".format(type(self).__name__, self)

    def hms(self, warn=True):
        """Convert to a tuple (hours, minutes, seconds).

        All three quantities will have the same sign as the angle itself.

        """
        if warn and self.preference != "hours":
            raise WrongUnitError("hms")
        sign, units, minutes, seconds = _sexagesimalize_to_float(self._hours)
        return sign * units, sign * minutes, sign * seconds

    def signed_hms(self, warn=True):
        """Convert to a tuple (sign, hours, minutes, seconds).

        The ``sign`` will be either +1 or -1, and the other quantities
        will all be positive.

        """
        if warn and self.preference != "hours":
            raise WrongUnitError("signed_hms")
        return _sexagesimalize_to_float(self._hours)

    def hstr(self, places=2, warn=True, format=_hfmt):
        """Return a string like ``12h 07m 30.00s``; see `Formatting angles`.

        .. versionadded:: 1.39

           Added the ``format=`` parameter.

        """
        if warn and self.preference != "hours":
            raise WrongUnitError("hstr")
        hours = self._hours
        shape = getattr(hours, "shape", ())
        fmt = format.format  # `format()` method of `format` string
        if shape:
            return [_sfmt(fmt, h, places) for h in hours]
        return _sfmt(fmt, hours, places)

    def dms(self, warn=True):
        """Convert to a tuple (degrees, minutes, seconds).

        All three quantities will have the same sign as the angle itself.

        """
        if warn and self.preference != "degrees":
            raise WrongUnitError("dms")
        sign, units, minutes, seconds = _sexagesimalize_to_float(self._degrees)
        return sign * units, sign * minutes, sign * seconds

    def signed_dms(self, warn=True):
        """Convert to a tuple (sign, degrees, minutes, seconds).

        The ``sign`` will be either +1 or -1, and the other quantities
        will all be positive.

        """
        if warn and self.preference != "degrees":
            raise WrongUnitError("signed_dms")
        return _sexagesimalize_to_float(self._degrees)

    def dstr(self, places=1, warn=True, format=None):
        """Return a string like ``181deg 52' 30.0"``; see `Formatting angles`.

        .. versionadded:: 1.39

           Added the ``format=`` parameter.

        """
        if warn and self.preference != "degrees":
            raise WrongUnitError("dstr")
        degrees = self._degrees
        signed = self.signed
        if format is None:
            format = _dsgn if signed else _dfmt
        fmt = format.format  # `format()` method of `format` string
        shape = getattr(degrees, "shape", ())
        if shape:
            return [_sfmt(fmt, d, places) for d in degrees]
        return _sfmt(fmt, degrees, places)

    def to(self, unit):
        """Convert this angle to the given AstroPy unit."""
        from astropy.units import rad

        return (self.radians * rad).to(unit)

        # Or should this do:
        from astropy.coordinates import Angle
        from astropy.units import rad

        return Angle(self.radians, rad).to(unit)


class WrongUnitError(ValueError):

    def __init__(self, name):
        unit = "hours" if (name.startswith("h") or "_h" in name) else "degrees"
        usual = "hours" if (unit == "degrees") else "degrees"
        message = (
            f"this angle is usually expressed in {0}, not {1};"
            f" if you want to use {1} anyway,".format(usual, unit)
        )
        if name == unit:
            message += f" then please use the attribute _{0}".format(unit)
        else:
            message += f" then call {0}() with warn=False".format(name)
        self.args = (message,)


def _sexagesimalize_to_float(value):
    """Decompose `value` into units, minutes, and seconds.

    Note that this routine is not appropriate for displaying a value,
    because rounding to the smallest digit of display is necessary
    before showing a value to the user.  Use `_sexagesimalize_to_int()`
    for data being displayed to the user.

    This routine simply decomposes the floating point `value` into a
    sign (+1.0 or -1.0), units, minutes, and seconds, returning the
    result in a four-element tuple.

    >>> _sexagesimalize_to_float(12.05125)
    (1.0, 12.0, 3.0, 4.5)
    >>> _sexagesimalize_to_float(-12.05125)
    (-1.0, 12.0, 3.0, 4.5)

    """
    sign = np.sign(value)
    n = abs(value)
    minutes, seconds = divmod(n * 3600.0, 60.0)
    units, minutes = divmod(minutes, 60.0)
    return sign, units, minutes, seconds


def _sexagesimalize_to_int(value, places=0):
    """Decompose `value` into units, minutes, seconds, and second fractions.

    This routine prepares a value for sexagesimal display, with its
    seconds fraction expressed as an integer with `places` digits.  The
    result is a tuple of five integers:

    ``(sign [either +1 or -1], units, minutes, seconds, second_fractions)``

    The integers are properly rounded per astronomical convention so
    that, for example, given ``places=3`` the result tuple ``(1, 11, 22,
    33, 444)`` means that the input was closer to 11u 22' 33.444" than
    to either 33.443" or 33.445" in its value.

    """
    power = 10**places
    n = int((power * 3600 * value + 0.5) // 1.0)
    sign = np.sign(n)
    n, fraction = divmod(abs(n), power)
    n, seconds = divmod(n, 60)
    n, minutes = divmod(n, 60)
    return sign, n, minutes, seconds, fraction


def _sfmt(fmt, value, places):
    """Decompose floating point `value` into sexagesimal, and format."""
    if isnan(value):
        return "nan"
    sgn, h, m, s, fraction = _sexagesimalize_to_int(value, places)
    sign = "-" if sgn < 0.0 else ""
    return fmt(sign, h, m, s, fraction, places)


def wms(whole, minutes=0.0, seconds=0.0):
    """Return a quantity expressed with 1/60 minutes and 1/3600 seconds."""
    return whole + copysign(minutes, whole) / 60.0 + copysign(seconds, whole) / 3600.0


def _unsexagesimalize(value):
    """Return `value` after interpreting a (units, minutes, seconds) tuple.

    When `value` is not a tuple, it is simply returned.

    >>> _unsexagesimalize(3.25)
    3.25

    An input tuple is interpreted as units, minutes, and seconds.  Note
    that only the sign of `units` is significant!  So all of the
    following tuples convert into exactly the same value:

    >>> '%f' % _unsexagesimalize((-1, 2, 3))
    '-1.034167'
    >>> '%f' % _unsexagesimalize((-1, -2, 3))
    '-1.034167'
    >>> '%f' % _unsexagesimalize((-1, -2, -3))
    '-1.034167'

    """
    if isinstance(value, tuple):
        components = iter(value)
        value = next(components)
        factor = 1.0
        for component in components:
            factor *= 60.0
            value += copysign(component, value) / factor
    return value


def _interpret_angle(name, angle_object, angle_float, unit="degrees"):
    """Return an angle in radians from one of two arguments.

    It is common for Skyfield routines to accept both an argument like
    `alt` that takes an Angle object as well as an `alt_degrees` that
    can be given a bare float or a sexagesimal tuple.  A pair of such
    arguments can be passed to this routine for interpretation.

    """
    if angle_object is not None:
        if isinstance(angle_object, Angle):
            return angle_object.radians
    elif angle_float is not None:
        return _unsexagesimalize(angle_float) / 360.0 * TAU
    raise ValueError(
        f"you must either provide the {0}= parameter with"
        f" an Angle argument or supply the {0}_{1}= parameter"
        f" with a numeric argument".format(name, unit)
    )


def units_aware(value):
    """check if object has either pint or astropy units."""
    # examine unit attribute
    if isinstance(value, ap_u.Quantity):  # units for pint; unit for astropy
        ua_ret = "astropy"  # ua_out = units aware return
    elif isinstance(value, ureg.Quantity):  # units for pint; unit for astropy
        ua_ret = "pint"
    else:
        ua_ret = None  # no unit detected
    return ua_ret


def contains_angle(unit1):
    """
    Check pint units then for angle unit; degrees or radians.
    TODO: make compatable with astropy units
    """
    ck_units = units_aware(value=unit1)  # test for units in unit1
    out1 = False  # not a pint angle dimension
    if ck_units == "pint":
        # ~ = pint print short units
        if unit1.units == ureg.radian:
            out1 = True
        elif unit1.units == ureg.degree:
            out1 = True
        else:  # unit1 does not units
            out1 = False
            print(f"pint angle units not found in: {unit1}")
    else:
        print("Non-pint units or no units; further test not coded yet.")
    return out1
