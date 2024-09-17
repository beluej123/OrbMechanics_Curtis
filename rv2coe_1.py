"""
These utility routines from David Vallado "sgp4ext.cpp" converted to python.
2024-03-05, my copy from hyperlink below;
Convert position & velocity vectors to COE (classical orbital elements ; i.e. Kepler)
Note, I used the Black formatter; not 100% happy with it, but it helps with readability!
2024-03-01, https://github.com/brandon-rhodes/python-sgp4/blob/e6178785d15996885c6ed7fb33585ca6e5d8dd67/sgp4/ext.py#L224
also checkout, https://github.com/aerospaceresearch/orbitdeterminator/blob/master/orbitdeterminator/kep_determination/orbital_elements.py
References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
    [4] Vallado, David A., (2022, 5th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
"""

from math import acos, asinh, atan2, copysign, cos, fabs, fmod, pi, sin, sinh, sqrt, tan

undefined = None


def mag(x):
    """
    /* -----------------------------------------------------------------------------
    *                           function mag
    *  this procedure finds the magnitude of a vector.  the tolerance is set to
    *    0.000001, thus the 1.0e-12 for the squared test of underflows.
    *
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *  inputs          description                    range / units
    *    vec         - vector
    *  outputs       :
    *    vec         - answer stored in fourth component
    *  locals        :
    *    none.
    *  coupling      :
    *    none.
    * --------------------------------------------------------------------------- */
    """
    return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def cross(vec1, vec2, outvec):
    """
    /* -----------------------------------------------------------------------------
    *                           procedure cross
    *  this procedure crosses two vectors.
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *  inputs          description                    range / units
    *    vec1        - vector number 1
    *    vec2        - vector number 2
    *  outputs       :
    *    outvec      - vector result of a x b
    *  locals        :
    *    none.
    *  coupling      :
    *    mag           magnitude of a vector
    ---------------------------------------------------------------------------- */
    """
    outvec[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    outvec[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    outvec[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]


def dot(x, y):
    """
    /* -----------------------------------------------------------------------------
    *                           function dot
    *  this function finds the dot product of two vectors.
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *  inputs          description                    range / units
    *    vec1        - vector number 1
    *    vec2        - vector number 2
    *  outputs       :
    *    dot         - result
    *  locals        :
    *    none.
    *  coupling      :
    *    none.
    * --------------------------------------------------------------------------- */
    """
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def angle(vec1, vec2):
    """
    /* -----------------------------------------------------------------------------
    *                           procedure angle
    *  this procedure calculates the angle between two vectors.  the output is
    *    set to 999999.1 to indicate an undefined value.  be sure to check for
    *    this at the output phase.
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *  inputs          description                    range / units
    *    vec1        - vector number 1
    *    vec2        - vector number 2
    *  outputs       :
    *    theta       - angle between the two vectors  -pi to pi
    *  locals        :
    *    temp        - temporary real variable
    *  coupling      :
    *    dot           dot product of two vectors
    * --------------------------------------------------------------------------- */
    """

    small = 0.00000001
    undefined = 999999.1

    magv1 = mag(vec1)
    magv2 = mag(vec2)

    if magv1 * magv2 > small * small:

        temp = dot(vec1, vec2) / (magv1 * magv2)
        if fabs(temp) > 1.0:
            temp = copysign(1.0, temp)
        return acos(temp)

    else:
        return undefined


def newtonnu(ecc, nu):
    """
    /* -----------------------------------------------------------------------------
    *                           function newtonnu
    *  this function solves keplers equation when the true anomaly is known.
    *    the mean and eccentric, parabolic, or hyperbolic anomaly is also found.
    *    the parabolic limit at 168° is arbitrary. the hyperbolic anomaly is also
    *    limited. the hyperbolic sine is used because it's not double valued.
    *  author        : david vallado                  719-573-2600   27 may 2002
    *  revisions
    *    vallado     - fix small                                     24 sep 2002
    *  inputs          description                    range / units
    *    ecc         - eccentricity                   0.0  to
    *    nu          - true anomaly                   -2pi to 2pi rad
    *  outputs       :
    *    e0          - eccentric anomaly              0.0  to 2pi rad       153.02 °
    *    m           - mean anomaly                   0.0  to 2pi rad       151.7425 °
    *  locals        :
    *    e1          - eccentric anomaly, next value  rad
    *    sine        - sine of e
    *    cose        - cosine of e
    *    ktr         - index
    *  coupling      :
    *    asinh       - arc hyperbolic sine
    *  references    :
    *    vallado       2007, 85, alg 5
    * --------------------------------------------------------------------------- */
    """
    #  ---------------------  implementation   ---------------------
    e0 = 999999.9
    m = 999999.9
    small = 0.00000001
    #  --------------------------- circular ------------------------
    if fabs(ecc) < small:
        m = nu
        e0 = nu
    else:
        #  ---------------------- elliptical -----------------------
        if ecc < 1.0 - small:
            sine = (sqrt(1.0 - ecc * ecc) * sin(nu)) / (1.0 + ecc * cos(nu))
            cose = (ecc + cos(nu)) / (1.0 + ecc * cos(nu))
            e0 = atan2(sine, cose)
            m = e0 - ecc * sin(e0)
        else:
            #  -------------------- hyperbolic  --------------------
            if ecc > 1.0 + small:

                if ecc > 1.0 and fabs(nu) + 0.00001 < pi - acos(1.0 / ecc):

                    sine = (sqrt(ecc * ecc - 1.0) * sin(nu)) / (1.0 + ecc * cos(nu))
                    e0 = asinh(sine)
                    m = ecc * sinh(e0) - e0

            else:
                #  ----------------- parabolic ---------------------
                if fabs(nu) < 168.0 * pi / 180.0:

                    e0 = tan(nu * 0.5)
                    m = e0 + (e0 * e0 * e0) / 3.0

    if ecc < 1.0:

        m = fmod(m, 2.0 * pi)
        if m < 0.0:
            m = m + 2.0 * pi
        e0 = fmod(e0, 2.0 * pi)

    return e0, m


def rv2coe(r, v, mu):
    """
    -------------------------------------------------------------------------
    Find the classical orbital elements given the geocentric equatorial
        position and velocity vectors; account or special orbits.
    origional author: david vallado, 2007-04-02
    minor edits: Jeff Belue, 2024-09-09

    inputs:
        r           - [km] ijk position vector
        v           - [km/s] ijk velocity vector
        mu          - [km^3/s^2] gravitational parameter

    outputs:
        p           - [km] semilatus rectum
        a           - [km] semimajor axis
        ecc         - [--] eccentricity
        incl        - [rad] inclination
        omega       - [rad] longitude of ascending node
        argp        - [rad] argument of perigee
        nu          - [rad] true anomaly
        m           - [rad] mean anomaly
        arglat      - [rad] argument of latitude (ci, circular inclined)
        truelon     - [rad] true longitude (ce, circular equatorial)
        lonper      - [rad] longitude of periapsis (ee, every thing else)

    locals:
        hbar        - [km^2/s] angular momentum h vector
        ebar        - eccentricity     e vector
        nbar        - line of nodes    n vector
        c1          - v**2 - u/r
        rdotv       - r dot v
        hk          - hk unit vector
        sme         - [km^2/s^2] specfic mechanical energy
        i           - index
        e           - eccentric, parabolic,
                      hyperbolic angle/anomaly
        temp        - temporary variable
        typeorbit   - orbit type; ee, ei, ce, ci

    coupling:
        mag         - magnitude of a vector
        cross       - cross product of two vectors
        angle       - find the angle between two vectors
        newtonnu    - find the mean anomaly

    references:
        vallado       2007, 126, alg 9, ex 2-5
    -------------------------------------------------------------------------- */
    """

    hbar = [None, None, None]
    nbar = [None, None, None]
    ebar = [None, None, None]
    typeorbit = [None, None, None]

    twopi = 2.0 * pi
    halfpi = 0.5 * pi
    small = 0.00000001
    undefined = 999999.1
    infinite = 999999.9

    #  -------------------------  implementation   -----------------
    magr = mag(r)
    magv = mag(v)

    #  ------------------  find h n and e vectors   ----------------
    cross(r, v, hbar)
    magh = mag(hbar)
    if magh > small:

        nbar[0] = -hbar[1]
        nbar[1] = hbar[0]
        nbar[2] = 0.0
        magn = mag(nbar)
        c1 = magv * magv - mu / magr
        rdotv = dot(r, v)
        for i in range(0, 3):
            ebar[i] = (c1 * r[i] - rdotv * v[i]) / mu
        ecc = mag(ebar)

        #  ------------  find a e and semi-latus rectum   ----------
        sme = (magv * magv * 0.5) - (mu / magr)
        if fabs(sme) > small:
            a = -mu / (2.0 * sme)
        else:
            a = infinite
        p = magh * magh / mu

        #  -----------------  find inclination   -------------------
        hk = hbar[2] / magh
        incl = acos(hk)

        #  --------  determine type of orbit for later use  --------
        #  ------ elliptical, parabolic, hyperbolic inclined -------
        typeorbit = "ei"
        if ecc < small:
            #  ----------------  circular equatorial ---------------
            if incl < small or fabs(incl - pi) < small:
                typeorbit = "ce"
            else:
                #  --------------  circular inclined ---------------
                typeorbit = "ci"

        else:
            #  - elliptical, parabolic, hyperbolic equatorial --
            if incl < small or fabs(incl - pi) < small:
                typeorbit = "ee"

        #  ----------  find longitude of ascending node ------------
        if magn > small:

            temp = nbar[0] / magn
            if fabs(temp) > 1.0:
                temp = copysign(1.0, temp)
            omega = acos(temp)
            if nbar[1] < 0.0:
                omega = twopi - omega

        else:
            omega = undefined

        #  ---------------- find argument of perigee ---------------
        if typeorbit == "ei":

            argp = angle(nbar, ebar)
            if ebar[2] < 0.0:
                argp = twopi - argp

        else:
            argp = undefined

        #  ------------  find true anomaly at epoch    -------------
        if typeorbit[0] == "e":

            nu = angle(ebar, r)
            if rdotv < 0.0:
                nu = twopi - nu

        else:
            nu = undefined

        #  ----  find argument of latitude - circular inclined -----
        if typeorbit == "ci":

            arglat = angle(nbar, r)
            if r[2] < 0.0:
                arglat = twopi - arglat
            m = arglat

        else:
            arglat = undefined

        #  -- find longitude of perigee - elliptical equatorial ----
        if ecc > small and typeorbit == "ee":

            temp = ebar[0] / ecc
            if fabs(temp) > 1.0:
                temp = copysign(1.0, temp)
            lonper = acos(temp)
            if ebar[1] < 0.0:
                lonper = twopi - lonper
            if incl > halfpi:
                lonper = twopi - lonper

        else:
            lonper = undefined

        #  -------- find true longitude - circular equatorial ------
        if magr > small and typeorbit == "ce":

            temp = r[0] / magr
            if fabs(temp) > 1.0:
                temp = copysign(1.0, temp)
            truelon = acos(temp)
            if r[1] < 0.0:
                truelon = twopi - truelon
            if incl > halfpi:
                truelon = twopi - truelon
            m = truelon

        else:
            truelon = undefined

        #  ------------ find mean anomaly for all orbits -----------
        if typeorbit[0] == "e":
            e, m = newtonnu(ecc, nu)

    else:
        p = undefined
        a = undefined
        ecc = undefined
        incl = undefined
        omega = undefined
        argp = undefined
        nu = undefined
        m = undefined
        arglat = undefined
        truelon = undefined
        lonper = undefined

    return p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper


def jday(year, mon, day, hr, minute, sec):
    """
    /* -----------------------------------------------------------------------------
    *                           procedure jday
    *  this procedure finds the julian date given the year, month, day, and time.
    *    the julian date is defined by each elapsed day since noon, jan 1, 4713 bc.
    *
    *  algorithm     : calculate the answer in one step for efficiency
    *
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *
    *  inputs          description                    range / units
    *    year        - year                           1900 .. 2100
    *    mon         - month                          1 .. 12
    *    day         - day                            1 .. 28,29,30,31
    *    hr          - universal time hour            0 .. 23
    *    min         - universal time min             0 .. 59
    *    sec         - universal time sec             0.0 .. 59.999
    *
    *  outputs       :
    *    jd          - julian date                    days from 4713 bc
    *
    *  locals        :
    *    none.
    *
    *  coupling      :
    *    none.
    *
    *  references    :
    *    vallado       2007, 189, alg 14, ex 3-14
    *
    * --------------------------------------------------------------------------- */
    """

    return (
        367.0 * year
        - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0
        + 275.0 * mon // 9.0
        + day
        + 1721013.5
        + ((sec / 60.0 + minute) / 60.0 + hr) / 24.0  #  ut in days
        #  - 0.5*sgn(100.0*year + mon - 190002.5) + 0.5;
    )


def jday2(year, mon, day, hr, minute, sec):
    """
    Return two floats that, when added, produce the specified Julian date.

    The first float specifies the day, while the second float specifies
    an additional offset for the hour, minute, and second.  Because the
    second float is much smaller in magnitude it can, unlike the first
    float, be accurate down to very small fractions of a second.

    """
    jd = (
        367.0 * year
        - 7 * (year + ((mon + 9) // 12.0)) * 0.25 // 1.0
        + 275 * mon / 9.0 // 1.0
        + day
        + 1721013.5
    )
    fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0
    return jd, fr


def days2mdhms(year, days):
    """
    /* -----------------------------------------------------------------------------
    *
    *                           procedure days2mdhms
    *
    *  this procedure converts the day of the year, days, to the equivalent month
    *    day, hour, minute and second.
    *
    *  algorithm     : set up array for the number of days per month
    *                  find leap year - use 1900 because 2000 is a leap year
    *                  loop through a temp value while the value is < the days
    *                  perform int conversions to the correct day and month
    *                  convert remainder into h m s using type conversions
    *
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *
    *  inputs          description                    range / units
    *    year        - year                           1900 .. 2100
    *    days        - julian day of the year         0.0  .. 366.0
    *
    *  outputs       :
    *    mon         - month                          1 .. 12
    *    day         - day                            1 .. 28,29,30,31
    *    hr          - hour                           0 .. 23
    *    min         - minute                         0 .. 59
    *    sec         - second                         0.0 .. 59.999
    *
    *  locals        :
    *    dayofyr     - day of year
    *    temp        - temporary extended values
    *    inttemp     - temporary int value
    *    i           - index
    *    lmonth[12]  - int array containing the number of days per month
    *
    *  coupling      :
    *    none.
    * --------------------------------------------------------------------------- */
    """

    lmonth = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    dayofyr = int(days // 1.0)
    #  ----------------- find month and day of month ----------------
    if (year % 4) == 0:
        lmonth = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    i = 1
    inttemp = 0
    while dayofyr > inttemp + lmonth[i - 1] and i < 12:

        inttemp = inttemp + lmonth[i - 1]
        i += 1

    mon = i
    day = dayofyr - inttemp

    #  ----------------- find hours minutes and seconds -------------
    temp = (days - dayofyr) * 24.0
    hr = int(temp // 1.0)
    temp = (temp - hr) * 60.0
    minute = int(temp // 1.0)
    sec = (temp - minute) * 60.0

    return mon, day, hr, minute, sec


def invjday(jd):
    """
    /* -----------------------------------------------------------------------------
    *                           procedure invjday
    *  this procedure finds the year, month, day, hour, minute and second
    *  given the julian date. tu can be ut1, tdt, tdb, etc.
    *
    *  algorithm     : set up starting values
    *                  find leap year - use 1900 because 2000 is a leap year
    *                  find the elapsed days through the year in a loop
    *                  call routine to find each individual value
    *
    *  author        : david vallado                  719-573-2600    1 mar 2001
    *
    *  inputs          description                    range / units
    *    jd          - julian date                    days from 4713 bc
    *
    *  outputs       :
    *    year        - year                           1900 .. 2100
    *    mon         - month                          1 .. 12
    *    day         - day                            1 .. 28,29,30,31
    *    hr          - hour                           0 .. 23
    *    min         - minute                         0 .. 59
    *    sec         - second                         0.0 .. 59.999
    *
    *  locals        :
    *    days        - day of year plus fractional
    *                  portion of a day               days
    *    tu          - julian centuries from 0 h
    *                  jan 0, 1900
    *    temp        - temporary double values
    *    leapyrs     - number of leap years from 1900
    *
    *  coupling      :
    *    days2mdhms  - finds month, day, hour, minute and second given days and year
    *
    *  references    :
    *    vallado       2007, 208, alg 22, ex 3-13
    * --------------------------------------------------------------------------- */
    """

    #  --------------- find year and days of the year ---------------
    temp = jd - 2415019.5
    tu = temp / 365.25
    year = 1900 + int(tu // 1.0)
    leapyrs = int(((year - 1901) * 0.25) // 1.0)

    #  optional nudge by 8.64x10-7 sec to get even outputs
    days = temp - ((year - 1900) * 365.0 + leapyrs) + 0.00000000001

    #  ------------ check for case of beginning of a year -----------
    if days < 1.0:
        year = year - 1
        leapyrs = int(((year - 1901) * 0.25) // 1.0)
        days = temp - ((year - 1900) * 365.0 + leapyrs)

    #  ----------------- find remaing data  -------------------------
    mon, day, hr, minute, sec = days2mdhms(year, days)
    sec = sec - 0.00000086400
    return year, mon, day, hr, minute, sec


# *******************************************************************************
def test_rv2coe():
    print(f"\nTest Vallado function rv2cov():")
    # function does not need input parameters.
    # r_vec = [1000, 5000, 7000]  # km
    # v_vec = [3.0, 4.0, 5.0]  # km/s
    # - earth R,V data 1988-04-08, from Horizons, https://ssd.jpl.nasa.gov/horizons/app.html#/
    r_vec = [-1.409206342323255e08, 4.884114655615644e07, 2.926503946314380e04]
    v_vec = [-1.039747672698875e01, -2.820517858596729e01, 2.495229861189330e-03]
    # r_vec = [-1.42019455e+08, -4.37089122e+07, -1.89514749e+07]  # [km]
    # v_vec = [8.97691931, -26.01359291, -11.27913009]  # [km/s]
    # mu = 3.98600e5  # earth [km^3/s^2]
    mu = 1.3271544e11  # sun [km^3/s^2], sun
    p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper = rv2coe(
        r_vec, v_vec, mu
    )

    rad2deg = 180 / pi

    incl_deg = incl * rad2deg
    omega_deg = omega * rad2deg
    argp_deg = argp * rad2deg
    nu_deg = nu * rad2deg
    m_deg = m * rad2deg
    arglat_deg = arglat * rad2deg
    truelon_deg = truelon * rad2deg
    lonper_deg = lonper * rad2deg

    print(
        f"p= {p:.8g} [km]; "
        f"\na or sma= {a:.8g} [km]; "
        f"\necc= {ecc:.8g}; "
        f"\nincl= {incl_deg:.8g} [deg]; "
        f"\nomega or RAAN=, {omega_deg:.6g} [deg]; "
        f"\nargp= {argp_deg:.6g} [deg]; "
        f"\ntrue anomaly, nu=, {nu_deg:.6g} [deg]"
        f"\nmean anomaly, m=, {m_deg:.6g} [deg]"
        f"\narguement of lattitude, arglat=, {arglat_deg:.6g} [deg]"
        f"\ntrue anomaly, truelon=, {truelon_deg:.6g} [deg]"
        f"\nlongitude of periapsis, lonper=, {lonper_deg:.6g} [deg]"
    )

    return None


# use the following to test/examine functions
if __name__ == "__main__":

    test_rv2coe()  # test Vallado's rv2coe
