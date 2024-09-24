"""
General goal: find local equinox.

Several routines in this file.  See testing guidance at the end of this file.
2024-09-22 copied from the web:
Find the nearest syzygy (solstice or equinox) to a given date.
https://gist.github.com/nealmcb/858bbc14e9a40b5d88f1428fe53456d0
Solution for https://stackoverflow.com/questions/55838712/finding-equinox-and-solstice-times-with-astropy
TODO: need to ensure we're using the right sun position functions for the syzygy definition....
"""

import math
from datetime import datetime

import astropy.coordinates
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.time import Time, TimeDelta
from scipy.optimize import brentq

# We'll usually find a zero crossing if we look this many days before and after
# given time, except when it is is within a few days of a cross-quarter day.
# But going over 50,000 years back, season lengths can vary from 85 to 98 days!
#  https://individual.utoronto.ca/kalendis/seasons.htm#seasons
delta = 44.0


def mjd_to_time(mjd):
    "Return a Time object corresponding to the given Modified Julian Date."

    return Time(mjd, format="mjd", scale="utc")


def sunEclipticLongitude(mjd):
    "Return ecliptic longitude of the sun in degrees at given time (MJD)"

    t = mjd_to_time(mjd)
    sun = astropy.coordinates.get_body("sun", t)
    # TODO: Are these the right functions to call for the definition of syzygy? Geocentric? True?
    eclipticOfDate = astropy.coordinates.GeocentricTrueEcliptic(equinox=t)
    sunEcliptic = sun.transform_to(eclipticOfDate)
    return sunEcliptic.lon.deg


def linearize(angle):
    """
    Map angle values in degrees near the quadrants of the circle
    into smooth linear functions crossing zero, for root-finding algorithms.
    Note that for angles near 90 or 270, increasing angles yield decreasing results
    >>> linearize(5) > 0 > linearize(355)
    True
    >>> linearize(95) > 0 > linearize(85)
    False
    """
    return math.sin(math.radians(angle * 2))


def map_syzygy(t):
    "Map times into linear functions crossing zero at each syzygy"
    return linearize(sunEclipticLongitude(t))


def find_nearest_syzygy(t):
    """
    Return time of the nearest syzygy to the given Time, which must be
        within 43 days of one syzygy.

    syzygy = the nearly straight-line configuration of three celestial
        bodies (such as the sun, moon, and earth during a solar or lunar eclipse).
    """

    syzygy_mjd = brentq(map_syzygy, t.mjd - delta, t.mjd + delta)

    syzygy = mjd_to_time(syzygy_mjd)
    syzygy.format = "isot"
    return syzygy


# ******* seperate collection below *********************************
"""
Daylight hours by latitude.
https://gist.github.com/mluis7/4caeb4edcadcef0e74d0a7c3fde8df5c

Values for Mendoza, Argentina; year 2024.

By changing the dlatitudes dictionary values it can be used for any place on
    earth.  Fill dlatitudes dictionary with label and latitude of places of
    interest.  Also change loc1, loc2 with keys to highlight from dlatitudes
    (max/min hours, equinox and solstice days).
Generates two png images; one with curves of hours per latitude and another of
    isochrones per latitude.
The latitudes in dlatitudes must be floating point with 4 decimal places.
"""

def daylength(dayOfYear, lat):
    """Taken from:
    https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
    but using math instead of nympy.

    Computes the length of the day (the time between sunrise and sunset)
        given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example, Forsythe et al., "A model comparison
        for daylength as a function of latitude and day of year",
        Ecological Modelling, 1995.
    Input Parameters:
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = math.radians(lat)
    declinationOfEarth = 23.45 * math.sin(
        math.radians(360.0 * (283.0 + dayOfYear) / 365.0)
    )
    if -math.tan(latInRad) * math.tan(math.radians(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -math.tan(latInRad) * math.tan(math.radians(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = math.degrees(
            math.acos(-math.tan(latInRad) * math.tan(math.radians(declinationOfEarth)))
        )
        return 2.0 * hourAngle / 15.0


def year_day_to_date(year, yday):
    return datetime.strptime(f"{year}-{yday}", "%Y-%j").strftime("%Y-%m-%d")


def date_to_year_day(d):
    return datetime.strptime(d, "%Y-%m-%d").strftime("%j")


def get_xtick_labels(year):
    xticks = {}
    xticks[-50.0] = ""
    xticks[0.0] = f"{year}-01-01"
    for m in range(2, 13):
        for d in [1]:
            xticks[float(date_to_year_day(f"{year}-{m}-{d}"))] = f"{year}-{m:02}-{d:02}"
    return xticks


def format_plot(plt, df, equinoxes, lat_key, lat_value):
    for s in equinoxes:  #:
        plt.axvline(
            x=s,
            color="gray",
            label=year_day_to_date(year, s),
            linestyle="dashdot",
            alpha=0.5,
            zorder=1,
        )
        plt.text(s - 23, 9.4, year_day_to_date(year, s))

    plt.axhline(
        y=df.min()[lat_value],
        color="gray",
        label="Min",
        linestyle="dotted",
        alpha=0.5,
        zorder=1,
    )
    plt.text(
        385, df.min()[lat_value], f"Mín. {lat_key} ({round(df.min()[lat_value], 2)})"
    )
    plt.axhline(
        y=df.max()[lat_value],
        color="gray",
        label="Max",
        linestyle="dotted",
        alpha=0.5,
        zorder=1,
    )
    plt.text(
        385, df.max()[lat_value], f"Máx. {lat_key} ({round(df.max()[lat_value], 2)})"
    )

    plt.xlabel(None)
    plt.ylabel("Horas de Luz", fontsize=14)
    plt.title(f"Horas de Luz [Latitud Sur]", fontsize=18)


def dayLength_lat():

    # Create a list of data to
    # be represented in x-axis
    year = 2024
    year_days = list(range(1, 366))

    # lavalle -32.72417230633202, -68.59460899793771
    # mendoza -32.88980681767816, -68.822876003072
    # tunuyan -33.58149627625932, -69.01873380990953
    # san rafael -34.616467633671355, -68.33825669564037
    # malargüe -35.473122175850946, -69.58419276211484
    # Ranquil Norte -36.65897417899151, -69.83031603573339
    # City to latitude dictionary.
    dlatitudes = {
        # "Usuhaia": -54.7965,
        # "Trelew": -43.2119,
        "Ranquil Norte": -36.6589,
        "Malargüe": -35.4731,
        "San Rafael": -34.6165,
        "Tunuyán": -33.5815,
        "Mendoza": -32.8898,
        "Capilla del Rosario": -32.1442,
        # "Jujuy": -24.1853,
        # "Quito (Ecuador)": -0.1866
    }
    # location of interest
    loc1 = "Mendoza"
    loc2 = "Ranquil Norte"

    latlegends = [f"{k:20} (Lat: {v})" for k, v in dlatitudes.items()]
    latitudes = dlatitudes.values()
    # day of year for equinoxes and solstices
    equinoxes = []
    # build X-axe date labels
    year_days_season = [year_day_to_date(year, y) for y in year_days]
    xticks = get_xtick_labels(year)

    # Hours of daylight per latitude - y axe Hours
    lat_hours = {"days": year_days_season}

    # build series of hours of daylight per latitude
    for lat in latitudes:
        if lat not in lat_hours.keys():
            lat_hours[lat] = []
        for day in year_days:
            dlen = daylength(day, lat)
            lat_hours[lat].append(round(dlen, 4))
            # equinoxes for latitude of interest
            # if lat == dlatitudes[loc1] and round(dlen,2) >= 11.98 and round(dlen,2) <= 12.01:
            #     # add spring/autumn equinoxes
            #     equinoxes.append(day)

    # Create a dataframe
    df = pd.DataFrame(lat_hours)
    # dataframe for a single location
    dfmdz = df[[dlatitudes[loc1]]]
    # find spring/autumn equinoxes days (day = night = 12.0 hours)
    equinoxes = dfmdz.iloc[
        (dfmdz[dlatitudes[loc1]] - 12.0).abs().argsort()[:2]
    ].index.values.tolist()
    # find max, min hours indexes for a location - summer/winter solstices
    equinoxes.append(dfmdz.idxmax()[dlatitudes[loc1]])
    equinoxes.append(dfmdz.idxmin()[dlatitudes[loc1]])
    equinoxes.sort()

    ax = plt.gca()

    # use plot() method on the dataframe
    df.plot(x="days", ax=ax, figsize=(20, 12))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, latlegends, fontsize=11)
    format_plot(plt, dfmdz, equinoxes, loc1, dlatitudes[loc1])
    # Add X-axe ticks every first day of month
    plt.xticks([x for x in xticks.keys()], xticks.values(), rotation=30)

    plt.savefig("lat-hours.png")
    print("Done hours by latitude")
    plt.cla()

    # Isochrones - latitudes with same hours of daylight - Y axe latitude
    # ###################################################################
    hours_lat = {"days": year_days_season}

    # Lines of equal hours per latitude (isochrones).
    # Transpose Latitud to hours series to Hours to Latitude.
    # Massage series for plotting to avoid cluttering the graph.
    float_filter = [10.4, 11.4, 11.8, 12.4, 12.8, 13.4]
    for i, lt in enumerate(latitudes):
        for d, h in enumerate(df[lt]):
            k = int(h)

            # do not show some repeated values that draw as a flat line on top
            if int(k) in [10, 11, 12, 13] and round(h, 1) not in float_filter:
                continue

            # use float for particular values so those hours are better represented in plot.
            if round(h, 1) in float_filter:
                k = round(h, 1)

            # create an initialize list
            if k not in hours_lat.keys():
                hours_lat[k] = []
                for day in year_days:
                    hours_lat[k].insert(day, None)

            hours_lat[k][d] = lt

    dfhl = pd.DataFrame.from_dict(hours_lat, orient="index")
    dfhl = dfhl.transpose()

    print("Done rev df")

    dfhl.plot(x="days", ax=ax, figsize=(15, 5), marker="o", markersize=2)
    plt.xlabel(None)
    plt.ylabel("Latitud Sur", fontsize=14)
    plt.title("Isócronas por Latitudes")

    # horizontal lines for latitudes of interest
    for loc in [loc1, loc2]:
        plt.axhline(
            y=dlatitudes[loc],
            color="gray",
            label=f"{loc} ({round(dlatitudes[loc],1)})",
            linestyle="dotted",
            alpha=0.5,
            zorder=1,
        )
    # vertical lines for solstices/equinoxes
    for s in equinoxes:  #:
        plt.axvline(
            x=s,
            color="dimgray",
            label=year_day_to_date(year, s),
            linestyle="dashdot",
            alpha=0.5,
            zorder=1,
        )

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.3, -0.15),
        ncol=3,
        fontsize="x-small",
    )
    plt.xticks(
        [x for x in xticks.keys()], xticks.values(), rotation=30, fontsize="x-small"
    )

    plt.savefig("lat-hours-iso.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    return None  # dayLength_lat()


def test_local_equinox():
    # import doctest # 2024-09-23, I've not used doctest before...
    # doctest.testmod()

    t0 = Time("2019-09-23T07:50:10", format="isot", scale="utc")
    td = TimeDelta(1.0 * u.day)

    seq = t0 + td * range(0, 365, 15)

    for t in seq:
        try:
            syzygy = find_nearest_syzygy(t)
        except ValueError as e:
            print(f"{e=}, {t.value=}, {t.mjd-delta=}, {map_syzygy(t.mjd-delta)=}")
            continue
        print(f"{t.value=}, {syzygy.value=}, {sunEclipticLongitude(syzygy)=}")
    return None


def test_day_length():
    dayLength_lat()
    return None

def main():  # used navigational aid for editor
    return

# function test coordination
if __name__ == "__main__":
    test_local_equinox()  # 2024-09-23, runs, but not fooled with much
    # test_day_length()  # 2024-09-23, well not fooled with much
