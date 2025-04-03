"""
References collection that were used somewhere in ...

References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.),
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.),
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.), yes later versions exist,
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
    [4] Vallado, David A., (2022, 5th ed., 1st printing),
        Fundamentals of Astrodynamics and Applications, Microcosm Press.
            Focus on chap.12 interplanetary mission analysis, but support from
            the whole book; be careful there are errors
    [5] Wertz, James R., (2001, 5th ed.),
        Mission Geometry; Orbit and Constellation Design and Management,
        Space Technology Library, Microcosm Press et.al.
        Julian dates, chap.4, section time, p.185,
            review https://en.wikipedia.org/wiki/Julian_day
    [6] Meeus, Jean, (1998 2nd. ed.) Astronomical Algorithms,
        Willmann-Bell.  Julian dates, pp.61-62 in particular
    [7] Duffett-Smith et.al. (2017 4th ed.),
        Practical Astronomy with your Calculator or Spreadsheet,
            section 4, Julian dates (CDJD), pp.8-31
    [8] Brown, C.D., (1992 2nd printing), AIAA Education Series,
        Spacecraft Mission Design,
            section 6, Interplanetary Trajectories, pp.95-131
Notes:
----------
    Julian date conversions (BCE, pre-Gregorian/Gregorian) is not as
        straight forward as I expected.
        [6] Meeus outlines reasonable conversion boundry tests, but it took
            [7] Duffett-Smith to make the calculation implementation clear.
            I understand the conversion concepts, but the algorythm explanation
            is still not clear to me.  Note some modern conversions may not pass
            the boundary tests.

"""
