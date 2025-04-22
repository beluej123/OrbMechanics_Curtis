""" astro_classes attempts to collect/organize data sources
    search: python formulate class of solar system body
"""


class SolarSystemBody:
    def __init__(self, name, mass, radius, orbital_period):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.orbital_period = orbital_period

    def __str__(self):
        return f"{self.name} (Mass: {self.mass}, Radius: {self.radius}, Orbital Period: {self.orbital_period})"


class Star(SolarSystemBody):
    def __init__(self, name, mass, radius, surface_temperature):
        super().__init__(name, mass, radius, orbital_period=0)
        self.surface_temperature = surface_temperature

    def __str__(self):
        return f"{super().__str__()} Surface Temperature: {self.surface_temperature}"


class Planet(SolarSystemBody):
    def __init__(self, name, mass, radius, orbital_period, distance_from_star, btype):
        super().__init__(name, mass, radius, orbital_period)
        self.distance_from_star = distance_from_star
        self.btype = btype  # e.g., body type "Gas Giant", "Terrestrial"

    def __str__(self):
        return f"{super().__str__()} Distance from Star: {self.distance_from_star}, Type: {self.btype}"


class DwarfPlanet(Planet):
    def __init__(self, name, mass, radius, orbital_period, distance_from_star):
        super().__init__(
            name, mass, radius, orbital_period, distance_from_star, btype="Dwarf Planet"
        )


class Comet(Planet):
    def __init__(self, name, mass, radius, orbital_period, distance_from_star):
        super().__init__(
            name, mass, radius, orbital_period, distance_from_star, btype="Comet"
        )


class Moon(SolarSystemBody):
    def __init__(self, name, mass, radius, orbital_period, parent_planet):
        super().__init__(name, mass, radius, orbital_period)
        self.parent_planet = parent_planet

    def __str__(self):
        return f"{super().__str__()} Parent Planet: {self.parent_planet}"


if __name__ == "__main__":
    # Example Usage
    sun = Star(name="Sun", mass=1.989e30, radius=695700, surface_temperature=5778)
    earth = Planet(
        name="Earth",
        mass=5.972e24,
        radius=6371,
        orbital_period=365.25,
        distance_from_star=149.6e6,
        btype="Terrestrial",
    )
    mars = Planet(
        name="Mars",
        mass=5.972e24,
        radius=6371,
        orbital_period=365.25,
        distance_from_star=149.6e6,
        btype="Terrestrial",
    )
    moon = Moon(
        name="Moon",
        mass=7.348e22,
        radius=1737,
        orbital_period=27.3,
        parent_planet="Earth",
    )
    pluto = DwarfPlanet(
        name="Pluto",
        mass=1.309e22,
        radius=1188,
        orbital_period=90560,
        distance_from_star=5.906e9,
    )
    ceres = DwarfPlanet(
        name="Ceres",
        mass=1.309e22,
        radius=1188,
        orbital_period=90560,
        distance_from_star=5.906e9,
    )
    halley = Comet(
        name="Halley",
        mass=1.309e22,
        radius=1188,
        orbital_period=90560,
        distance_from_star=5.906e9,
    )

    print(sun)
    print(earth)
    print(moon)
    print(mars)
    print(pluto)
    print(ceres)
    print(halley)
