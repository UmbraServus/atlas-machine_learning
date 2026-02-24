#!/usr/bin/env python3
"""module for using requests to retrieve data from swapi
abt sentient species"""
import requests as r


def sentientPlanets():
    """method that returns the list of names of the home planets of
    all sentient species.
    returns:
        list of names of all home planets"""
    url = "https://swapi.dev/api/species/"
    planets = []

    while url:
        response = r.get(url)
        data = response.json()

        for species in data['results']:
            if (species['classification'] == "sentient"
                    or species['designation'] == "sentient"):
                planet_url = species['homeworld']

                if planet_url:
                    planet_response = r.get(planet_url)
                    planet_data = planet_response.json()
                    planets.append(planet_data['name'])

        url = data['next']
    return planets
