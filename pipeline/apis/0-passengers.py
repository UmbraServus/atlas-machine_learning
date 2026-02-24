#!/usr/bin/env python3
"""module that returns list of ships that can carry passengers from swapi"""
import requests as r


def availableShips(passengerCount):
    """returns list of ships that can carry passengers from swapi
    args:
    passengerCount (int): number of passengers

    returns:
    list of ships that can carry passengers from swapi"""
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships'

    while url:
        response = r.get(url)
        data = response.json()

        for ship in data['results']:
            passengers = ship['passengers']
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship['name'])

        url = data['next']
    return ships
