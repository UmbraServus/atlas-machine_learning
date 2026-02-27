#!/usr/bin/env python3
"""Displays the first SpaceX launch in a readable format"""
import requests as r
from datetime import datetime
from operator import itemgetter


def get_first_launch():
    """Fetch and display the first SpaceX launch"""
    launch_url = 'https://api.spacexdata.com/v4/launches/'
    rockets_url = 'https://api.spacexdata.com/v4/rockets/'
    launchpads_url = 'https://api.spacexdata.com/v4/launchpads/'

    # Fetch all launches
    response = r.get(launch_url)
    response.raise_for_status()
    launches = response.json()

    # Use min() with itemgetter to get the launch with the smallest date_unix
    first_launch = min(launches, key=itemgetter('date_unix'))

    # Launch name
    launch_name = first_launch.get('name')

    # launch date local time
    date_unix = first_launch.get('date_unix')
    local_date = datetime.fromtimestamp(date_unix)

    # Rocket name
    rocket_id = first_launch.get('rocket')
    rocket_response = r.get(rockets_url + rocket_id)
    rocket_response.raise_for_status()
    rocket_name = rocket_response.json().get('name')

    # Launchpad name and locality
    launchpad_id = first_launch.get('launchpad')
    launchpad_response = r.get(launchpads_url + launchpad_id)
    launchpad_response.raise_for_status()
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data.get('name')
    launchpad_location = launchpad_data.get('locality')

    # Print formatted output
    print(f"{launch_name} ({local_date}) {rocket_name} - {launchpad_name} ({launchpad_location})")

if __name__ == "__main__":
    get_first_launch()
