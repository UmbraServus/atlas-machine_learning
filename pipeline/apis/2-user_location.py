#!/usr/bin/env python3
"""module for getting user location"""
import requests
import sys
import time

import requests as r

if __name__ == '__main__':
    url = sys.argv[1]
    response = r.get(url)

    if response.status_code == 404:
        print("Not Found")

    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-RateLimit-Reset'))
        current_time = int(time.time())
        time_remaining = (reset_time - current_time) // 60

        print(f"Reset in {time_remaining} min")

    elif response.status_code == 200:
        data = response.json()
        print(data.get('location'))

    else:
        print("Not Found")
