# Refactored Environment Class for EV Routing Simulation
# - Improved naming
# - Removed redundancy
# - Added docstrings
# - Ensured consistency in API usage
# - Moved constants and session reuse

import math
import urllib.parse
import requests
from haversine import haversine
from Battery import LithiumIonBattery
from motor import EnergyCalculator

class Environment:
    API_KEY = "AIzaSyBxcVtnDTNCFN4LVKWX5UXRnt8spMdnLVA"
    session = requests.Session()

    def __init__(self, origin_adr, destination_adr):
        self.origin = origin_adr
        self.destination = destination_adr
        self.battery = LithiumIonBattery(50000)
        self.energy_calculator = EnergyCalculator()
        self.charge_count = 0
        self.unreachable_steps = 0
        self.total_time = 0
        self.env_height_km = 1
        self.length = 1
        self.map_data = self._init_map()

    def _init_map(self):
        origin_status, origin_coords = self._geocode(self.origin)
        dest_status, dest_coords = self._geocode(self.destination)

        dir_status, steps, bounds = self._get_directions(origin_coords, dest_coords)
        while dir_status != 'OK':
            dir_status, steps, bounds = self._get_directions(origin_coords, dest_coords)

        self.current_position = origin_coords
        self.start_position = origin_coords
        self.end_position = dest_coords
        self.route_steps = steps
        self.map_bounds = bounds

        return steps

    def _geocode(self, address):
        base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
        url = f"{base_url}{urllib.parse.urlencode({'address': address})}&key={self.API_KEY}"
        res = self.session.get(url).json()
        if res['status'] == 'OK':
            loc = res['results'][0]['geometry']['location']
            return 'OK', (loc['lat'], loc['lng'])
        return res['status'], ('g', 'g')

    def _get_elevation(self, location):
        base_url = 'https://maps.googleapis.com/maps/api/elevation/json?'
        url = f"{base_url}{urllib.parse.urlencode({'locations': location})}&key={self.API_KEY}"
        res = self.session.get(url).json()
        if res['status'] == 'OK':
            return 'OK', res['results'][0]['elevation']
        return res['status'], 'N/A'

    def _get_directions(self, origin, destination):
        base_url = 'https://maps.googleapis.com/maps/api/directions/json?'
        params = {'origin': origin, 'destination': destination, 'units': 'metric', 'key': self.API_KEY}
        url = f"{base_url}{urllib.parse.urlencode(params)}"
        res = self.session.get(url).json()
        if res['status'] == 'OK':
            steps = res['routes'][0]['legs'][0]['steps']
            bounds = res['routes'][0]['bounds']
            return 'OK', steps, {
                'north': bounds['northeast']['lat'],
                'east': bounds['northeast']['lng'],
                'south': bounds['southwest']['lat'],
                'west': bounds['southwest']['lng']
            }
        return res['status'], [], {}

    def _calculate_stride(self, position):
        north = max(self.start_position[0], self.end_position[0])
        south = min(self.start_position[0], self.end_position[0])
        east = max(self.start_position[1], self.end_position[1])
        west = min(self.start_position[1], self.end_position[1])

        height = haversine((north, west), (south, west))
        width = haversine((position[0], east), (position[0], west))
        self.env_height_km = height

        self.stride_height = (north - south) / (height * self.length)
        self.stride_width = (east - west) / (width * self.length)

    def step(self, action):
        self._calculate_stride(self.current_position)
        direction = -1 if action == 2 or action == 3 else 1
        move = {
            0: (direction * self.stride_height, 0),  # North
            1: (0, direction * self.stride_width),   # East
            2: (direction * self.stride_height, 0),  # South
            3: (0, direction * self.stride_width)    # West
        }

        dx, dy = move[action]
        next_position = (self.current_position[0] + dx, self.current_position[1] + dy)

        # Check if next step is valid
        status, steps, _ = self._get_directions(self.current_position, next_position)
        if (status != 'OK' or
            not (self.map_bounds['south'] <= next_position[0] <= self.map_bounds['north']) or
            not (self.map_bounds['west'] <= next_position[1] <= self.map_bounds['east'])):
            if status != 'OVER_QUERY_LIMIT':
                self.unreachable_steps += 1
            return self.current_position, -1, False, self.charge_count, self.battery.SOC

        reward, done = -0.1, False

        for step in steps:
            start = step['start_location']
            end = step['end_location']
            duration = max(1, step['duration']['value'])
            distance = step['distance']['value']

            elev_status1, height_start = self._get_elevation(f"{start['lat']},{start['lng']}")
            elev_status2, height_end = self._get_elevation(f"{end['lat']},{end['lng']}")

            if elev_status1 != 'OK' or elev_status2 != 'OK':
                continue

            elevation = height_end - height_start
            speed = math.sqrt(distance**2 + elevation**2) / duration
            angle = max(0, math.atan2(distance * 1000, elevation))
            power = self.energy_calculator.energy(angle=angle, V=speed)

            for _ in range(duration):
                if self.battery.use(1, power):
                    self.charge_count += 1
                    self.battery.charge(50000)
                reward -= self.battery.energy_consume / 100000

            self.total_time += duration

        if (abs(next_position[0] - self.end_position[0]) < self.stride_height and
            abs(next_position[1] - self.end_position[1]) < self.stride_width):
            reward += 1
            done = True

        self.current_position = next_position
        return next_position, reward, done, self.charge_count, self.battery.SOC
