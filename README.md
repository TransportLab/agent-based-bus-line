# Agent Based Bus Line
A python package for simulating a multi-lane bus route using agent-based modelling.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Installation

1.  Download and unzip (or clone) this repository
2.  Install python
3.  Install the required python packages with `pip install -r requirements.txt`
4.  Run the code with `python ptips.py`. Parameters you can adjust are contained at the bottom of that file.

# Features
1.  Closed loop traffic flow. All vehicles move in a single direction along the route. Cars have a probability of exiting the flow at green traffic lights and re-enter at red traffic lights to maintain a constant number of vehicles overall.
2.  Multi-lane traffic. Busses remain in left-most lane at all times. Other vehicles change lanes as necessary.
3.  Bus passenger routing system. Passengers arrive at bus stops over time. When loaded onto a bus, passengers are routed towards stops uniformly (i.e. all stops have equal weighting).

# Parameters
All parameters are defined in a `json5` file. See `default.json` for some default values:

## Road system
-   `L = 1000`  # circumference of circle (m)

## Time marching
-   `t_max = 1e3`  # maximum time (s)
-   `dt = 1e-1`  # time increment (s)

## Traffic properties
-   `initial_vehicle_spacing = 100`  # (m/vehicle)
-   `speed_limit = 60 / 3.6`  # maximum velocity (m/s)
-   `free_flowing_acceleration = 3`  # typical vehicle acceleration (m/s^2)
-   `lanes = 2`  # how many lanes

## Bus system
-   `bus_fraction = 0.1`  # what fraction of vehicles are busses (-)
-   `passenger_accumulation_rate = 0.1`  # passengers arriving at a stop every second (passengers/s)
-   `passenger_ingress_egress_rate = 1`  # how long to get on/off the bus (passengers/s)
-   `bus_max_capacity = 50`  # maximum number of passengers on an individual bus (passengers/vehicle)
-   `bus_stop_traffic_light_offset = 0.5`  # 0.1ish for just after the traffic lights, 0.9ish for just before traffic lights, 0.5 for in between (-)

## Traffic light properties
-   `traffic_light_spacing = L / 4.0`  # (m)
-   `traffic_light_period = 60`  # (s)
-   `traffic_light_green_fraction = 0.5`  # fraction of time it is _green_ (-)
-   `car_entry_exit_probability = 0.1`  # probability of moving to a different traffic light

## Vehicle interaction properties
-   `stiffness = 1e4`  # how much cars repel each other (also used for traffic lights, which are the same as stopped cars)
-   `sigma = 10`  # typical stopping distance (m)

## PTIPS stuff
-   `scheduled_velocity = 0.6` * speed_limit  # how fast the busses are scheduled to move (m/s)
-   `ptips_delay_time = 10`  # how much delay before PTIPS kicks in (s)
-   `ptips_capacity_threshold = 0.8`  # how full should the busses be before ptips kicks in (-)
