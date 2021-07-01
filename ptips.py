"""
Questions to ask of model:
  - How does PTIPS behave when the _schedule_ is bad?

"""

from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt
from plotting import show_road_system


def gaussian(d, sigma):
    return np.exp(-(d ** 2) / 2 / sigma ** 2) / (np.sqrt(2 * np.pi) * sigma)


def time_march(p, verbose, GRAPH):
    # Initialise system
    t = 0  # current time (s)
    tstep = 0  # time step (-)
    nt = int(p.t_max // p.dt)
    gamma = p.free_flowing_acceleration / (p.speed_limit ** 2)  # drag coefficient
    num_vehicles = int(p.L // p.initial_vehicle_spacing)
    num_traffic_lights = int(p.L // p.traffic_light_spacing)

    position = np.linspace(0, p.L, num_vehicles, endpoint=False)
    # position += vehicle_spacing*np.random.rand(num_vehicles)
    total_displacement = np.zeros_like(position)
    velocity = np.zeros_like(position)

    traffic_lights = np.linspace(0, p.L, num_traffic_lights, endpoint=False)
    traffic_light_phasing = np.linspace(0, 2 * np.pi, num_traffic_lights, endpoint=False)

    bus_stop_locations = traffic_lights.copy() + p.bus_stop_traffic_light_offset * (
        traffic_lights[1] - traffic_lights[0]
    )
    bus_stop_queue = np.zeros_like(bus_stop_locations)  # no passengers anywhere
    bus = np.random.choice(num_vehicles, int(p.bus_fraction * num_vehicles), replace=False)
    car = np.delete(range(num_vehicles), bus)
    bus_fullness = np.zeros(
        [len(bus), len(bus_stop_locations)]
    )  # for each bus, a list of passengers by destination
    bus_motion = np.zeros_like(bus, dtype=int)  # 0=moving, 1=unloading, 2=loading

    for tstep in progressbar(range(nt)):
        # Everyone loves to accelerate
        acceleration = p.free_flowing_acceleration - gamma * velocity ** 2
        # acceleration[0] = 0 # fix first car

        # Check car in front
        relative_distance = np.roll(position, -1) - position
        relative_distance[relative_distance < 0] = (
            p.L + relative_distance[relative_distance < 0]
        )  # account for periodicity
        interaction = p.stiffness * gaussian(relative_distance, p.sigma)
        interaction[velocity <= 0] = 0
        acceleration -= interaction

        # Check traffic lights
        # if t % p.traffic_light_period < p.traffic_light_green_fraction * p.traffic_light_period:
        # green = True
        # else:
        # green = False
        green = (
            0.5 * (np.cos(2 * np.pi * t / p.traffic_light_period + traffic_light_phasing) + 1)
            < p.traffic_light_green_fraction
        )

        for i, light in enumerate(traffic_lights):
            if not green[i]:
                distance_to_light = light - position
                distance_to_light[position > light] += p.L  # account for periodicity
                stopping_vehicles = (distance_to_light < 3 * p.sigma) * (distance_to_light > 0)
                acceleration[stopping_vehicles] -= (
                    10 * p.stiffness * gaussian(distance_to_light[stopping_vehicles], p.sigma)
                )

        # Update passengers at stops
        bus_stop_queue += p.passenger_accumulation_rate * p.dt

        # Check bus stops
        for j, b in enumerate(bus):
            for i, stop in enumerate(bus_stop_locations):
                if position[b] > stop:
                    distance_to_stop = stop - position[b] + p.L
                else:
                    distance_to_stop = stop - position[b]
                if distance_to_stop < 3 * p.sigma and distance_to_stop > 0:
                    if (
                        (
                            (bus_stop_queue[i] > 1 and np.sum(bus_fullness[j, :]) < p.bus_max_capacity)
                            or bus_fullness[j, i] > 1
                        )
                        and bus_motion[j] == 0
                    ):  # bus is moving and hits a stop with at least one person or one person wants to get off
                        bus_motion[j] = 1  # move to unloading phase
                    if bus_motion[j] > 0:
                        acceleration[b] -= p.stiffness * gaussian(distance_to_stop, p.sigma)
                    if velocity[b] == 0.0:  # only once the bus has stopped
                        if bus_motion[j] == 1:  # unloading
                            # print(f'unloading {j} at stop {i}. Current fullness {bus_fullness[j,i]}')
                            if (
                                bus_fullness[j, i] > 0
                            ):  # if there are passengers on this bus who want to get off here
                                bus_fullness[j, i] -= (
                                    p.passenger_ingress_egress_rate * p.dt
                                )  # passengers leave bus
                            else:
                                bus_motion[j] = 2
                        elif bus_motion[j] == 2:  # loading
                            # print(f'LOADING {j} at stop {i}. Current fullness {bus_fullness[j,i]}')
                            if bus_stop_queue[i] > 0 and np.sum(bus_fullness[j, :]) <= p.bus_max_capacity:
                                bus_fullness[j, :i] += (
                                    p.passenger_ingress_egress_rate * p.dt / len(bus_stop_queue - 1)
                                )  # passengers go onto bus
                                bus_fullness[j, i + 1 :] += (
                                    p.passenger_ingress_egress_rate * p.dt / len(bus_stop_queue - 1)
                                )  # passengers go onto bus
                                bus_stop_queue[i] -= (
                                    p.passenger_ingress_egress_rate * p.dt
                                )  # passengers leave stop
                            else:
                                bus_motion[j] = 0  # start moving again
            bus_fullness[
                bus_fullness < 0
            ] = 0  # HACK!!! real issue is not checking if we are going to over-empty the bus in a given timestep, but this is way too small an issue to bother fixing

        # Update positions
        velocity += acceleration * p.dt
        velocity[velocity < 0] = 0.0
        position += velocity * p.dt
        total_displacement += velocity * p.dt
        position[position > p.L] -= p.L
        # delay = p.scheduled_velocity * t - total_displacement

        if verbose:
            print(bus_motion)
            print(bus_stop_queue)
            print(bus_fullness)
            print("")
        if GRAPH:
            if tstep % 1e2 == 0:
                show_road_system(
                    p,
                    t,
                    green,
                    car,
                    bus,
                    position,
                    traffic_lights,
                    bus_stop_locations,
                    bus_stop_queue,
                    bus_fullness,
                )
        t += p.dt

    mean_velocity = np.mean(total_displacement / t)
    return position, mean_velocity

    # print(f'Optimal scheduling velocity is: {np.mean(total_displacement/t)}')


class params:
    def __init__(self):
        # The road is a single round loop of radius R
        self.L = 1000  # circumference of circle (m)
        # Time marching
        self.t_max = 1e3  # maximum time (s)
        self.dt = 1e-2  # time increment (s)
        # Traffic properties
        self.initial_vehicle_spacing = 100  # (m/vehicle)
        self.speed_limit = 60 / 3.6  # maximum velocity (m/s)
        self.free_flowing_acceleration = 3  # typical vehicle acceleration (m/s^2)
        # Bus system
        self.bus_fraction = 0.1  # what fraction of vehicles are busses (-)
        self.passenger_accumulation_rate = 0.1  # passengers arriving at a stop every second (passengers/s)
        self.passenger_ingress_egress_rate = 1  # how long to get on/off the bus (passengers/s)
        self.bus_max_capacity = 50  # maximum number of passengers on an individual bus (passengers/vehicle)
        self.bus_stop_traffic_light_offset = 0.5  # 0.1ish for just after the traffic lights, 0.9ish for just before traffic lights, 0.5 for in between (-)
        # Traffic light properties
        self.traffic_light_spacing = self.L / 4.0  # (m)
        self.traffic_light_period = 60  # (s)
        self.traffic_light_green_fraction = 0.5  # fraction of time it is _green_ (-)
        self.car_entry_exit_rate = 0.5  #
        # Vehicle interaction properties
        self.stiffness = 1e4  # how much cars repel each other (also used for traffic lights, which are the same as stopped cars)
        self.sigma = 10  # typical stopping distance (m)
        # PTIPS stuff
        self.scheduled_velocity = 0.6 * self.speed_limit  # how fast the busses are scheduled to move (m/s)
        self.ptips_delay_time = 10  # how much delay before PTIPS kicks in (s)
        self.ptips_capacity_threshold = 0.8  # how full should the busses be before ptips kicks in (-)


# single case
p = params()
position, mean_velocity = time_march(p, verbose=False, GRAPH=True)

# parameter study
# p = params()
# vehicle_spacings = np.logspace(1.2, 3, 21)
# vel = []
# for i in vehicle_spacings:
#     p.bus_fraction = 0
#     p.initial_vehicle_spacing = i
#     position, mean_velocity = time_march(p, False, False)
#     vel.append(mean_velocity)
#
# flow_rate = vehicle_spacings ** -1 * vel * 3600  # vehicles/hr = vehicles/m * m/s * s/hr
#
# plt.clf()
# plt.plot(vehicle_spacings ** -1 * 1000, flow_rate)
# plt.xlabel("Density (vehicles/km)")
# plt.ylabel("Flow rate (vehicles/hour)")
# plt.show()
