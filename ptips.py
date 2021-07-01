"""
Questions to ask of model:
  - How does PTIPS behave when the _schedule_ is bad?

"""

from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt


def gaussian(d, sigma):
    return np.exp(-(d ** 2) / 2 / sigma ** 2) / (np.sqrt(2 * np.pi) * sigma)


def time_march(p, verbose, GRAPH):
    # Initialise system
    t = 0  # current time (s)
    tstep = 0  # time step (-)
    nt = int(p.t_max // p.dt)
    gamma = p.free_flowing_acceleration / (p.speed_limit ** 2)  # drag coefficient
    R = p.L / (2 * np.pi)  # circumference (m)
    num_vehicles = int(p.L // p.initial_vehicle_spacing)
    num_traffic_lights = int(p.L // p.traffic_light_spacing)

    position = np.linspace(0, p.L, num_vehicles, endpoint=False)
    # position += vehicle_spacing*np.random.rand(num_vehicles)
    total_displacement = np.zeros_like(position)
    velocity = np.zeros_like(position)
    traffic_lights = np.linspace(0, p.L, num_traffic_lights, endpoint=False)
    bus_stop_locations = traffic_lights.copy() + (traffic_lights[1] - traffic_lights[0]) / 2.0
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
        if t % p.traffic_light_period < p.traffic_light_green_fraction * p.traffic_light_period:
            green = True
        else:
            green = False

        if not green:
            for light in traffic_lights:
                distance_to_light = light - position
                distance_to_light[position>light] += p.L # account for periodicity
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
                    if ((bus_stop_queue[i] > 1 and np.sum(bus_fullness[j,:])<p.bus_max_capacity) or bus_fullness[j, i] > 1) and bus_motion[
                        j
                    ] == 0:  # bus is moving and hits a stop with at least one person or one person wants to get off
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
            bus_fullness[bus_fullness < 0] = 0  # HACK!!! real issue is not checking if we are going to over-empty the bus in a given timestep, but this is way too small an issue to bother fixing

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
                theta = position / p.L * 2 * np.pi
                traffic_light_theta = traffic_lights / p.L * 2 * np.pi
                bus_stop_theta = bus_stop_locations / p.L * 2 * np.pi
                bus_stop_scaling = 5

                plt.ion()
                plt.clf()
                plt.suptitle(t)
                plt.plot(
                    R * np.sin(np.linspace(0, 2 * np.pi, 101)),
                    R * np.cos(np.linspace(0, 2 * np.pi, 101)),
                    "k--",
                    label="Road",
                )
                if green:
                    plt.plot(
                        R * np.sin(traffic_light_theta),
                        R * np.cos(traffic_light_theta),
                        "o",
                        mec="g",
                        mfc="None",
                        markersize=10,
                        label="Traffic light",
                    )
                else:
                    plt.plot(
                        R * np.sin(traffic_light_theta),
                        R * np.cos(traffic_light_theta),
                        "o",
                        mec="r",
                        mfc="None",
                        markersize=10,
                        label="Traffic light",
                    )
                plt.scatter(
                    R * np.sin(bus_stop_theta),
                    R * np.cos(bus_stop_theta),
                    np.square(bus_stop_queue),
                    marker="o",
                    facecolors="none",
                    edgecolors="b",
                    label="Bus stop",
                )
                plt.plot(
                    R * np.sin(theta[car]), R * np.cos(theta[car]), "ko", label="Car",
                )
                plt.scatter(
                    R * np.sin(theta[bus]),
                    R * np.cos(theta[bus]),
                    np.square(p.bus_max_capacity)/bus_stop_scaling,
                    marker="o",
                    facecolors="none",
                    edgecolors="b",
                    label="Bus",
                )
                plt.scatter(
                    R * np.sin(theta[bus]),
                    R * np.cos(theta[bus]),
                    np.square(np.sum(bus_fullness, axis=1))/bus_stop_scaling,
                    marker="o",
                    facecolors="b",
                    edgecolors="none",
                    # label="Bus",
                )
                plt.xlim(-1.2 * R, 1.2 * R)
                plt.ylim(-1.2 * R, 1.2 * R)
                plt.axis("equal")
                plt.legend(loc=0, fancybox=True)
                plt.pause(1e-6)
        t += p.dt

    mean_velocity = np.mean(total_displacement / t)
    return position, mean_velocity

    # print(f'Optimal scheduling velocity is: {np.mean(total_displacement/t)}')


verbose = False

GRAPH = True
# GRAPH = False


class params:
    def __init__(self):
        # The road is a single round loop of radius R
        self.L = 1000  # circumference of circle (m)
        # Time marching
        self.t_max = 1e4  # maximum time (s)
        self.dt = 1e-2  # time increment (s)
        # Traffic properties
        self.initial_vehicle_spacing = 100  # (m/vehicle)
        self.speed_limit = 60 / 3.6  # maximum velocity (m/s)
        self.free_flowing_acceleration = 3  # typical vehicle acceleration (m/s^2)
        # Bus system
        self.bus_fraction = 0.1
        self.passenger_accumulation_rate = 0.1  # passengers arriving at a stop every second (passengers/s)
        self.passenger_ingress_egress_rate = 1  # how long to get on/off the bus (passengers/s)
        self.bus_max_capacity = 50  # maximum number of passengers on an individual bus
        # Traffic light properties
        self.traffic_light_spacing = self.L / 4.0  # (m)
        self.traffic_light_period = 60  # (s)
        self.traffic_light_green_fraction = 0.5  # fraction of time it is _green_
        # Vehicle interaction properties
        self.stiffness = 1e4  # how much cars repel each other (also used for traffic lights, which are the same as stopped cars)
        self.sigma = 10  # typical stopping distance (m)
        # PTIPS stuff
        self.scheduled_velocity = 0.6 * self.speed_limit  # how fast the busses are scheduled to move (m/s)
        self.ptips_delay_time = 10  # how much delay before PTIPS kicks in (s)
        self.ptips_capacity_threshold = 0.8  # how full should the busses be before ptips kicks in (-)


p = params()
# vehicle_spacings = np.logspace(1.2,3,21)
# vel = []
# for initial_vehicle_spacing in vehicle_spacings:
position, mean_velocity = time_march(p, verbose, GRAPH)

# flow_rate = vehicle_spacings ** -1 * vel * 3600  # vehicles/hr = vehicles/m * m/s * s/hr
#
# plt.clf()
# plt.plot(vehicle_spacings ** -1 * 1000, flow_rate)
# plt.xlabel("Density (vehicles/km)")
# plt.ylabel("Flow rate (vehicles/hour)")
# plt.show()
