"""
Questions to ask of model:
  - How does PTIPS behave when the _schedule_ is bad?

"""


import os
import sys
import json5
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from plotting import show_road_system

class dict_to_class(dict):
    def __init__(self, dict):
        lists = []
        for key in dict:
            setattr(self, key, dict[key])
            if isinstance(dict[key], list):
                lists.append(key)
        setattr(self, 'lists', lists)


def update_order(p, vehicle_position):
    vehicle_order = []
    vehicle_order_order = []
    for i in range(p.lanes):
        vehicle_order.append(np.argsort(vehicle_position[i]))
        vehicle_order_order.append(np.argsort(vehicle_order[i]))
    return vehicle_order, vehicle_order_order


def gaussian(d, sigma):
    return np.exp(-(d ** 2) / 2 / sigma ** 2) / (np.sqrt(2 * np.pi) * sigma)


def time_march(p):
    # Initialise system
    t = 0  # current time (s)
    tstep = 0  # time step (-)
    nt = int(p.t_max // p.dt) + 1
    # gamma = p.free_flowing_acceleration / (p.speed_limit ** 2)  # drag coefficient --- FOUND ISSUE ON 28/10/2022 --- WHY WAS THIS SQUARED?!?
    gamma = p.free_flowing_acceleration / p.speed_limit  # drag coefficient
    initial_num_vehicles_per_lane = int(p.L // p.initial_vehicle_spacing)
    # num_vehicles = p.lanes * initial_num_vehicles_per_lane

    # Instantiate things that have multiple lanes
    vehicle_position = []
    total_displacement = []
    velocity = []
    acceleration = []
    relative_distance = []
    for i in range(p.lanes):
        vehicle_position.append(np.linspace(0, p.L, initial_num_vehicles_per_lane, endpoint=False))
        total_displacement.append(np.zeros(initial_num_vehicles_per_lane))
        velocity.append(np.zeros(initial_num_vehicles_per_lane))
        acceleration.append(np.zeros(initial_num_vehicles_per_lane))
        relative_distance.append(np.zeros(initial_num_vehicles_per_lane))

    traffic_lights = np.linspace(0, p.L, p.num_traffic_lights, endpoint=False)
    traffic_light_phasing = np.linspace(0, 2 * np.pi, p.num_traffic_lights, endpoint=False)

    if p.num_traffic_lights > 0:
        bus_stop_locations = traffic_lights.copy() + p.bus_stop_traffic_light_offset * (
            traffic_lights[1] - traffic_lights[0]
        )
    else: bus_stop_locations = np.array([])
    bus_stop_queue = np.zeros_like(bus_stop_locations)  # no passengers anywhere
    # bus = np.random.choice(
    # initial_num_vehicles_per_lane, int(p.bus_fraction * initial_num_vehicles_per_lane), replace=False
    # )
    # bus = np.arange(int(p.bus_fraction * initial_num_vehicles_per_lane))  # first vehicles are busses
    bus = np.arange(1)
    np.random.shuffle(vehicle_position[0])  # but they can be anywhere physically
    bus_occupancy = np.zeros(
        [len(bus), len(bus_stop_locations)]
    )  # for each bus, a list of passengers by destination
    bus_motion = np.zeros_like(bus, dtype=int)  # 0=moving, 1=unloading, 2=loading

    vehicle_order, vehicle_order_order = update_order(p, vehicle_position)
    update_order_flag = False

    if not os.path.exists(p.output_folder):
        os.makedirs(p.output_folder)

    if p.verbose:
        traffic_light_spacing = traffic_lights[1] - traffic_lights[0]

        theoretical_bus_occupancy = (
            p.passenger_accumulation_rate
            * traffic_light_spacing
            * (p.num_traffic_lights ** 2)
            / p.scheduled_velocity
        )
        print(f'Theoretical bus fullness: {theoretical_bus_occupancy}. Bus max capacity: {p.bus_max_capacity}')
        print('\n' + str(len(bus)) + '\n')

    for tstep in tqdm(range(nt), leave=False, desc="Time"):
        # if tstep%1000 == 0: print(bus_occupancy.sum())
        # Check traffic lights
        green = (
            0.5 * (np.cos(2 * np.pi * t / p.traffic_light_period + traffic_light_phasing) + 1)
            < p.traffic_light_green_fraction
        )

        for lane in range(p.lanes):
            if update_order_flag:
                vehicle_order, vehicle_order_order = update_order(p, vehicle_position)
                update_order_flag = False
            # Everyone loves to accelerate
            acceleration[lane] = p.free_flowing_acceleration - gamma * velocity[lane]

            # Check car in front
            relative_distance[lane] = (
                vehicle_position[lane][np.roll(vehicle_order[lane], -1)]
                - vehicle_position[lane][vehicle_order[lane]]
            )[
                vehicle_order_order[lane]
            ]  # magic! never ask questions

            relative_distance[lane][relative_distance[lane] < 0] = (
                p.L + relative_distance[lane][relative_distance[lane] < 0]
            )  # account for periodicity

            interaction = p.stiffness * gaussian(relative_distance[lane], p.sigma)
            interaction[velocity[lane] <= 0] = 0
            acceleration[lane] -= interaction

            for i, light in enumerate(traffic_lights):
                distance_to_light = light - vehicle_position[lane]
                distance_to_light[vehicle_position[lane] > light] += p.L  # account for periodicity
                if green[i]:  # remove some cars as they go through
                    in_intersection = np.nonzero(
                        (distance_to_light > -(velocity[lane] * p.dt))
                        * (distance_to_light < (velocity[lane] * p.dt))
                    )[0]
                    for j in in_intersection:
                        # print(f'\nvehicle {j} moving through light {i}')
                        if j not in bus:  # just cars
                            if np.random.rand() < p.car_entry_exit_probability:
                                choices = np.nonzero(~green)[0]
                                # if len(choices) == 1:
                                # red_light = choices[0]
                                # else:
                                red_light = np.random.choice(choices, 1)[0]
                                if p.verbose:
                                    print(f"all red lights: {choices}")
                                    print(f"{len(choices)}")
                                    print(f"chose {red_light}")
                                vehicle_position[lane][j] = (
                                    traffic_lights[red_light] + np.random.rand() * p.sigma
                                )  # move the car to another red light
                                velocity[lane][j] = 0
                                # now updated order of vehicles so collisions still work
                                update_order_flag = True

                else:  # cars stop at red lights
                    stopping_vehicles = (distance_to_light < 3 * p.sigma) * (distance_to_light > 0)
                    acceleration[lane][stopping_vehicles] -= (
                        10 * p.stiffness * gaussian(distance_to_light[stopping_vehicles], p.sigma)
                    )

        # Update passengers at stops
        bus_stop_queue += p.passenger_accumulation_rate * p.dt

        # Check bus stops - NOTE: BUSSES ONLY IN 0-th LANE!!!
        for j, b in enumerate(bus):
            for i, stop in enumerate(bus_stop_locations):
                if vehicle_position[0][b] > stop:
                    distance_to_stop = stop - vehicle_position[0][b] + p.L
                else:
                    distance_to_stop = stop - vehicle_position[0][b]
                if distance_to_stop < 3 * p.sigma and distance_to_stop > 0:
                    if (
                        (
                            (bus_stop_queue[i] > 1 and np.sum(bus_occupancy[j, :]) < p.bus_max_capacity)
                            or bus_occupancy[j, i] > 1
                        )
                        and bus_motion[j] == 0
                    ):  # bus is moving and hits a stop with at least one person or one person wants to get off
                        bus_motion[j] = 1  # move to unloading phase
                    if bus_motion[j] > 0:
                        acceleration[0][b] -= p.stiffness * gaussian(distance_to_stop, p.sigma)
                    if velocity[0][b] == 0.0:  # only once the bus has stopped
                        if bus_motion[j] == 1:  # unloading
                            # print(f'unloading {j} at stop {i}. Current fullness {bus_occupancy[j,i]}')
                            if (
                                bus_occupancy[j, i] > 0
                            ):  # if there are passengers on this bus who want to get off here
                                bus_occupancy[j, i] -= (
                                    p.passenger_ingress_egress_rate * p.dt
                                )  # passengers leave bus
                            else:
                                bus_motion[j] = 2
                        elif bus_motion[j] == 2:  # loading
                            # print(f'LOADING {j} at stop {i}. Current fullness {bus_occupancy[j,i]}')
                            if bus_stop_queue[i] > 0 and np.sum(bus_occupancy[j, :]) <= p.bus_max_capacity:
                                bus_occupancy[j, :i] += (
                                    p.passenger_ingress_egress_rate * p.dt / len(bus_stop_queue - 1)
                                )  # passengers go onto bus
                                bus_occupancy[j, i + 1 :] += (
                                    p.passenger_ingress_egress_rate * p.dt / len(bus_stop_queue - 1)
                                )  # passengers go onto bus
                                bus_stop_queue[i] -= (
                                    p.passenger_ingress_egress_rate * p.dt
                                )  # passengers leave stop
                            else:
                                bus_motion[j] = 0  # start moving again
            bus_occupancy[
                bus_occupancy < 0
            ] = 0  # HACK!!! real issue is not checking if we are going to over-empty the bus in a given timestep, but this is way too small an issue to bother fixing

        # Lane changing
        for lane in range(p.lanes):
            # distance to all cars in lane to the left
            if lane == 0:
                left_distances = np.zeros_like(vehicle_position[lane])  # can't go into this imaginary lane
            else:
                left_distances = cdist(
                    vehicle_position[lane].reshape(len(vehicle_position[lane]), -1),
                    vehicle_position[lane - 1].reshape(len(vehicle_position[lane - 1]), -1),
                )
            if lane == p.lanes - 1:
                right_distances = np.zeros_like(vehicle_position[lane])  # can't go into this imaginary lane
            else:
                right_distances = cdist(
                    vehicle_position[lane].reshape(len(vehicle_position[lane]), -1),
                    vehicle_position[lane + 1].reshape(len(vehicle_position[lane + 1]), -1),
                )

            deleted = []  # keep track of vehicles deleted in this lane
            for i in range(len(vehicle_position[lane])):
                moves = []  # currently cant move anywhere
                if i not in bus:
                    if np.all(left_distances[i] > 2 * p.sigma):
                        moves.append(-1)
                    if np.all(right_distances[i] > 2 * p.sigma):
                        moves.append(1)
                    # if not accelerating
                    if (acceleration[lane][i] < 0) and (len(moves) > 0):
                        direction = np.random.choice(moves, 1)[0]
                        # print(lane, i)  # ,vehicle_position[lane+1].shape,vehicle_position[lane].shape)
                        vehicle_position[lane + direction] = np.append(
                            vehicle_position[lane + direction], vehicle_position[lane][i]
                        )
                        velocity[lane + direction] = np.append(velocity[lane + direction], velocity[lane][i])
                        acceleration[lane + direction] = np.append(
                            acceleration[lane + direction], acceleration[lane][i]
                        )
                        total_displacement[lane + direction] = np.append(
                            total_displacement[lane + direction], total_displacement[lane][i]
                        )
                        deleted.insert(0, i)  # add to start of list
            for i in deleted:
                vehicle_position[lane] = np.delete(vehicle_position[lane], i)
                velocity[lane] = np.delete(velocity[lane], i)
                acceleration[lane] = np.delete(acceleration[lane], i)
                total_displacement[lane] = np.delete(total_displacement[lane], i)
                update_order_flag = True

        # Update positions
        for lane in range(p.lanes):
            velocity[lane] += acceleration[lane] * p.dt
            velocity[lane][velocity[lane] < 0] = 0.0
            vehicle_position[lane] += velocity[lane] * p.dt
            total_displacement[lane] += velocity[lane] * p.dt
            vehicle_position[lane][vehicle_position[lane] > p.L] -= p.L
        # delay = p.scheduled_velocity * t - total_displacement
        
        if p.verbose:
            print(bus_motion)
            print(bus_stop_queue)
            print(bus_occupancy)
            print("")
        if p.GRAPH:
            if tstep % 1e2 == 0:
                show_road_system(
                    p,
                    t,
                    green,
                    bus,
                    vehicle_position,
                    traffic_lights,
                    bus_stop_locations,
                    bus_stop_queue,
                    bus_occupancy,
                )
        t += p.dt

    total = 0
    N = 0
    for lane in range(p.lanes):
        total += np.sum(total_displacement[lane])
        N += len(total_displacement[lane])
    mean_vehicle_velocity = total / N / p.t_max

    total = 0
    for j, b in enumerate(bus):
        total += total_displacement[0][b]
    mean_bus_velocity = total / len(bus) / p.t_max

    mean_bus_occupancy = bus_occupancy.sum() / len(bus)



    np.save(f'{p.output_folder}/vehicle_positions.npy', vehicle_position)
    np.save(f'{p.output_folder}/mean_vehicle_velocity.npy', mean_vehicle_velocity)
    np.save(f'{p.output_folder}/mean_bus_velocity.npy', mean_bus_velocity)
    np.save(f'{p.output_folder}/mean_bus_occupancy.npy', mean_bus_occupancy)

    # return vehicle_position, mean_velocity, 

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as params:       
        # parse file
        dict = json5.loads(params.read())
        dict['input_filename'] = (sys.argv[1].split('/')[-1]).split('.')[0]
        p_init = dict_to_class(dict)

        for i in tqdm(p_init.passenger_accumulation_rate, desc="Pass accum", disable=(len(p_init.passenger_accumulation_rate) == 1)):
            for j in tqdm(p_init.initial_vehicle_spacing, desc="Veh spacing", leave=False, disable=(len(p_init.initial_vehicle_spacing) == 1)):
                p = dict_to_class(dict)
                p.passenger_accumulation_rate = i
                p.initial_vehicle_spacing = j
                p.output_folder = f'output/passenger_accumulation_rate_{i}/initial_vehicle_spacing_{j}'
                time_march(p)

