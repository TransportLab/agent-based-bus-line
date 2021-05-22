'''
Questions to ask of model:
  - How does PTIPS behave when the _schedule_ is bad?

'''

import numpy as np
import matplotlib.pyplot as plt

def gaussian(d,sigma):
    return np.exp(-d**2/2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

# The road is a single round loop of radius R
R = 100 # radius (m)
L = 2*np.pi*R # circumference (m)

# Time marching
t = 0 # current time (s)
tstep = 0 # time step (-)
t_max = 1e4 # maximum time (s)
dt = 1e-2 # time increment (s) - NOTE: COULD BE WAY SMALLER, JUST USED FOR ANIMATION NOW

# Traffic properties
speed_limit = 60/3.6 # maximum velocity (m/s)
free_flowing_acceleration = 1 # typical vehicle acceleration (m/s^2)
gamma = free_flowing_acceleration/speed_limit**2 # drag coefficient
initial_vehicle_spacing = 50 # (m)
num_vehicles = int(L//initial_vehicle_spacing)
bus_fraction = 0.2
num_traffic_lights = 5
scheduled_velocity = 0.5*speed_limit # how fast the busses are scheduled to move (m/s)

# Traffic light properties
traffic_light_period = 60 # (s)
traffic_light_green_fraction = 0.5 # fraction of time it is _green_
bus_waiting_time = 10 # how long a bus waits at a stop (s)
ptips_delay_time = 10 # how much delay before PTIPS kicks in (s)
# Vehicle interaction properties
stiffness = 1e4 # how much cars repel each other (also used for traffic lights, which are the same as stopped cars)
sigma = 10 # typical stopping distance (m)

# Initialise system
position = np.linspace(0,L,num_vehicles,endpoint=False)
# position += vehicle_spacing*np.random.rand(num_vehicles)
velocity = np.zeros_like(position)
traffic_lights = np.linspace(0,L,num_traffic_lights,endpoint=False)
bus_stops = traffic_lights.copy() + (traffic_lights[1] - traffic_lights[0])/2.
# traffic_lights = np.array([])

bus = np.random.choice(num_vehicles,int(bus_fraction*num_vehicles),replace=False)
wait_time = -1*np.ones_like(position) # how long the bus has been waiting at the stop, negative values indicates the last stop number
car = np.delete(range(num_vehicles),bus)

while t < t_max:
    # Everyone loves to accelerate
    acceleration = free_flowing_acceleration - gamma*velocity**2
    # acceleration[0] = 0 # fix first car

    # Check car in front
    relative_distance = np.roll(position,-1) - position
    relative_distance[relative_distance < 0] = L + relative_distance[relative_distance < 0] # account for periodicity
    interaction = stiffness*gaussian(relative_distance,sigma)
    interaction[velocity<=0] = 0
    acceleration -= interaction


    # Check traffic lights
    if t%traffic_light_period < traffic_light_green_fraction*traffic_light_period: green = True
    else: green = False

    if not green:
        for light in traffic_lights:
            distance_to_light = light - position
            distance_to_light[distance_to_light > L/2] = L - distance_to_light[distance_to_light > L/2] # account for periodicity
            for i in range(num_vehicles):
                if distance_to_light[i] < 3*sigma and distance_to_light[i] > 0 and velocity[i] > 0:
                    # print(i,position[i],distance_to_light[i])
                    if i in b and delay[b] > ptips_delay_time:
                        pass
                    else:
                        acceleration[i] -= stiffness*gaussian(distance_to_light[i],sigma)

    # Check bus stops
    for b in bus:
        for stop in bus_stops:
            distance_to_stop = stop - position[b]
            if distance_to_stop > L/2: distance_to_stop = L - distance_to_stop # account for periodicity
            if distance_to_stop < 3*sigma and distance_to_stop > 0 and velocity[b] > 0:
                if wait_time[b] != -stop:
                    if wait_time[b] < 0:
                        wait_time[b] = dt
                        acceleration[b] -= stiffness*gaussian(distance_to_stop,sigma)
                    elif wait_time[b] < bus_waiting_time:
                        wait_time[b] += dt
                        acceleration[b] -= stiffness*gaussian(distance_to_stop,sigma)
                    else:
                        wait_time[b] = -stop



    velocity += acceleration*dt
    velocity[velocity<0] = 0.
    position += velocity*dt
    position[position > L] -= L

    theta = position/L*2*np.pi
    traffic_light_theta = traffic_lights/L*2*np.pi
    bus_stop_theta = bus_stops/L*2*np.pi

    delay = position - scheduled_velocity*t

    if tstep%100 == 0:
        plt.ion()
        plt.clf()
        plt.plot(L*np.sin(np.linspace(0,2*np.pi,101)),L*np.cos(np.linspace(0,2*np.pi,101)),'k--',label='Road')
        if green: plt.plot(L*np.sin(traffic_light_theta),L*np.cos(traffic_light_theta),'o',mec='g',mfc='None',markersize=10,label='Traffic light')
        else:     plt.plot(L*np.sin(traffic_light_theta),L*np.cos(traffic_light_theta),'o',mec='r',mfc='None',markersize=10,label='Traffic light')
        plt.plot(L*np.sin(bus_stop_theta),L*np.cos(bus_stop_theta),'o',mec='b',mfc='None',markersize=10,label='Bus stop')
        # plt.plot(L*np.sin(theta),L*np.cos(theta),'ko',label='Vehicle')
        plt.plot(L*np.sin(theta[car]),L*np.cos(theta[car]),'ko',label='Car')
        plt.plot(L*np.sin(theta[bus]),L*np.cos(theta[bus]),'bo',label='Bus')
        plt.xlim(-1.2*L,1.2*L)
        plt.ylim(-1.2*L,1.2*L)
        plt.axis('equal')
        plt.legend(loc=0,fancybox=True)
        plt.pause(1e-6)
    t += dt
    tstep += 1
