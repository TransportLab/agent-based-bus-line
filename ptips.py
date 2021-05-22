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
dt = 1e-2 # time increment (s)
v_max = 60/3.6 # maximum velocity (m/s)
gravity = 1 # typical vehicle acceleration (m/s^2)
mass = 1 # mass of vehicles (nominal)
gamma = gravity/v_max**2 # drag coefficient

traffic_light_period = 60 # (s)
traffic_light_green_fraction = 0.5 # fraction of time it is _green_

stiffness = 1e4
sigma = 10 # (m)
vehicle_spacing = 50 # (m)
num_vehicles = int(L//vehicle_spacing)
bus_fraction = 0.5
position = np.linspace(0,L,num_vehicles,endpoint=False)
# position += vehicle_spacing*np.random.rand(num_vehicles)
velocity = np.zeros_like(position)

traffic_lights = np.linspace(0,L,5,endpoint=False)
# traffic_lights = np.array([])

bus = np.random.choice(num_vehicles,int(bus_fraction*num_vehicles),replace=False)
print(bus)

while t < t_max:
    # Everyone loves to accelerate
    acceleration = gravity - gamma*velocity**2
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
                    acceleration[i] -= stiffness*gaussian(distance_to_light[i],sigma)



    velocity += acceleration*dt
    velocity[velocity<0] = 0.
    position += velocity*dt
    position[position > L] -= L

    theta = position/L*2*np.pi
    traffic_light_theta = traffic_lights/L*2*np.pi

    if tstep%100 == 0:
        plt.ion()
        plt.clf()
        plt.plot(L*np.sin(np.linspace(0,2*np.pi,101)),L*np.cos(np.linspace(0,2*np.pi,101)),'k--',label='Road')
        if green: plt.plot(L*np.sin(traffic_light_theta),L*np.cos(traffic_light_theta),'go',markersize=10,label='Traffic light')
        else:     plt.plot(L*np.sin(traffic_light_theta),L*np.cos(traffic_light_theta),'ro',markersize=10,label='Traffic light')
        plt.plot(L*np.sin(theta),L*np.cos(theta),'ko',label='Vehicle')
        # plt.plot(L*np.sin(theta[~bus]),L*np.cos(theta[~bus]),'ko',label='Car')
        # plt.plot(L*np.sin(theta[ bus]),L*np.cos(theta[ bus]),'bo',label='Bus')
        plt.xlim(-1.2*L,1.2*L)
        plt.ylim(-1.2*L,1.2*L)
        plt.axis('equal')
        plt.legend(loc=0,fancybox=True)
        plt.pause(1e-6)
    t += dt
    tstep += 1
