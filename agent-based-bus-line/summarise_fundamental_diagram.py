import json5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use("transport-findings.mplstyle")

with open('json/fundamental_diagram.json', 'r') as params:       
    p = json5.loads(params.read())

fig, ax = plt.subplots(nrows=1, ncols=2)
for i, r_pa in enumerate(p["passenger_accumulation_rate"]):
    vel = []
    bus_occupancy = []
    bus_vel = []
    for j in p["initial_vehicle_spacing"]:
        output_folder = f'output/passenger_accumulation_rate_{r_pa}/initial_vehicle_spacing_{j}'
        mean_vehicle_velocity = np.load(f'{output_folder}/mean_vehicle_velocity.npy')
        # std_vehicle_velocity = np.load(f'{output_folder}/std_vehicle_velocity.npy')
        mean_bus_velocity = np.load(f'{output_folder}/mean_bus_velocity.npy')
        mean_bus_occupancy = np.load(f'{output_folder}/mean_bus_occupancy.npy')
        vel.append(mean_vehicle_velocity)
        bus_occupancy.append(mean_bus_occupancy)
        bus_vel.append(mean_bus_velocity)
    
    flow_rate = 1./np.array(p["initial_vehicle_spacing"]) * vel * 3600 / p["lanes"] # vehicles/hr/lane = vehicles/m * m/s * s/hr / lanes
    vehicle_density = 1./np.array(p["initial_vehicle_spacing"]) * 1000 # vehicles/km
    linear_density = 2*p["sigma"]/np.array(p["initial_vehicle_spacing"])
    # ax[0].plot(vehicle_density, flow_rate, label=f"$r_{pa}={i}$")
    bus_fullness = np.array(bus_occupancy)/p["bus_max_capacity"]

    c=cm.plasma(i / len(p["passenger_accumulation_rate"]))
    ax[0].plot(linear_density, flow_rate, ls='-', c=c, label="$r_{pa}="+str(r_pa)+"$")
    ax[1].plot(bus_vel, bus_fullness, ls='None', marker='.', c=c, label="$r_{pa}="+str(r_pa)+"$")

ax[0].set_xlabel("Density (vehicles/km)")
# ax[0].set_xlabel("Linear density (-)")
ax[0].set_xlim(0,1)
# ax[0].set_ylabel("Flow rate (vehicles/hour/lane)")
ax[0].set_ylabel('Average velocity (m/s)')
plt.sca(ax[0])
plt.legend(loc=0)

ax[1].set_xlabel("Bus velocity (m/s)")
# ax[1].set_xlabel("Linear density (-)")
# ax[1].set_xlabel("Flow rate (vehicles/hour)")
# ax[1].set_ylabel("Bus occupancy (passengers/bus)")
ax[1].set_ylabel("Bus fullness (-)")

plt.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.945, wspace=0.25, hspace=0.75)
plt.savefig(f"fundamental_diagram.png", dpi=200)