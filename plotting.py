import numpy as np
import matplotlib.pyplot as plt


def show_road_system(
    p, t, green, car, bus, position, traffic_lights, bus_stop_locations, bus_stop_queue, bus_fullness
):
    R = p.L / (2 * np.pi)  # circumference (m)
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
    plt.plot(
        R * np.sin(traffic_light_theta[green]),
        R * np.cos(traffic_light_theta[green]),
        "o",
        mec="g",
        mfc="None",
        markersize=10,
        label="Traffic light",
    )
    plt.plot(
        R * np.sin(traffic_light_theta[~green]),
        R * np.cos(traffic_light_theta[~green]),
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
        np.square(p.bus_max_capacity) / bus_stop_scaling,
        marker="o",
        facecolors="none",
        edgecolors="b",
        label="Bus",
    )
    plt.scatter(
        R * np.sin(theta[bus]),
        R * np.cos(theta[bus]),
        np.square(np.sum(bus_fullness, axis=1)) / bus_stop_scaling,
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
