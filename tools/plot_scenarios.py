import argparse
import os

import dill
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

COLORS = {"home": "#002B5C", "guest": "#008348", "ball": "#FFA500", "past": "red"}
DPI_RESOLUTION = 200
FONT_SIZE = 8


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("scene_image", type=str)
    return parser.parse_args()


def get_color(node_name):
    for node_type, color in COLORS.items():
        if node_type in node_name:
            return color


def plot_image(axis, scene_image):
    court = plt.imread(scene_image)
    axis.imshow(court, zorder=0, extent=[0, 100, 50, 0], alpha=0.7)


def plot_target(target_history, target_future, color, axis):
    c_target = plt.Circle(
        (target_history[-1, 0], target_history[-1, 1]),
        1.2,
        facecolor="r",
        edgecolor=color,
        fill=True,
        zorder=100,
    )
    axis.plot(target_history[:, 0], target_history[:, 1], "-", color=color, linewidth=1)
    axis.plot(target_future[:, 0], target_future[:, 1], "--", color=color, linewidth=1)
    axis.add_patch(c_target)


def plot_neighbors(neighbors_trajectories, axis):
    for neighbor, neighbor_history in neighbors_trajectories.items():
        color = get_color(neighbor)
        cn = plt.Circle(
            (neighbor_history[-1, 0], neighbor_history[-1, 1]),
            1.2,
            edgecolor=color,
            facecolor="m",
            fill=False,
            alpha=1,
        )
        axis.add_patch(cn)
        axis.plot(
            neighbor_history[:, 0],
            neighbor_history[:, 1],
            "-",
            alpha=0.8,
            zorder=100,
            color=color,
            linewidth=1,
        )
        axis.text(
            neighbor_history[-1, 0] + 1,
            neighbor_history[-1, 1] + 1,
            neighbor.split("/")[-1],
            alpha=0.8,
            fontsize=FONT_SIZE,
        )


def get_angle(vector_a, vector_b):
    angle = np.arctan(vector_b / vector_a)
    return 180.0 * angle / np.pi


def plot_distribution(axis, means, covariances, mixture_weights):
    """Plot a heatmap given the predicted GMM distribution parameters."""
    for timestep in range(means.shape[0]):
        for component_index in range(means.shape[1]):
            eigen_values, eigen_vectors = scipy.linalg.eigh(covariances[timestep, component_index])
            eigen_values = 2.0 * np.sqrt(2.0) * np.sqrt(eigen_values)
            eigen_vectors_normalized = eigen_vectors[0] / scipy.linalg.norm(eigen_vectors[0])
            angle = get_angle(eigen_vectors_normalized[0], eigen_vectors_normalized[1])
            ellipse = matplotlib.patches.Ellipse(
                means[timestep, component_index],
                eigen_values[0],
                eigen_values[1],
                180.0 + angle,
                color="blue",
            )
            ellipse.set_edgecolor(None)
            ellipse.set_clip_box(axis.bbox)
            ellipse.set_alpha(mixture_weights[timestep, component_index] / 2)
            axis.add_artist(ellipse)


def plot_shapley_values(shapley_values):
    plt.figure(dpi=DPI_RESOLUTION)
    axis = plt.gca()
    plt_x, plt_height, plt_color = [], [], []
    for node_name, shapley_value in shapley_values.items():
        plt_x.append(node_name.split("/")[-1])
        plt_height.append(shapley_value)
        plt_color.append(get_color(node_name))
    axis.bar(plt_x, plt_height, color=plt_color)
    plt.xticks(fontsize=FONT_SIZE, rotation=90)
    plt.yticks(fontsize=FONT_SIZE)
    plt.show()


def plot_scene(data, node_color, scene_image):
    plt.figure(dpi=DPI_RESOLUTION)
    axis = plt.gca()
    plot_image(axis, scene_image)
    plot_target(data["history"], data["future"], node_color, axis)
    plot_neighbors(data["neighbors"], axis)
    plot_distribution(axis, data["pred_means"], data["pred_covariance"], data["pred_pis"])
    axis.axis("off")
    plt.show()


if __name__ == "__main__":
    parameters = parse_arguments()
    node_type = os.path.basename(parameters.input).split("_")[1]
    node_color = COLORS[node_type]
    with open(parameters.input, "rb") as file_reader:
        data = dill.load(file_reader)
        for node_index, node_data in data.items():
            for timestep, timestep_data in node_data.items():
                plot_scene(timestep_data, node_color, parameters.scene_image)
                plot_shapley_values(timestep_data["shapley_values"])
