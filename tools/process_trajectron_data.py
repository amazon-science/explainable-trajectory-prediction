"""Process the SportVU text files into pickle objects.

    This code is based on the Trajectron++ repository at
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage.
    For every dataset split, this script reads all processed log files, creates
    a scene for every file, and stores all scenes of an environment in one pickle file."""
import glob
import logging
import os
import pdb
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from thirdparty.Trajectron_plus_plus.trajectron.environment import Environment, Node, Scene

MAXIMUM_NUMBER_OF_TEST_GAMES = 100
DEFAULT_FRAME_DIFFERENCE_DURATION = 0.4
ATTENTION_RADIUS = 0.3
SCENE_NUMBER_OF_NODES = 11
RANDOM_ROTATION_INTERVAL = 180
DATASET_SPLITS = ["train", "valid", "test"]
NODE_TYPES = ["home", "guest", "ball"]
NODE_STATES = ["position", "velocity", "acceleration"]
STATE_COMPONENTS = ["x", "y"]
STANDARDIZATION_MEAN = 0.0
STANDARDIZATION_STD = {
    "position": 1.0,
    "velocity": 2.0,
    "acceleration": 3.0,
}


def derivative_of_trajectory(trajectory: np.array, frame_difference_duration: float) -> np.array:
    """Compute the derivative of a trajectory along its dimension.

    Args:
        trajectory: 1D array containing either x or y positions over time.
        frame_difference_duration: the duration in seconds between two consecutive frames.

    Returns:
        First order derivative of the input array or zero array if input array has less than two non-nan elements.
    """
    not_nan_mask = ~np.isnan(trajectory)
    masked_trajectory = trajectory[not_nan_mask]

    # return zero derivative if the input trajectory has less than two valid elements.
    if masked_trajectory.shape[-1] < 2:
        return np.zeros_like(trajectory)

    derivative_trajectory = np.full_like(trajectory, np.nan)
    derivative_trajectory[not_nan_mask] = (
        np.ediff1d(masked_trajectory, to_begin=(masked_trajectory[1] - masked_trajectory[0]))
        / frame_difference_duration
    )

    return derivative_trajectory


def prepare_standardization_parameters() -> dict:
    """Prepare a dictionary for the standardization parameters.

    Returns:
        The dictionary containing all parameters for standardizing the environment.
    """
    standardization = dict()
    for node_type in NODE_TYPES:
        standardization[node_type] = dict()
        for node_state in NODE_STATES:
            standardization[node_type][node_state] = dict()
            for component in STATE_COMPONENTS:
                standardization[node_type][node_state][component] = {
                    "mean": STANDARDIZATION_MEAN,
                    "std": STANDARDIZATION_STD[node_state],
                }
    return standardization


def rotate_point(point: np.array, radian_angle: float) -> np.array:
    """Rotate point (x,y) using alpha.

    Args:
        point: 2D array for the x,y components of a point.
        radian_angle: An angle (in radians) to rotate the point.

    Returns:
        2D array for the rotated point.
    """

    rotation_matrix = np.array(
        [
            [np.cos(radian_angle), -np.sin(radian_angle)],
            [np.sin(radian_angle), np.cos(radian_angle)],
        ]
    )
    return rotation_matrix @ point


def compute_velocity_acceleration(
    trajectory: np.array, frame_difference_duration: float
) -> (np.array, np.array):
    """Compute the velocity and acceleration given a single trajectory.

    Args:
        trajectory: 1D array containing either x or y positions over time.
        frame_difference_duration: the duration in seconds between two consecutive frames.

    Returns:
        (velocity, acceleration): both has same type and shape as the input trajectory.
    """
    velocity = derivative_of_trajectory(trajectory, frame_difference_duration)
    acceleration = derivative_of_trajectory(velocity, frame_difference_duration)
    return velocity, acceleration


def augment_scene(scene: Scene, angle: float) -> Scene:
    """Augment the given scene by randomly rotate it and all its objects.

    Args:
        scene: The scene object to be augmented.
        angle: An angle (in degrees) to rotate the scene.

    Returns:
        A copy of the scene that is augmented (rotated).
    """
    data_columns = pd.MultiIndex.from_product([NODE_STATES, STATE_COMPONENTS])
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)
    radian_angle = angle * np.pi / 180.0

    for node in scene.nodes:
        trajectory_x = node.data.position.x.copy()
        trajectory_y = node.data.position.y.copy()
        trajectory_x, trajectory_y = rotate_point(
            np.array([trajectory_x, trajectory_y]), radian_angle
        )

        velocity_x, acceleration_x = compute_velocity_acceleration(trajectory_x, scene.dt)
        velocity_y, acceleration_y = compute_velocity_acceleration(trajectory_y, scene.dt)

        data_dictionary = {
            ("position", "x"): trajectory_x,
            ("position", "y"): trajectory_y,
            ("velocity", "x"): velocity_x,
            ("velocity", "y"): velocity_y,
            ("acceleration", "x"): acceleration_x,
            ("acceleration", "y"): acceleration_y,
        }

        node_data = pd.DataFrame(data_dictionary, columns=data_columns)
        node = Node(
            node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep
        )
        scene_aug.nodes.append(node)
    return scene_aug


def create_rotated_versions_of_scene(scene: Scene) -> list():
    """Create a list of rotated versions of a scene.

    Args:
        scene: the scene object.

    Returns:
        list of rotated versions of the scene.
    """
    rotated_scenes = list()
    angles = np.arange(0, 360, RANDOM_ROTATION_INTERVAL)
    for angle in angles:
        rotated_scenes.append(augment_scene(scene, angle))
    return rotated_scenes


def choose_augmented_scene_at_random(scene: Scene) -> Scene:
    """Choose randomly from the set of augmented scenes.

    Every scene has a list of augmented scenes, rotated versions of the scene.
    We select randomly one of those versions during training.
    This function will be stored in the object to be called during training.

       Args:
           scene: The scene object to select one of its augmented versions.

       Returns:
           A link to an augmented version of the scene.
    """
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


def create_result_directory(path: str):
    """Create a directory for the processed pickle files.

    Args:
        path: The parent path where the results folder to be created.

    Returns:
        The created directory path.
    """
    result_directory_path = os.path.join(path, "processed")
    if os.path.exists(result_directory_path):
        raise FileExistsError(
            f"Could not create directory {result_directory_path}, it already exists."
        )
    os.mkdir(result_directory_path)
    return result_directory_path


def select_split_files(files: list(), dataset_split: str) -> list():
    """Select the first set of files for either test/valid and all files for train.

    The number of files to be selected for test/valid is determined by a threshold.

       Args:
           files: A list of files to select from.
           dataset_split: The dataset split name (train, valid, test).

       Returns:
           A list of files.

       Raises:
            ValueError: If the dataset_split is unexpected.
    """
    if dataset_split == "train":
        return files
    elif dataset_split in ["test", "valid"]:
        return files[:MAXIMUM_NUMBER_OF_TEST_GAMES]
    else:
        raise ValueError(
            f"Could not select files from an unexpected dataset split {dataset_split}."
        )


def generate_node(
    node_id: str,
    node_data_frame: pd.DataFrame,
    scene_max_timestep: int,
    scene_frame_difference_duration: float,
    mean_position: np.array,
) -> Node:
    """Generate a node for a specific object.

    Args:
        node_id: The unique id of the node.
        node_data_frame: The input data to parse.
        scene_max_timestep: The maximum timestep of the scene.
        scene_frame_difference_duration: The duration in second between frames.
        mean_position: The average mean position on x-y.

    Returns:
        The generated node object containing the trajectory state over time or None.
    """
    # ignore the node if it has missing data for some frames.
    if not np.all(np.diff(node_data_frame["frame_id"]) == 1):
        return None

    node_values = node_data_frame[["pos_x", "pos_y"]].values
    node_type = node_data_frame["node_type"].values[0]
    new_first_idx = node_data_frame["frame_id"].iloc[0]
    new_last_idx = node_data_frame["frame_id"].iloc[-1]

    # ignore the node if its trajectory is not present for the whole scene.
    if new_first_idx or new_last_idx != scene_max_timestep:
        return None

    trajectory_x = node_values[:, 0]
    trajectory_y = node_values[:, 1]
    velocity_x, acceleration_x = compute_velocity_acceleration(
        trajectory_x, scene_frame_difference_duration
    )
    velocity_y, acceleration_y = compute_velocity_acceleration(
        trajectory_y, scene_frame_difference_duration
    )

    data_dictionary = {
        ("position", "x"): trajectory_x,
        ("position", "y"): trajectory_y,
        ("velocity", "x"): velocity_x,
        ("velocity", "y"): velocity_y,
        ("acceleration", "x"): acceleration_x,
        ("acceleration", "y"): acceleration_y,
        ("mean", "x"): mean_position[0],
        ("mean", "y"): mean_position[1],
    }

    node_data_columns = pd.MultiIndex.from_product([NODE_STATES + ["mean"], STATE_COMPONENTS])
    node_data = pd.DataFrame(data_dictionary, columns=node_data_columns)
    node = Node(node_type=env_node_types[node_type], node_id=node_id, data=node_data)
    node.first_timestep = new_first_idx
    return node


def generate_scene(file_path: str, scene_name: str) -> Scene:
    """Generate a scene for every input file.

    The generated scene has multiple nodes, where each node represents
    a player in the game.

       Args:
           file_path: The path to the text file to parse.
           scene_name: A name of the scene object.

       Returns:
           The generated scene object or None if the scene has less than a
           specific number of objects.
    """
    data = pd.read_csv(file_path, sep="\t", index_col=False, header=None)
    data.columns = ["frame_id", "track_id", "pos_x", "pos_y", "node_type"]
    data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
    data["node_id"] = data["track_id"].astype(str)
    data.sort_values("frame_id", inplace=True)

    mean_x = data["pos_x"].mean()
    mean_y = data["pos_y"].mean()
    data["pos_x"] = data["pos_x"] - mean_x
    data["pos_y"] = data["pos_y"] - mean_y

    scene_max_timestep = data["frame_id"].max()
    scene = Scene(
        timesteps=scene_max_timestep + 1,
        dt=DEFAULT_FRAME_DIFFERENCE_DURATION,
        name=scene_name,
        aug_func=choose_augmented_scene_at_random if scene_name == "train" else None,
    )

    for node_id in pd.unique(data["node_id"]):
        node_data_frame = data[data["node_id"] == node_id]
        node = generate_node(
            node_id, node_data_frame, scene_max_timestep, scene.dt, np.array([mean_x, mean_y])
        )
        if node:
            scene.nodes.append(node)

    if len(scene.nodes) != SCENE_NUMBER_OF_NODES:
        return None

    if dataset_split == "train":
        scene.augmented = create_rotated_versions_of_scene(scene)
    return scene


if __name__ == "__main__":
    attention_radius = dict()
    for source_edge_type in NODE_TYPES:
        for destination_edge_type in NODE_TYPES:
            attention_radius[(source_edge_type, destination_edge_type)] = ATTENTION_RADIUS

    standardization_parameters = prepare_standardization_parameters()

    dataset_path = "../data/sport_vu"
    result_directory_path = create_result_directory(dataset_path)

    for dataset_split in DATASET_SPLITS:
        env = Environment(node_type_list=NODE_TYPES, standardization=standardization_parameters)
        env.attention_radius = attention_radius
        env_node_types = {
            "home": env.NodeType.home,
            "guest": env.NodeType.guest,
            "ball": env.NodeType.ball,
        }

        all_files = glob.glob(os.path.join(dataset_path, dataset_split) + "/*.txt")
        selected_files = select_split_files(all_files, dataset_split)
        scenes = []
        for file in tqdm(selected_files):
            scene = generate_scene(file, scene_name=dataset_split)
            if scene:
                scenes.append(scene)

        if scenes:
            env.scenes = scenes
            with open(
                os.path.join(result_directory_path, dataset_split + ".pkl"), "wb"
            ) as file_writer:
                joblib.dump(env, file_writer)
            logging.info(f"Store environment: ({dataset_split}, number of scenes: {len(scenes)}).")
