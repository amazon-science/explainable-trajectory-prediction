"""Process the SportVU text files into pickle object to be passed to the PECNet framework.

    For every dataset split, this script reads all processed log files, creates
    a numpy object for the trajectories of all scenes."""

import glob
import os
import pickle
import random

import numpy as np
import pandas as pd
import tqdm

DATASET_SPLITS = ["train", "test"]
BATCH_SPLITS_SIZE = [181, 1]
MAXIMUM_NUMBER_OF_TEST_GAMES = 100
NUMBER_OF_PLAYERS = 11
MINIMUM_SCENE_LENGTH = 20


def select_split_files(files, dataset_split):
    """Select a maximum number of files for test and all files for train.

    Args:
        files: A list of files to select from.
        dataset_split: The dataset split name (train, test).

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
        raise ValueError(f"unexpected dataset split {dataset_split}.")


def check_validity_of_node(data_frame, scene_max_timestep):
    """Return if the node data is valid or not."""
    if not np.all(np.diff(data_frame["frame_id"]) == 1):
        return False
    new_first_idx = data_frame["frame_id"].iloc[0]
    new_last_idx = data_frame["frame_id"].iloc[-1]
    if new_first_idx or new_last_idx != scene_max_timestep:
        return False
    return True


def load_game_data(file):
    """Load the data of the whole game (stored in a file)."""
    data = pd.read_csv(file, sep="\t", index_col=False, header=None)
    data.columns = ["frame_id", "track_id", "pos_x", "pos_y", "node_type"]
    data = data.drop(["node_type"], axis=1)
    data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
    data.sort_values("frame_id", inplace=True)
    scene_max_timestep = data["frame_id"].max()
    if scene_max_timestep < MINIMUM_SCENE_LENGTH - 1:
        return None
    result = []
    for track_id in pd.unique(data["track_id"]):
        node_data_frame = data[data["track_id"] == track_id]
        if not check_validity_of_node(node_data_frame, scene_max_timestep):
            continue
        result.append(node_data_frame[["pos_x", "pos_y"]].values)
    if len(result) != NUMBER_OF_PLAYERS:
        return None
    return np.stack(result, axis=0)


def get_all_samples(data):
    """Get the trajectories of all agents in a game."""
    game_timesteps = data.shape[1]
    samples = []
    for i in range(game_timesteps - MINIMUM_SCENE_LENGTH - 1):
        samples.append(data[:, i : i + MINIMUM_SCENE_LENGTH, :])
    return np.stack(samples, axis=0)


if __name__ == "__main__":
    dataset_path = "../data/sport_vu"
    for index, dataset_split in enumerate(DATASET_SPLITS):
        all_files = glob.glob(os.path.join(dataset_path, dataset_split) + "/*.txt")
        selected_files = select_split_files(all_files, dataset_split)
        samples = []
        for file in tqdm.tqdm(selected_files):
            data = load_game_data(file)
            if data is None:
                continue
            samples.append(get_all_samples(data))

        random.seed(0)
        random.shuffle(samples)
        trajectory_data = np.concatenate(samples, axis=0)
        batches = np.split(
            trajectory_data, trajectory_data.shape[0] / BATCH_SPLITS_SIZE[index], axis=0
        )
        batches = [
            x.reshape(BATCH_SPLITS_SIZE[index] * NUMBER_OF_PLAYERS, MINIMUM_SCENE_LENGTH, 2)
            for x in batches
        ]
        with open(
            "sport_%s_all_%d_0_100.pkl"
            % (dataset_split, BATCH_SPLITS_SIZE[index] * NUMBER_OF_PLAYERS),
            "wb",
        ) as f:
            pickle.dump([batches], f)
