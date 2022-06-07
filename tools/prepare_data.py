"""Pre-process the sport VU dataset.

    Log files of the data are downloaded from
    https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs"""
import glob
import logging
import os
import random
import shutil

import numpy as np
import pandas as pd
import py7zr
from tqdm import tqdm

MINIMUM_DURATION = 9.0
SAMPLING_RATIO = 10
MAXIMUM_NUMBER_OF_GAMES = 6000
TRAIN_SPLIT_RATIO = 0.7
VALID_SPLIT_RATIO = 0.3


def extract_logfiles(path: str):
    """Extract all log files.

    Args:
        path: Relative path where the log files are stored.
    """
    files = os.listdir(path)
    for file in tqdm(files):
        with py7zr.SevenZipFile(os.path.join(path, file), mode="r") as zipfile:
            zipfile.extractall(path=path)


def convert_json_to_text(path: str):
    """Convert json files to text format similar to ETH-UCY dataset.

    Every line in the text file correspond to a player at specific time-step
    Format: frame_index, object_id, object_position_x, object_position_y, object_type

        Args:
            path: Relative path where the json files are stored.
    """
    game_files = glob.glob(path + "/*.json")
    random.seed(0)
    random.shuffle(game_files)
    number_of_processed_games = 0
    for game_file in tqdm(game_files):
        data_frame = pd.read_json(game_file)

        if number_of_processed_games >= MAXIMUM_NUMBER_OF_GAMES:
            break
        for event in data_frame["events"]:
            home_players = event["home"]["players"]
            guest_players = event["visitor"]["players"]
            all_players = home_players + guest_players
            player_ids = [player["playerid"] for player in all_players]
            player_names = [
                " ".join([player["firstname"], player["lastname"]]) for player in all_players
            ]
            player_jerseys = [player["jersey"] for player in all_players]
            player_names_jerseys = list(zip(player_names, player_jerseys))
            player_ids_dict = dict(zip(player_ids, player_names_jerseys))
            home_players_ids = [player["playerid"] for player in home_players]

            if len(event["moments"]) <= 1:
                continue
            duration = event["moments"][0][2] - event["moments"][-1][2]
            if duration < MINIMUM_DURATION:
                continue

            text_lines = []
            included_players = []
            for moment_index, moment in enumerate(event["moments"]):
                if moment_index % SAMPLING_RATIO != 0:
                    continue
                ball = moment[5][0]
                text_lines.append(
                    "\t".join(
                        [
                            "%d" % int(moment_index / SAMPLING_RATIO),
                            "-1",
                            "%.2f" % ball[2],
                            "%.2f" % ball[3],
                            "ball",
                        ]
                    )
                    + "\n"
                )
                players = moment[5][1:]
                for player in players:
                    team = "home" if player[1] in home_players_ids else "guest"
                    team_abb = "H" if player[1] in home_players_ids else "G"
                    player_jersey = player_ids_dict[player[1]][1]
                    if not player_jersey:
                        continue
                    text_lines.append(
                        "\t".join(
                            [
                                "%d" % int(moment_index / SAMPLING_RATIO),
                                player_jersey + team_abb,
                                "%.2f" % player[2],
                                "%.2f" % player[3],
                                team,
                            ]
                        )
                        + "\n"
                    )
                    included_players.append(player_ids_dict[player[1]][1] + team_abb)

            # Reject games with less or more than 10 players.
            if len(np.unique(included_players)) != 10:
                continue
            with open(game_file.replace(".json", "-%s.txt" % event["eventId"]), "w") as f_writer:
                f_writer.writelines(text_lines)
            number_of_processed_games += 1


def make_splits(path: str):
    """Split the processed text files into train/valid/test.

    Args:
        path: Relative path where the json files are stored.

    Raises:
        FileExistsError: If the splits directories exist.
    """
    files = glob.glob(path + "/*.txt")
    number_of_files = len(files)
    random.shuffle(files)
    train_size = int(number_of_files * TRAIN_SPLIT_RATIO)
    valid_size = int((number_of_files - train_size) * VALID_SPLIT_RATIO)
    scenes = {
        "train": files[:train_size],
        "valid": files[train_size : train_size + valid_size],
        "test": files[train_size + valid_size :],
    }
    for split, split_scenes in scenes.items():
        split_path = os.path.join(path, "../%s" % split)
        if os.path.exists(split_path):
            raise FileExistsError(f"Could not create directory {split_path}, it already exists.")
        os.mkdir(split_path)
        for file in tqdm(split_scenes):
            shutil.move(file, os.path.join(split_path, os.path.basename(file)))


if __name__ == "__main__":
    logs_path = "../data/sport_vu/logs"

    logging.info("Extracting files...")
    extract_logfiles(logs_path)

    logging.info("Converting files...")
    convert_json_to_text(logs_path)

    logging.info("Making splits ...")
    make_splits(logs_path)
