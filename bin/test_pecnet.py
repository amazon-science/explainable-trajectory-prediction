# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging

from explainable_trajectory_prediction.pecnet import dataset
from explainable_trajectory_prediction.pecnet import test
from explainable_trajectory_prediction.trajectron import utilities


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_file", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--number_of_runs", type=int, default=1)
    parser.add_argument("--without_neighbours", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utilities.initialize_device_and_seed(0)
    parameters = parse_arguments()
    tester = test.Tester(utilities.get_device(), parameters.load_file)
    if "sport" in parameters.dataset_path:
        get_data = dataset.get_sport_data
    else:
        get_data = dataset.get_sdd_data
    tester.run(
        parameters.dataset_path, parameters.number_of_runs, get_data, parameters.without_neighbours
    )
