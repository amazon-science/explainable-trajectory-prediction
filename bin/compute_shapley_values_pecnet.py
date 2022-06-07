# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os

import dill

from explainable_trajectory_prediction.pecnet import dataset
from explainable_trajectory_prediction.pecnet import shapley_values
from explainable_trajectory_prediction.pecnet import test
from explainable_trajectory_prediction.trajectron import utilities


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_file", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("scene_index", help="scene index", type=int)
    parser.add_argument("output_path", help="result directory", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parameters = parse_arguments()
    utilities.initialize_device_and_seed(0)
    tester = test.Tester(utilities.get_device(), parameters.load_file)
    if "sport" in parameters.dataset_path:
        get_data = dataset.get_sport_data
        get_indices = dataset.get_indices_no_mask
        get_random_neighbor = dataset.get_random_neighbor_no_mask
    else:
        get_data = dataset.get_sdd_data
        get_indices = dataset.get_indices_mask
        get_random_neighbor = None
    shapley_values_estimator = shapley_values.ShapleyValues(
        tester, get_indices, get_random_neighbor
    )
    output_file_path = parameters.output_path + "_agent_%d_ade.pkl" % (parameters.scene_index)
    if os.path.exists(output_file_path):
        raise FileExistsError(f"Output file already exists {output_file_path}.")
    result = shapley_values_estimator.run(parameters.dataset_path, parameters.scene_index, get_data)
    with open(output_file_path, "wb") as file_writer:
        dill.dump(result, file_writer)
