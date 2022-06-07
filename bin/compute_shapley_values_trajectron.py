# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os

import dill

from explainable_trajectory_prediction.trajectron import shapley_values
from explainable_trajectory_prediction.trajectron import test
from explainable_trajectory_prediction.trajectron import utilities


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model full path", type=str)
    parser.add_argument("checkpoint", help="checkpoint number", type=int)
    parser.add_argument("data", help="full path to data file", type=str)
    parser.add_argument("device", help="gpu or cpu", type=str)
    parser.add_argument("node_type", help="node type", type=str)
    parser.add_argument(
        "metric", choices=shapley_values.METRICS.values(), type=shapley_values.create_metric
    )
    parser.add_argument(
        "variant", choices=shapley_values.VARIANTS.values(), type=shapley_values.create_variant
    )
    parser.add_argument("scene_index", help="scene index", type=int)
    parser.add_argument("output_path", help="result directory", type=str)
    parser.add_argument(
        "--random_node_types", help="list of random node types", type=str, nargs="+"
    )
    parser.add_argument("--store_visualization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utilities.initialize_device_and_seed(0)
    parameters = parse_arguments()
    tester = test.Tester(
        parameters.data, parameters.model, parameters.checkpoint, parameters.device
    )
    shapley_values_estimator = shapley_values.ShapleyValues(
        tester, parameters.node_type, parameters.random_node_types, parameters.metric
    )
    output_file_path = parameters.output_path + "_%s_%d_%s.pkl" % (
        parameters.node_type,
        parameters.scene_index,
        parameters.metric,
    )
    if os.path.exists(output_file_path):
        raise FileExistsError(f"Output file already exists {output_file_path}.")
    get_replacement_trajectory = lambda edge_type: parameters.variant(
        tester.environment.scenes,
        shapley_values_estimator.data_parameters,
        tester.environment,
        edge_type,
    )
    test_scene = tester.environment.scenes[parameters.scene_index]
    result = shapley_values_estimator.run(
        test_scene, get_replacement_trajectory, store_visualization=parameters.store_visualization
    )
    with open(output_file_path, "wb") as file_writer:
        dill.dump(result, file_writer)
