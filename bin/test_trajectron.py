# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging

from explainable_trajectory_prediction.trajectron import test
from explainable_trajectory_prediction.trajectron import utilities


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model full path", type=str)
    parser.add_argument("checkpoint", help="checkpoint number", type=int)
    parser.add_argument("data", help="full path to data file", type=str)
    parser.add_argument("device", help="gpu or cpu", type=str)
    parser.add_argument("--without_neighbours", action="store_true")
    parser.add_argument("--resolution", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utilities.initialize_device_and_seed(0)
    parameters = parse_arguments()
    tester = test.Tester(
        parameters.data, parameters.model, parameters.checkpoint, parameters.device
    )
    tester.run(parameters.without_neighbours, parameters.resolution)
