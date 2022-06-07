# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json

from explainable_trajectory_prediction.trajectron import train


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config file for hyperparameters", type=str)
    parser.add_argument(
        "--device", help="what device to perform training on", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--preprocess_workers",
        help="number of processes to spawn for preprocessing",
        type=int,
        default=0,
    )
    parser.add_argument("--data_dir", help="directory containing the training data", type=str)
    parser.add_argument("--train_data_dict", help="name of the training data pickle file", type=str)
    parser.add_argument("--log_dir", help="directory to save trained models", type=str)
    parser.add_argument(
        "--log_tag",
        help="tag for this experiment model",
        type=str,
    )
    parser.add_argument(
        "--train_epochs", help="number of iterations to train for", type=int, default=1
    )
    parser.add_argument(
        "--save_every", help="how often to save during training", type=int, default=10
    )
    parser.add_argument("--seed", help="manual seed to use, default is 123", type=int, default=123)

    # model parameters
    parser.add_argument(
        "--offline_scene_graph", help="precompute the scene graphs offline", type=str, default="yes"
    )
    parser.add_argument(
        "--augment", help="whether to augment the scene during training", action="store_true"
    )
    parser.add_argument("--batch_size", help="training batch size", type=int, default=256)
    parser.add_argument(
        "--deeper_action", help="apply deeper action state decoding.", action="store_true"
    )
    parser.add_argument(
        "--late_fusion", help="apply late fusion on the neighbors", action="store_true"
    )
    parser.add_argument(
        "--added_random_player", help="add a random player during training", action="store_true"
    )
    return parser.parse_args()


def override_parameters(parameters, args):
    parameters["batch_size"] = args.batch_size
    parameters["offline_scene_graph"] = args.offline_scene_graph
    parameters["augment"] = args.augment
    parameters["late_fusion"] = args.late_fusion
    parameters["deeper_action"] = args.deeper_action


if __name__ == "__main__":
    parameters = parse_arguments()
    with open(parameters.config, "r", encoding="utf-8") as config_json:
        model_parameters = json.load(config_json)
    override_parameters(model_parameters, parameters)
    trainer = train.Trainer(train_parameters, model_parameters)
    trainer.fit()
