# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tester class of Trajectron++ framework.

    Some of the functions are based on the evaluation code under
    thirdparty/Trajectron_plus_plus/experiments/pedestrians/evaluate.py
    of the following repository:
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage."""
import copy
import json
import logging
import os
import pathlib
import random
import time

import dill
import joblib
import model.dataset as dataset
import model.model_registrar
import numpy as np
import tensorboardX
import torch
import tqdm

from explainable_trajectory_prediction.trajectron import network
from explainable_trajectory_prediction.trajectron import utilities

MIN_FUTURE_HORIZON = 12
MIN_HISTORY_LENGTH = 7
NUM_SAMPLES = 20
POSITION_STATE = {"position": ["x", "y"]}


class Tester(object):
    def __init__(self, data, model_path, checkpoint, device):
        self.device = device
        self.load_model(model_path, checkpoint)
        self.load_environment(data)
        self.calculate_scene_graph()

    def load_environment(self, data):
        with open(data, "rb") as f:
            if "sport" in data:
                self.environment = joblib.load(f)
            else:
                self.environment = dill.load(f, encoding="latin1")
            self._override_radius()
            self.trajectron.set_environment(self.environment)
            self.trajectron.set_annealing_params()

    def load_model(self, model_path, checkpoint):
        model_registrar = model.model_registrar.ModelRegistrar(model_path, self.device)
        model_registrar.load_models(checkpoint)
        with open(os.path.join(model_path, "config.json"), "r") as config_json:
            self.hyperparams = json.load(config_json)
        self.trajectron = network.ExplainableTrajectron(
            model_registrar, self.hyperparams, None, self.device
        )

    def _override_radius(self):
        if "override_attention_radius" in self.hyperparams:
            utilities.override_radius_parameters(
                self.environment.attention_radius, self.hyperparams["override_attention_radius"]
            )
        self.environment.neighbors_radius = copy.deepcopy(self.environment.attention_radius)
        if "override_neighbors_radius" in self.hyperparams:
            utilities.override_radius_parameters(
                self.environment.neighbors_radius, self.hyperparams["override_neighbors_radius"]
            )

    def calculate_scene_graph(self):
        for scene in self.environment.scenes:
            scene.calculate_scene_graph(
                self.environment.neighbors_radius,
                self.hyperparams["edge_addition_filter"],
                self.hyperparams["edge_removal_filter"],
            )

    def predict_single_sample(
        self, node_type, scene, timestep, without_neighbours=False, resolution=0.01
    ):
        prior_sampler = network.PriorSampler(latent=None, num_samples=NUM_SAMPLES, full_dist=False)
        return self.trajectron.predict(
            scene,
            timestep,
            self.hyperparams["prediction_horizon"],
            prior_sampler,
            min_future_timesteps=MIN_FUTURE_HORIZON,
            min_history_timesteps=MIN_HISTORY_LENGTH,
            gmm_mode=False,
            resolution=resolution,
            target_node_type=node_type,
            without_neighbors=without_neighbours,
        )

    def get_future(self, node, timestep):
        future = node.get(
            np.array([timestep + 1, timestep + self.hyperparams["prediction_horizon"]]),
            POSITION_STATE,
        )
        return future[~np.isnan(future.sum(axis=1))]

    def get_history(self, node, timestep):
        history = node.get(
            np.array([timestep - self.hyperparams["maximum_history_length"], timestep]),
            POSITION_STATE,
        )
        return history[~np.isnan(history.sum(axis=1))]

    def run(self, without_neighbours=False, resolution=0.01):
        results = {}
        for node_type in self.environment.NodeType:
            logging.info("Testing %s ..." % node_type)
            results[node_type] = {"ade": [], "fde": [], "kde": [], "nll": []}
            for scene_index, scene in enumerate(tqdm.tqdm(self.environment.scenes)):
                for timestep in range(self.hyperparams["maximum_history_length"], scene.timesteps):
                    prediction_dictionary = self.predict_single_sample(
                        node_type,
                        scene,
                        np.arange(timestep, timestep + 1),
                        without_neighbours=without_neighbours,
                        resolution=resolution,
                    )
                    if prediction_dictionary:
                        for node, node_prediction in prediction_dictionary.items():
                            future = self.get_future(node, timestep)
                            node_prediction["samples"] = node_prediction["samples"][
                                :, :, : future.shape[0]
                            ]
                            if node_prediction["samples"].shape[2] == 0:
                                continue
                            results[node_type]["ade"].append(
                                utilities.average_displacement_error(
                                    node_prediction["samples"], future
                                )
                            )
                            results[node_type]["fde"].append(
                                utilities.final_displacement_error(
                                    node_prediction["samples"], future
                                )
                            )
                            results[node_type]["kde"].append(
                                utilities.kernel_density_estimator(
                                    node_prediction["samples"], future
                                )
                            )
                            results[node_type]["nll"].append(
                                utilities.nll_gaussian_mixture_model(
                                    node_prediction["GMM"],
                                    torch.tensor(future).to(self.device),
                                    self.hyperparams["log_p_yt_xz_max"],
                                )
                            )
            for error_metric, metric_values in results[node_type].items():
                logging.info("%s: %.2f" % (error_metric, np.mean(metric_values)))
        return results
