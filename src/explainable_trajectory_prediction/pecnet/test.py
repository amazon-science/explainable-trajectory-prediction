# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tester class of the PECNet framework.

    Some of the functions are based on the testing code under
    thirdparty/PECNet/scripts/test_pretrained_model.py
    of the following repository:
    https://github.com/HarshayuGirase/PECNet,
    see LICENSE under thirdparty/PECNet/LICENSE for usage."""
import copy
import logging
import pickle
import sys

import models
import numpy as np
import social_utils
import torch
import tqdm
import yaml

from explainable_trajectory_prediction.trajectron import utilities

NUM_SAMPLES = 20


class Tester(object):
    def __init__(self, device, model_path):
        self.device = device
        self.load_model(model_path)

    def load_model(self, file_path):
        """Create and load the model from the checkpoint file."""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.hyper_parameters = checkpoint["hyper_params"]
        self.model = models.PECNet(
            self.hyper_parameters["enc_past_size"],
            self.hyper_parameters["enc_dest_size"],
            self.hyper_parameters["enc_latent_size"],
            self.hyper_parameters["dec_size"],
            self.hyper_parameters["predictor_hidden_size"],
            self.hyper_parameters["non_local_theta_size"],
            self.hyper_parameters["non_local_phi_size"],
            self.hyper_parameters["non_local_g_size"],
            self.hyper_parameters["fdim"],
            self.hyper_parameters["zdim"],
            self.hyper_parameters["nonlocal_pools"],
            self.hyper_parameters["non_local_dim"],
            self.hyper_parameters["sigma"],
            self.hyper_parameters["past_length"],
            self.hyper_parameters["future_length"],
            False,
        )
        self.model = self.model.double().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def load_data(self, dataset_path, get_data):
        """Load the testing data from the specified path."""
        with open(dataset_path, "rb") as file_reader:
            data = pickle.load(file_reader)
            return get_data(data, self.hyper_parameters["data_scale"], self.device)

    def predict(self, input, mask, initial_pos, output):
        """Predict the future of a batch and returns the errors.

        Args:
            input: The past trajectory of the agents.
            mask: A binary mask denoting the neighborhood.
            initial_pos: The absolute initial position of every agent in 2D.
            output: The ground truth future trajectory.
        """
        x = input.contiguous().view(-1, input.shape[1] * input.shape[2]).to(self.device)
        all_predicted_futures = []
        for _ in range(NUM_SAMPLES):
            predicted_destination = self.model.forward(x, initial_pos, device=self.device)
            interpolated_future = self.model.predict(x, predicted_destination, mask, initial_pos)
            all_predicted_futures.append(
                torch.cat([interpolated_future, predicted_destination], dim=1).reshape(
                    -1, self.hyper_parameters["future_length"], 2
                )
            )
        predicted_futures = torch.stack(all_predicted_futures, dim=0).cpu().numpy()
        min_ade = (
            utilities.average_displacement_error(predicted_futures, output, squeeze=False)
            / self.hyper_parameters["data_scale"]
        )
        min_fde = (
            utilities.final_displacement_error(
                predicted_futures, output.transpose(1, 0, 2), squeeze=False
            )
            / self.hyper_parameters["data_scale"]
        )
        return min_ade, min_fde

    def run(self, dataset_path, number_of_runs, get_data, without_neighbours=False):
        """Test the model on a dataset.

        Args:
            dataset_path: The dataset file path.
            number_of_runs: number of runs to repeat the test and report the average.
            get_data: A function that returns the input/output data.
            without_neighbours: Keep or remove neighbours when testing.
        """
        with torch.no_grad():
            min_ades = []
            min_fdes = []
            with torch.no_grad():
                trajectories, initial_positions, masks = self.load_data(dataset_path, get_data)
                for scene_index in tqdm.tqdm(range(trajectories.shape[0])):
                    batch_trajectory = trajectories[scene_index]
                    batch_initial_pos = initial_positions[scene_index]
                    batch_mask = masks[scene_index]
                    if without_neighbours:
                        batch_mask = torch.eye(
                            batch_mask.shape[0], dtype=batch_mask.dtype, device=batch_mask.device
                        )
                    input = batch_trajectory[:, : self.hyper_parameters["past_length"], :]
                    output = (
                        batch_trajectory[:, self.hyper_parameters["past_length"] :, :].cpu().numpy()
                    )
                    ade_runs = []
                    fde_runs = []
                    for _ in range(number_of_runs):
                        batch_ade, batch_fde = self.predict(
                            input, batch_mask, batch_initial_pos, output
                        )
                        ade_runs.append(np.mean(batch_ade))
                        fde_runs.append(np.mean(batch_fde))
                    min_ades.append(np.mean(ade_runs))
                    min_fdes.append(np.mean(fde_runs))
        average_min_ade = np.mean(min_ades)
        average_min_fde = np.mean(min_fdes)
        logging.info("ade: %.2f" % average_min_ade)
        logging.info("fde: %.2f" % average_min_fde)
        return average_min_ade, average_min_fde
