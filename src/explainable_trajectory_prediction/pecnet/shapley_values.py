# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import numpy as np
import random
import torch
import tqdm

NUM_ITERATION = 10


def create_positive_negative_pair_history(input, mask, device):
    random_masks = []
    positive_input = copy.deepcopy(input)
    negative_input = copy.deepcopy(input)
    negative_input[0, ...] = 0
    for _ in range(NUM_ITERATION):
        random_mask = torch.randint(2, mask.shape).to(device)
        random_mask[:, 0] = 1
        random_masks.append(random_mask)
    return positive_input, negative_input, random_masks


def create_positive_negative_pair_neighbor(input, mask, device, neighbor_index):
    inputs = []
    positive_masks = []
    negative_masks = []
    for _ in range(NUM_ITERATION):
        input_copy = copy.deepcopy(input)
        if random.randint(0, 1) == 1:
            input_copy[0, ...] = 0
        inputs.append(input_copy)
        random_mask = torch.randint(2, mask.shape).to(device)
        random_mask[:, 0] = 1
        random_mask_positive = copy.deepcopy(random_mask)
        random_mask_positive[:, neighbor_index] = 1
        random_mask_negative = copy.deepcopy(random_mask)
        random_mask_negative[:, neighbor_index] = 0
        positive_masks.append(random_mask_positive)
        negative_masks.append(random_mask_negative)
    return inputs, positive_masks, negative_masks


class ShapleyValues:
    def __init__(self, tester, get_indices, get_random_neighbor):
        self.tester = tester
        self.get_indices = get_indices
        self.get_random_neighbor = get_random_neighbor

    def get_shapley_values(self, positive_batch, negative_batch, output):
        error_positive, _ = self.tester.predict(
            positive_batch[0], positive_batch[1], positive_batch[2], output
        )
        error_negative, _ = self.tester.predict(
            negative_batch[0], negative_batch[1], negative_batch[2], output
        )
        return error_negative[0] - error_positive[0]

    def get_scene(self, trajectories, initial_positions, masks, scene_index):
        batch_trajectory = trajectories[scene_index]
        batch_initial_pos = initial_positions[scene_index]
        if self.get_random_neighbor is not None:
            random_trajectory, random_initial_pos = self.get_random_neighbor(
                trajectories, initial_positions, scene_index
            )
            batch_trajectory = torch.cat([batch_trajectory, random_trajectory], dim=0)
            batch_initial_pos = torch.cat([batch_initial_pos, random_initial_pos], dim=0)
        return batch_trajectory, batch_initial_pos, masks[scene_index]

    def get_sample(self, trajectory, mask, initial_pos, index):
        indices = self.get_indices(trajectory.shape[0], mask, index, device=self.tester.device)
        sample_mask = torch.ones(indices.shape[0], indices.shape[0], device=self.tester.device)
        input = trajectory[indices][:, : self.tester.hyper_parameters["past_length"], :]
        output = (
            trajectory[indices][0:1, self.tester.hyper_parameters["past_length"] :, :].cpu().numpy()
        )
        return input, sample_mask, initial_pos[indices], output

    def run(self, dataset_path, scene_index, get_data):
        results = {}
        with torch.no_grad():
            trajectories, initial_positions, masks = self.tester.load_data(dataset_path, get_data)
            batch_trajectory, batch_initial_pos, batch_mask = self.get_scene(
                trajectories, initial_positions, masks, scene_index
            )
            for index in tqdm.tqdm(range(batch_trajectory.shape[0])):
                results[index] = {}
                input, sample_mask, sample_initial_pos, output = self.get_sample(
                    batch_trajectory, batch_mask, batch_initial_pos, index
                )
                (
                    input_positive,
                    input_negative,
                    random_masks,
                ) = create_positive_negative_pair_history(input, sample_mask, self.tester.device)
                errors = []
                for sample_index in range(len(random_masks)):
                    errors.append(
                        self.get_shapley_values(
                            (input_positive, random_masks[sample_index], sample_initial_pos),
                            (input_negative, random_masks[sample_index], sample_initial_pos),
                            output,
                        )
                    )
                results[index]["past"] = np.mean(errors)
                for neighbor_index in range(1, sample_mask.shape[0]):
                    inputs, positive_masks, negative_masks = create_positive_negative_pair_neighbor(
                        input, sample_mask, self.tester.device, neighbor_index
                    )
                    errors = []
                    for sample_index in range(len(inputs)):
                        errors.append(
                            self.get_shapley_values(
                                (
                                    inputs[sample_index],
                                    positive_masks[sample_index],
                                    sample_initial_pos,
                                ),
                                (
                                    inputs[sample_index],
                                    negative_masks[sample_index],
                                    sample_initial_pos,
                                ),
                                output,
                            )
                        )
                    name = (
                        "%d" % neighbor_index
                        if neighbor_index != (sample_mask.shape[0] - 1)
                        else "random"
                    )
                    results[index][name] = np.mean(errors)
        return results
