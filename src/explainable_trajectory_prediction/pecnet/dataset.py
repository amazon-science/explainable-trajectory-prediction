# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset loader.

    Some of the functions are based on the dataset loader under
    thirdparty/PECNet/utils/social_utils.py
    of the following repository:
    https://github.com/HarshayuGirase/PECNet,
    see LICENSE under thirdparty/PECNet/LICENSE for usage."""

import numpy as np
import random
import torch


def scale_and_convert_to_relative_trajectory(trajectory, data_scale):
    return (trajectory - trajectory[:, :, :1, :]) * data_scale


def get_initial_position_scaled(trajectory):
    return trajectory[:, :, 7, :].copy() / 1000


def get_sport_data(data, data_scale, device):
    trajectory = np.stack(data)
    initial_pos = get_initial_position_scaled(trajectory)
    trajectory = scale_and_convert_to_relative_trajectory(trajectory, data_scale)
    return (
        torch.DoubleTensor(trajectory).to(device),
        torch.DoubleTensor(initial_pos).to(device),
        torch.ones(trajectory.shape[0], trajectory.shape[1], trajectory.shape[1]).to(device),
    )


def get_sdd_data(data, data_scale, device):
    trajectory, mask = data
    trajectory = np.array(trajectory)[:, :, :, 2:]
    mask = np.array(mask)
    initial_pos = get_initial_position_scaled(trajectory)
    trajectory = scale_and_convert_to_relative_trajectory(trajectory, data_scale)
    return (
        torch.DoubleTensor(trajectory).to(device),
        torch.DoubleTensor(initial_pos).to(device),
        torch.DoubleTensor(mask).to(device),
    )


def get_indices_mask(batch_size, mask, index, device):
    valid_indices = mask[index].nonzero(as_tuple=True)[0]
    random_neighbor = random.choice([x for x in range(batch_size) if x not in valid_indices])
    neighbors_indices = valid_indices[valid_indices != index]
    return torch.cat(
        [
            torch.tensor([index], device=device),
            neighbors_indices,
            torch.tensor([random_neighbor], device=device),
        ],
        dim=0,
    )


def get_indices_no_mask(batch_size, _mask, index, device):
    valid_indices = torch.tensor([x for x in range(batch_size)])
    neighbors_indices = valid_indices[valid_indices != index]
    return torch.cat([torch.tensor([index]), neighbors_indices], dim=0).to(device)


def get_random_neighbor_no_mask(trajectories, initial_positions, scene_index):
    random_scene_index = random.choice(
        [x for x in range(trajectories.shape[0]) if x != scene_index]
    )
    random_batch_trajectory = trajectories[random_scene_index]
    random_batch_initial_pos = initial_positions[random_scene_index]
    random_player_index = random.choice([x for x in range(random_batch_trajectory.shape[0])])
    return (
        random_batch_trajectory[random_player_index : random_player_index + 1],
        random_batch_initial_pos[random_player_index : random_player_index + 1],
    )
