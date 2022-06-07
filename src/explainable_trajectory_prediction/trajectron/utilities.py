# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions.

    Some of the functions are based on the evaluation code under
    thirdparty/Trajectron_plus_plus/trajectron/evaluation
    of the following repository:
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage."""
import random

import numpy as np
import scipy
import torch

from explainable_trajectory_prediction.trajectron import network


def average_displacement_error(predicted_trajs, gt_traj, squeeze=True):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    if squeeze:
        return np.min(ade)
    return np.min(ade, axis=0)


def final_displacement_error(predicted_trajs, gt_traj, squeeze=True):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    if squeeze:
        return np.min(final_error)
    return np.min(final_error, axis=0)


def kernel_density_estimator(predicted_trajs, gt_traj):
    kde_ll = 0.0
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[1]
    for batch_index in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = scipy.stats.gaussian_kde(predicted_trajs[:, batch_index, timestep].T)
                pdf = np.clip(
                    kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None
                )[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                return np.nan
    return -kde_ll


def nll_gaussian_mixture_model(predicted_distribution, gt_traj, max_log_value, reduce_mean=True):
    log_probs = torch.clamp(predicted_distribution.log_prob(gt_traj), max=max_log_value)
    if not reduce_mean:
        return -torch.mean(log_probs, dim=-1).cpu().detach().numpy()
    return -torch.mean(log_probs).item()


def get_gmm_parameters(gmm, mean_position):
    def _to_numpy(x):
        return x.squeeze().detach().cpu().numpy()

    return (
        _to_numpy(gmm.mus) + mean_position,
        _to_numpy(gmm.get_covariance_matrix()),
        _to_numpy(gmm.pis_cat_dist.probs),
    )


def is_padded_neighbor(neighbor):
    return "padded" in str(neighbor)


def is_random_neighbor(neighbor):
    return "random" in str(neighbor)


def initialize_device_and_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return device


def override_radius_parameters(environment_radius, radius_parameters):
    for radius_parameter in radius_parameters:
        node_type1, node_type2, radius = radius_parameter.split(" ")
        environment_radius[(node_type1, node_type2)] = float(radius)


class ADE:
    """The average displacement error."""

    def __init__(self, squeeze=False, prior_sampler=None):
        self.prior_sampler = network.PriorSampler(
            latent=None,
            num_samples=20,
            full_dist=False,
        )
        if prior_sampler:
            self.prior_sampler = prior_sampler
        self.squeeze = squeeze

    def __call__(self, _predicted_gmm, predicted_samples, future, _max_log):
        return average_displacement_error(predicted_samples, future, squeeze=self.squeeze)

    def __repr__(self):
        return "ade"


class NLL:
    """The negative log likelihood."""

    def __init__(self, reduce_mean=False, prior_sampler=None):
        self.prior_sampler = network.PriorSampler(
            latent=None,
            num_samples=1,
            full_dist=True,
        )
        if prior_sampler:
            self.prior_sampler = prior_sampler
        self.reduce_mean = reduce_mean

    def __call__(self, predicted_gmm, _predicted_samples, future, max_log):
        return nll_gaussian_mixture_model(
            predicted_gmm,
            torch.tensor(future).to(predicted_gmm.mus.device),
            max_log,
            reduce_mean=self.reduce_mean,
        )

    def __repr__(self):
        return "nll"
