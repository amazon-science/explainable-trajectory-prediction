# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import random

import numpy as np
import torch
import tqdm

from explainable_trajectory_prediction.trajectron import dataset
from explainable_trajectory_prediction.trajectron import network
from explainable_trajectory_prediction.trajectron import utilities

MIN_FUTURE_HORIZON = 12
MIN_HISTORY_LENGTH = 7
NUM_ITERATION = 10
METRICS = {
    "ade": utilities.ADE(False, None),
    "nll": utilities.NLL(False, None),
}
VARIANTS = {
    "zero": dataset.get_zero_neighbor,
    "random": dataset.get_random_neighbor,
}


def create_metric(metric_name):
    return METRICS[metric_name]


def create_variant(metric_name):
    return VARIANTS[metric_name]


def create_positive_negative_pair_history(agent_data, get_replacement_trajectory):
    """Create positive and negative pairs where they differ in the history.

    Given the data of a single agent, we create a batch of data where we randomly remove neighbours.
    The main difference between the positive/negative is the history where we set it to zero in the negative.

        Args:
            agent_data: The input agent data object containing the data of single agent.
            get_replacement_trajectory: A new trajectory generator to replace an existing one.

        Return:
            positive and negative agent data.
    """
    positive_list = []
    negative_list = []
    for _ in range(NUM_ITERATION):
        agent_data_positive = copy.deepcopy(agent_data)
        agent_data_positive.replace_neighbours_at_random(get_replacement_trajectory)
        agent_data_negative = copy.deepcopy(agent_data_positive)
        agent_data_negative.replace_history(get_replacement_trajectory)
        positive_list.append(agent_data_positive)
        negative_list.append(agent_data_negative)
    return (
        network.AgentData.from_list_of_agent_data(positive_list),
        network.AgentData.from_list_of_agent_data(negative_list),
    )


def create_positive_negative_pair_neighbor(agent_data, neighbor_node, get_replacement_trajectory):
    """Create positive and negative pairs where they differ for a single neighbor.

    Given the data of a single agent, we create a batch of data where we randomly remove neighbours.
    The main difference between the positive/negative is a single neighbor which is removed in the negative.

        Args:
            agent_data: The input agent data object containing the data of single agent.
            neighbor_node: The neighboring agent to remove in the negative.
            get_replacement_trajectory: A new trajectory generator to replace an existing one.

        Return:
            positive and negative agent data.
    """
    positive_list = []
    negative_list = []
    for _ in range(NUM_ITERATION):
        agent_data_positive = copy.deepcopy(agent_data)
        agent_data_positive.replace_neighbours_at_random(
            get_replacement_trajectory, non_replaceable_node=neighbor_node
        )
        if random.randint(0, 1) == 1:
            agent_data_positive.replace_history(get_replacement_trajectory)
        agent_data_negative = copy.deepcopy(agent_data_positive)
        agent_data_negative.replace_neighbor(get_replacement_trajectory, neighbor_node)
        positive_list.append(agent_data_positive)
        negative_list.append(agent_data_negative)
    return (
        network.AgentData.from_list_of_agent_data(positive_list),
        network.AgentData.from_list_of_agent_data(negative_list),
    )


class ShapleyValues:
    """The main class for estimating the baseline Shapley values."""

    def __init__(self, tester, node_type, random_node_types, metric):
        self.tester = tester
        self.node_type = node_type
        self.model = tester.trajectron.node_models_dict[node_type]
        self.random_node_types = random_node_types
        self.metric = metric
        self.metric.prior_sampler.latent = self.model.latent
        self.data_parameters = dataset.DataParameters(
            tester.trajectron.state,
            tester.trajectron.pred_state,
            MIN_HISTORY_LENGTH,
            tester.trajectron.max_ht,
            MIN_FUTURE_HORIZON,
            self.model.edge_types,
            tester.trajectron.hyperparams,
        )

    def predict_batch(self, batch, future):
        """Predict and compute the NLL of the ground truth.

        Args:
            batch: The input batch of all input tensors.
            future: The future trajectory of the target agent.

        Return:
            The error of the model.
            The predicted GMM.
        """
        predicted_gmm, predicted_samples = self.model.predict(
            batch,
            self.tester.hyperparams["prediction_horizon"],
            self.metric.prior_sampler,
            True,
            0.1,
        )
        predicted_samples = predicted_samples.cpu().detach().numpy()
        return (
            self.metric(
                predicted_gmm, predicted_samples, future, self.tester.hyperparams["log_p_yt_xz_max"]
            ),
            predicted_gmm,
        )

    def get_data(self, scene, timestep):
        """Get the agent data of a single scene at timestep.

        Args:
            scene: A scene for which data to be extracted.
            timestep: A timestep for the data.

        Return:
            An agent data of the all nodes.
        """
        batch = dataset.get_scene_data(
            self.tester.environment,
            scene,
            np.arange(timestep, timestep + 1),
            self.node_type,
            self.data_parameters,
            scenes=self.tester.environment.scenes,
            random_node_types=self.random_node_types,
        )
        if batch is None:
            return None, None
        (
            (
                first_history_indices,
                history,
                future,
                history_standardized,
                future_standardized,
                neighbors,
                neighbors_standardized,
                neighbors_edge_value,
                neighbors_nodes,
                neighbors_mask,
                map,
            ),
            nodes,
            _,
        ) = batch
        history = history.to(self.tester.device)
        history_standardized = history_standardized.to(self.tester.device)
        if type(map) == torch.Tensor:
            map = map.to(self.tester.device)
        return (
            network.AgentData(
                history,
                history_standardized,
                first_history_indices,
                neighbors,
                neighbors_standardized,
                neighbors_edge_value,
                neighbors_mask,
                neighbors_nodes,
                map=map,
            ),
            nodes,
        )

    def get_shapley_values(self, positive_batch, negative_batch, future):
        """Compute the Shapley values as the NLL difference between the negative and positive prediction.

        Args:
            positive_batch: The positive batch.
            negative_batch: The negative batch.
            future: The future trajectory of the target agent.

        Return:
            The NLL difference.
        """
        error_positive, _ = self.predict_batch(positive_batch, future)
        error_negative, _ = self.predict_batch(negative_batch, future)
        return np.mean(error_negative - error_positive)

    def run(self, scene, get_replacement_trajectory, max_timesteps=None, store_visualization=False):
        """Run the Shapley values for a specific scene.

        Args:
            scene: The input scene.
            get_replacement_trajectory: A new trajectory generator to replace an existing one.
            max_timesteps: The maximum number of timesteps to run, otherwise use the scene maximum timesteps.
            store_visualization: Store additional information for scenario visualization.

        Return:
            A result dictionary with all Shapley values of agents.
        """
        result = {}
        if not max_timesteps:
            max_timesteps = scene.timesteps
        for timestep in tqdm.tqdm(
            range(self.tester.hyperparams["maximum_history_length"], max_timesteps)
        ):
            agent_data, nodes = self.get_data(scene, timestep)
            if agent_data is None:
                continue
            for node_index, node in enumerate(nodes):
                node_name = str(node)
                if node_name not in result.keys():
                    result[node_name] = {}
                if timestep not in result[node_name].keys():
                    result[node_name][timestep] = {"shapley_values": {}}
                agent_data_target = agent_data.get_single_item(node_index)
                future = self.tester.get_future(node, timestep)
                history = self.tester.get_history(node, timestep)
                if store_visualization:
                    mean_position = node.get(np.array([timestep]), {"mean": ["x", "y"]})
                    _, predicted_gmm = self.predict_batch(agent_data_target, future)
                    means, covariance, pis = utilities.get_gmm_parameters(
                        predicted_gmm, mean_position
                    )
                    result[node_name][timestep]["pred_means"] = means
                    result[node_name][timestep]["pred_covariance"] = covariance
                    result[node_name][timestep]["pred_pis"] = pis
                    result[node_name][timestep]["history"] = history + mean_position
                    result[node_name][timestep]["future"] = future + mean_position
                    result[node_name][timestep]["neighbors"] = {}
                positive_batch, negative_batch = create_positive_negative_pair_history(
                    agent_data_target, get_replacement_trajectory
                )
                result[node_name][timestep]["shapley_values"]["past"] = self.get_shapley_values(
                    positive_batch, negative_batch, future
                )
                for edge_type, edge_neighbors in agent_data_target.neighbors_nodes.items():
                    for neighbor in edge_neighbors[0]:
                        if utilities.is_padded_neighbor(neighbor):
                            continue
                        positive_batch, negative_batch = create_positive_negative_pair_neighbor(
                            agent_data_target, neighbor, get_replacement_trajectory
                        )
                        result[node_name][timestep]["shapley_values"][
                            str(neighbor)
                        ] = self.get_shapley_values(positive_batch, negative_batch, future)
                        if store_visualization and not utilities.is_random_neighbor(neighbor):
                            result[node_name][timestep]["neighbors"][str(neighbor)] = (
                                self.tester.get_history(neighbor, timestep) + mean_position
                            )
        return result
