# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The main dataset loader of Trajectron++.

    This class is derived from the data loader of the Trajectron++ repository at
    https://github.com/StanfordASL/Trajectron-plus-plus, under trajectron/model/dataset
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage.

    We change the preprocessing functions to allow adding zero/random neighbors and
    return the original trajectory of the neighbors (not only the standardized version)."""
import dill
import environment
import model.dataset as dataset
import numpy as np
import random
import torch
from torch.utils import data

MASK_STATUS = {
    "valid": 1.0,
    "padded": 0.0,
    "random": 2.0,
}


class DataParameters:
    """Data parameters for every node data."""

    def __init__(
        self,
        state,
        prediction_state,
        minimum_history_length,
        maximum_history_length,
        maximum_future_length,
        edge_types,
        model_hyper_parameters,
    ):
        self.state = state
        self.prediction_state = prediction_state
        self.minimum_history_length = minimum_history_length
        self.maximum_history_length = maximum_history_length
        self.maximum_future_length = maximum_future_length
        self.edge_types = edge_types
        self.model_hyper_parameters = model_hyper_parameters


class SingleNodeData:
    """Object of all data of a single node at time step."""

    def __init__(
        self,
        scene,
        time_step,
        node,
        environment,
        parameters,
        scene_graph=None,
        random_trajectory=None,
        random_node_type=None,
    ):
        self.scene = scene
        self.node = node
        self.time_step = time_step
        self.environment = environment
        self.scene_graph = scene_graph
        self.parameters = parameters
        self._get_node_trajectory_data()
        self._get_neighbors(random_trajectory, random_node_type)
        self._standardized_node_trajectory_data()
        self.map_tuple = self._get_map()

    def _get_node_trajectory_data(self):
        """Get the past and future trajectory of the agent."""
        self.history = self.node.get(
            np.array([self.time_step - self.parameters.maximum_history_length, self.time_step]),
            self.parameters.state[self.node.type],
        )
        self.future = self.node.get(
            np.array([self.time_step + 1, self.time_step + self.parameters.maximum_future_length]),
            self.parameters.prediction_state[self.node.type],
        )
        self.first_history_index = (
            self.parameters.maximum_history_length - self.node.history_points_at(self.time_step)
        ).clip(0)

    def _get_standard_deviation(self, edge_type):
        """Gets the standard deviation of an edge type.

        Args:
            edge_type: A tuble of the form (from_node, to_node).

        Returns:
            A vector of the standard deviation.
        """
        _, std = self.environment.get_standardize_params(
            self.parameters.state[edge_type[1]], edge_type[1]
        )
        std[0:2] = self.environment.attention_radius[edge_type]
        return std

    def _standardized_node_trajectory_data(self):
        """Standardize both the history and future relative to the last observed state."""
        std = self._get_standard_deviation((self.node.type, self.node.type))
        relative_state = np.zeros_like(self.history[0])
        relative_state[0:2] = np.array(self.history)[-1, 0:2]
        self.history_standardized = self.environment.standardize(
            self.history,
            self.parameters.state[self.node.type],
            self.node.type,
            mean=relative_state,
            std=std,
        )
        if list(self.parameters.prediction_state[self.node.type].keys())[0] == "position":
            self.future_standardized = self.environment.standardize(
                self.future,
                self.parameters.prediction_state[self.node.type],
                self.node.type,
                mean=relative_state[0:2],
            )
        else:
            self.future_standardized = self.environment.standardize(
                self.future, self.parameters.prediction_state[self.node.type], self.node.type
            )

    def _standardize_neighbor_data(self, edge_type, neighbor_state_data):
        """Standardize the neighbors data relative to the last observed state of the target agent.

        Args:
            edge_type: A tuble of the form (from_node, to_node).
            neighbor_state_data: An array of the trajectory of the neighboring agent.

        Returns:
            An array of the standardized trajectory of the neighboring agent.
        """
        std = self._get_standard_deviation(edge_type)
        equal_dims = np.min((neighbor_state_data.shape[-1], self.history.shape[-1]))
        relative_state = np.zeros_like(neighbor_state_data)
        relative_state[:, ..., :equal_dims] = self.history[-1, ..., :equal_dims]
        return self.environment.standardize(
            neighbor_state_data,
            self.parameters.state[edge_type[1]],
            node_type=edge_type[1],
            mean=relative_state,
            std=std,
        )

    def _add_random_neighbor(self, random_trajectory, edge_type):
        """Add a random trajectory as random neighbor.

        Args:
            edge_type: A tuble of the form (from_node, to_node).
            random_trajectory: A tensor of the trajectory of a random agent.
        """
        self.neighbors_data[edge_type].append(random_trajectory)
        self.neighbors_data_st[edge_type].append(random_trajectory)
        self.neighbors_mask[edge_type].append(torch.tensor(MASK_STATUS["random"]))

    def _pad_zero_neighbors(self, edge_type):
        """Pad the list of neighbors with zero neighbors until the maximum number of neighbors is reached.

        Args:
            edge_type: A tuble of the form (from_node, to_node).
        """
        padded_neighbors = []
        num_padded_neighbors = self.parameters.model_hyper_parameters[
            "max_number_of_neighbors"
        ] - len(self.neighbors_data[edge_type])
        neighbor_state_dim = np.sum([len(x) for x in self.parameters.state[edge_type[1]].values()])
        num_elements = self.history.shape[0]
        for k in range(num_padded_neighbors):
            self.neighbors_data_st[edge_type].append(
                torch.zeros((num_elements, neighbor_state_dim))
            )
            self.neighbors_mask[edge_type].append(torch.tensor(MASK_STATUS["padded"]))
            padded_neighbors.append(environment.Node(edge_type[1], "padded", None))
        return padded_neighbors

    def _get_neighbors(self, random_trajectory, random_node_type):
        """Get the neighbors of the current node.

        Args:
            random_trajectory: An array of the trajectory of a random agent.
            random_node_type: The node type of the random agent.
        """
        if not self.scene_graph:
            self.scene_graph = self.scene.get_scene_graph(
                self.time_step,
                self.environment.attention_radius,
                self.parameters.model_hyper_parameters["edge_addition_filter"],
                self.parameters.model_hyper_parameters["edge_removal_filter"],
            )
        self.neighbors_data = dict()
        self.neighbors_data_st = dict()
        self.neighbors_edge_value = dict()
        self.neighbors_nodes = dict()
        self.neighbors_mask = dict()
        for edge_type in self.parameters.edge_types:
            self.neighbors_data[edge_type] = list()
            self.neighbors_data_st[edge_type] = list()
            self.neighbors_mask[edge_type] = list()

            connected_nodes = self.scene_graph.get_neighbors(self.node, edge_type[1])
            self.neighbors_nodes[edge_type] = connected_nodes
            if self.parameters.model_hyper_parameters["dynamic_edges"] == "yes":
                self.neighbors_edge_value[edge_type] = torch.tensor(
                    self.scene_graph.get_edge_scaling(self.node), dtype=torch.float
                )

            for connected_node in connected_nodes:
                neighbor_state_array = connected_node.get(
                    np.array(
                        [self.time_step - self.parameters.maximum_history_length, self.time_step]
                    ),
                    self.parameters.state[connected_node.type],
                    padding=0.0,
                )
                self.neighbors_data[edge_type].append(
                    torch.tensor(neighbor_state_array, dtype=torch.float)
                )
                self.neighbors_data_st[edge_type].append(
                    torch.tensor(
                        self._standardize_neighbor_data(edge_type, neighbor_state_array),
                        dtype=torch.float,
                    )
                )
                self.neighbors_mask[edge_type].append(torch.tensor(MASK_STATUS["valid"]))

            invalid_neighbors = []
            if random_node_type == edge_type[1]:
                self._add_random_neighbor(random_trajectory, edge_type)
                invalid_neighbors.append(environment.Node(edge_type[1], "random", None))

            invalid_neighbors.extend(self._pad_zero_neighbors(edge_type))
            self.neighbors_nodes[edge_type] = np.concatenate(
                [self.neighbors_nodes[edge_type], invalid_neighbors]
            )

    def _get_map(self):
        """Get the semantic map of the scene where the current node exists."""
        if (
            not self.parameters.model_hyper_parameters["use_map_encoding"]
            or self.node.type not in self.parameters.model_hyper_parameters["map_encoder"]
        ):
            return None
        trajectory = self.history
        if self.node.non_aug_node is not None:
            trajectory = self.node.non_aug_node.get(
                np.array([self.time_step]), self.parameters.state[self.node.type]
            )
        me_hyp = self.parameters.model_hyper_parameters["map_encoder"][self.node.type]
        if "heading_state_index" in me_hyp:
            heading_state_index = me_hyp["heading_state_index"]
            # We have to rotate the map in the opposite direction of the agent to match them
            if type(heading_state_index) is list:
                heading_angle = (
                    -np.arctan2(
                        trajectory[-1, heading_state_index[1]],
                        trajectory[-1, heading_state_index[0]],
                    )
                    * 180
                    / np.pi
                )
            else:
                heading_angle = -trajectory[-1, heading_state_index] * 180 / np.pi
        else:
            heading_angle = None

        scene_map = self.scene.map[self.node.type]
        map_point = trajectory[-1, :2]
        patch_size = self.parameters.model_hyper_parameters["map_encoder"][self.node.type][
            "patch_size"
        ]
        return scene_map, map_point, heading_angle, patch_size

    def get_data(self):
        return (
            self.first_history_index,
            torch.tensor(self.history, dtype=torch.float),
            torch.tensor(self.future, dtype=torch.float),
            torch.tensor(self.history_standardized, dtype=torch.float),
            torch.tensor(self.future_standardized, dtype=torch.float),
            self.neighbors_data,
            self.neighbors_data_st,
            self.neighbors_edge_value,
            self.neighbors_nodes,
            self.neighbors_mask,
            self.map_tuple,
        )


class EnvironmentDataset(dataset.EnvironmentDataset):
    def __init__(
        self, environment, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs
    ):
        self.node_type_datasets = list()
        self._augment = False
        for node_type in environment.NodeType:
            if node_type not in hyperparams["pred_state"]:
                continue
            self.node_type_datasets.append(
                NodeTypeDataset(
                    environment,
                    node_type,
                    state,
                    pred_state,
                    node_freq_mult,
                    scene_freq_mult,
                    hyperparams,
                    **kwargs
                )
            )


class NodeTypeDataset(dataset.NodeTypeDataset):
    def __init__(
        self,
        environment,
        node_type,
        state,
        prediction_state,
        node_freq_multiplier,
        scene_freq_multiplier,
        hyperparams,
        augment=False,
        **kwargs
    ):
        super().__init__(
            environment,
            node_type,
            state,
            prediction_state,
            node_freq_multiplier,
            scene_freq_multiplier,
            hyperparams,
            augment,
            **kwargs
        )

    def __getitem__(self, i):
        (scene, time_step, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)
        data_parameters = DataParameters(
            self.state,
            self.pred_state,
            self.hyperparams["minimum_history_length"],
            self.hyperparams["maximum_history_length"],
            self.max_ft,
            self.edge_types,
            self.hyperparams,
        )
        node_data = SingleNodeData(scene, time_step, node, self.env, data_parameters)
        return node_data.get_data()


def get_random_neighbor(scenes, parameters, environment, edge_type):
    random_target_standardized_trajectory = None
    random_neighbor_standardized_trajectory = None
    while random_neighbor_standardized_trajectory is None:
        random_scene = scenes[random.choice([j for j in range(len(scenes))])]
        random_timestep = random.choice(
            [
                j
                for j in range(
                    parameters.minimum_history_length,
                    random_scene.timesteps - parameters.maximum_future_length,
                )
            ]
        )
        nodes_per_time_step = random_scene.present_nodes(
            np.arange(random_timestep, random_timestep + 1),
            type=edge_type[0],
            min_history_timesteps=parameters.minimum_history_length,
            min_future_timesteps=parameters.maximum_future_length,
            return_robot=True,
        )
        if nodes_per_time_step:
            scene_graph = random_scene.get_scene_graph(
                random_timestep,
                environment.attention_radius,
                parameters.model_hyper_parameters["edge_addition_filter"],
                parameters.model_hyper_parameters["edge_removal_filter"],
            )
            for node in nodes_per_time_step[random_timestep]:
                node_data = SingleNodeData(
                    random_scene, random_timestep, node, environment, parameters, scene_graph
                )
                if node_data.neighbors_mask[edge_type][0] == 1.0:
                    random_target_standardized_trajectory = node_data.history_standardized
                    random_neighbor_standardized_trajectory = node_data.neighbors_data_st[
                        edge_type
                    ][0]
                    break
    return (
        torch.tensor(random_target_standardized_trajectory),
        random_neighbor_standardized_trajectory,
        torch.tensor(1.0),
    )


def get_zero_neighbor(_scenes, parameters, _environment, edge_type):
    state_dimension = np.sum([len(x) for x in parameters.state[edge_type[0]].values()])
    zero_trajectory = torch.zeros(parameters.maximum_history_length + 1, state_dimension)
    return zero_trajectory, None, torch.tensor(0.0)


def get_scene_data(
    environment, scene, time_steps, node_type, parameters, scenes=None, random_node_types=[]
):
    """Batch the data of all nodes of a scene within a set of time steps.

    Args:
        environment: An environment object for the whole dataset.
        scene: The scene from which the data of the nodes to be extracted.
        time_steps: A range of timesteps for the nodes to exist.
        node_type: The type of the nodes to be extracted.
        parameters: Hyper-parameters for the data of every node.
        scenes: list of all environment scenes to sample a random neighbor from.
        random_node_types: list of allowed random node types.

    Returns:
        A batch of data, each for a specific node at specific time.
        A list of the nodes.
        A list of the time steps.
    """
    nodes_per_time_step = scene.present_nodes(
        time_steps,
        type=node_type,
        min_history_timesteps=parameters.minimum_history_length,
        min_future_timesteps=parameters.maximum_future_length,
        return_robot=True,
    )
    batch = list()
    nodes = list()
    observed_timesteps = list()
    for current_time_step in nodes_per_time_step.keys():
        scene_graph = scene.get_scene_graph(
            current_time_step,
            environment.attention_radius,
            parameters.model_hyper_parameters["edge_addition_filter"],
            parameters.model_hyper_parameters["edge_removal_filter"],
        )
        present_nodes = nodes_per_time_step[current_time_step]
        for node in present_nodes:
            nodes.append(node)
            observed_timesteps.append(current_time_step)
            random_trajectory = None
            random_node_type = None
            if random_node_types:
                random_node_type = random.choice(random_node_types)
                _, random_trajectory, _ = get_random_neighbor(
                    scenes, parameters, environment, (node_type, random_node_type)
                )
            node_data = SingleNodeData(
                scene,
                current_time_step,
                node,
                environment,
                parameters,
                scene_graph,
                random_trajectory,
                random_node_type,
            )
            batch.append(node_data.get_data())
    if len(observed_timesteps) == 0:
        return None
    return dataset.collate(batch), nodes, observed_timesteps
