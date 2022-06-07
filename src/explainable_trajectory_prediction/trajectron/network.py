# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The main graph neural network model.

    Some of these classes are derived from the thirdparty/trajectron/model
    which are part of the Trajectron++ repository at:
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage."""
import random
import warnings

import environment
import model
import model.dataset as third_party_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from explainable_trajectory_prediction.trajectron import dataset
from explainable_trajectory_prediction.trajectron import model_components

ACTION_STATE_HIDDEN_DIMENSION = 32


class AgentData(object):
    """Represents all relevant data for one target agent."""

    def __init__(
        self,
        history,
        history_standardized,
        first_history_indices,
        neighbors,
        neighbors_standardized,
        neighbors_edge_value,
        neighbors_mask,
        neighbors_nodes,
        future=None,
        future_standardized=None,
        map=None,
    ):
        self.history = history
        self.history_standardized = history_standardized
        self.first_history_indices = first_history_indices
        self.neighbors_standardized = neighbors_standardized
        self.neighbors = neighbors
        self.neighbors_edge_value = neighbors_edge_value
        self.neighbors_mask = neighbors_mask
        self.neighbors_nodes = neighbors_nodes
        self.future = future
        self.future_standardized = future_standardized
        self.map = map

    @classmethod
    def from_list_of_agent_data(cls, agent_data_list):
        """Combine a list of agent data in a single object."""
        history = torch.cat([x.history for x in agent_data_list], dim=0)
        history_standardized = torch.cat([x.history_standardized for x in agent_data_list], dim=0)
        first_history_indices = torch.cat([x.first_history_indices for x in agent_data_list], dim=0)
        neighbors = {}
        neighbors_standardized = {}
        neighbors_mask = {}
        neighbors_edge_value = {}
        neighbors_nodes = {}
        for edge_type in agent_data_list[0].neighbors.keys():
            neighbors[edge_type] = [x.neighbors[edge_type][0] for x in agent_data_list]
            neighbors_standardized[edge_type] = [
                x.neighbors_standardized[edge_type][0] for x in agent_data_list
            ]
            neighbors_mask[edge_type] = [x.neighbors_mask[edge_type][0] for x in agent_data_list]
            neighbors_edge_value[edge_type] = [
                x.neighbors_edge_value[edge_type][0] for x in agent_data_list
            ]
            neighbors_nodes[edge_type] = [x.neighbors_nodes[edge_type][0] for x in agent_data_list]
        return cls(
            history,
            history_standardized,
            first_history_indices,
            neighbors,
            neighbors_standardized,
            neighbors_edge_value,
            neighbors_mask,
            neighbors_nodes,
            map=agent_data_list[0].map,
        )

    def get_single_item(self, index):
        """Get the agent data of a single batch element."""
        first_history_indices_target = self.first_history_indices[index : index + 1]
        history_target = self.history[index : index + 1]
        history_standardized_target = self.history_standardized[index : index + 1]
        neighbors_target = {}
        neighbors_standardized_target = {}
        neighbors_edge_value_target = {}
        neighbors_mask_target = {}
        neighbors_nodes_target = {}
        for edge_type in self.neighbors.keys():
            neighbors_target[edge_type] = [self.neighbors[edge_type][index]]
            neighbors_standardized_target[edge_type] = [
                self.neighbors_standardized[edge_type][index]
            ]
            neighbors_edge_value_target[edge_type] = [self.neighbors_edge_value[edge_type][index]]
            neighbors_mask_target[edge_type] = [self.neighbors_mask[edge_type][index]]
            neighbors_nodes_target[edge_type] = [self.neighbors_nodes[edge_type][index]]
        return self.__class__(
            history_target,
            history_standardized_target,
            first_history_indices_target,
            neighbors_target,
            neighbors_standardized_target,
            neighbors_edge_value_target,
            neighbors_mask_target,
            neighbors_nodes_target,
            map=self.map,
        )

    def replace_history(self, get_replacement_trajectory):
        """Replace the history of the target agent.

        Args:
            get_replacement_trajectory: A new trajectory generator to replace an existing one.
        """
        replacement_history, _, _ = get_replacement_trajectory(list(self.neighbors_mask.keys())[0])
        self.history_standardized = (
            replacement_history.type(self.history_standardized.dtype)
            .to(self.history_standardized.device)
            .unsqueeze(0)
        )

    def replace_neighbours_at_random(self, get_replacement_trajectory, non_replaceable_node=None):
        """Randomly replace neighbors by replacing their trajectories and masks.

        Args:
            non_replaceable_node: A neighbor that should not be replaced.
            get_replacement_trajectory: A new trajectory generator to replace an existing one.
        """
        for edge_type in self.neighbors_mask.keys():
            edge_neighbors_mask = self.neighbors_mask[edge_type][0]
            edge_neighbors_standardized = self.neighbors_standardized[edge_type][0]
            for index in range(len(edge_neighbors_mask)):
                if non_replaceable_node and str(self.neighbors_nodes[edge_type][0][index]) == str(
                    non_replaceable_node
                ):
                    continue
                if random.randint(0, 1) == 1 and edge_neighbors_mask[index].item() != 0:
                    _, random_neighbor, random_mask = get_replacement_trajectory(edge_type)
                    if random_neighbor is not None:
                        edge_neighbors_standardized[index] = random_neighbor.type(
                            edge_neighbors_standardized[index].dtype
                        )
                    edge_neighbors_mask[index] = random_mask

    def replace_neighbor(self, get_replacement_trajectory, node):
        """Replace a specific neighbor by replacing its trajectory and mask.

        Args:
            node: the neighboring node to be replaced.
            get_replacement_trajectory: A new trajectory generator to replace an existing one.
        """
        for edge_type in self.neighbors_mask.keys():
            edge_neighbors_mask = self.neighbors_mask[edge_type][0]
            edge_neighbors_standardized = self.neighbors_standardized[edge_type][0]
            for index in range(len(edge_neighbors_mask)):
                if str(self.neighbors_nodes[edge_type][0][index]) == str(node):
                    _, random_neighbor, random_mask = get_replacement_trajectory(edge_type)
                    if random_neighbor is not None:
                        edge_neighbors_standardized[index] = random_neighbor.type(
                            edge_neighbors_standardized[index].dtype
                        )
                    edge_neighbors_mask[index] = random_mask


class SingleTimeStepDecoder(object):
    """The LSTM decoder for a single time step in the future."""

    def __init__(self, gmm_parameters_projector, lstm_cell, gmm_mode=False, resolution=0.1):
        """Initialize the decoder.

        Args:
            gmm_parameters_projector: A function which predicts the gmm parameters.
            lstm_cell: The LSTM cell function which predicts the next hidden state.
            gmm_mode: If True, the most likelihood sample of the predicted GMM is computed.
            resolution: The resolution of the grid when estimating the most likelihood sample.
        """
        self.gmm_mode = gmm_mode
        self.resolution = resolution
        self.gmm_parameters_projector = gmm_parameters_projector
        self.lstm_cell = lstm_cell

    def run(self, mode, input, previous_state):
        """Decode the parameters of a Gaussian mixture model for the next time step.

        Args:
            mode: Mode in which the model is operated. E.g. Train, or Predict.
            input: The input feature vector of the LSTM decoder.
            previous_state: The previous hidden state of the LSTM decoder.

        Return:
            mean: The means of the mixture components.
            log_sigma: The uncertainty of the mixture components (in log).
            correlation: The correlation matrix of the uncertainty matrix.
            hidden_state: The hidden state of the future time step.
            current_prediction: The prediction of the future state for the next time step.
        """
        hidden_state = self.lstm_cell(input, previous_state)
        log_pi, mean, log_sigma, correlation = self.gmm_parameters_projector(hidden_state)
        gmm_distribution = model_components.GMM2D(log_pi, mean, log_sigma, correlation)
        if mode == model.model_utils.ModeKeys.PREDICT and self.gmm_mode:
            current_prediction = gmm_distribution.mode(resolution=self.resolution)
        else:
            current_prediction = gmm_distribution.rsample()
        return mean, log_sigma, correlation, hidden_state, current_prediction


class PriorSampler(object):
    """The prior sampler for the latent space."""

    def __init__(
        self, latent, num_samples, latent_mode=False, full_dist=True, all_latent_separated=False
    ):
        """Initialize the prior sampler.

        Args:
            latent: The latent object to sample from.
            num_samples: Number of samples to draw from the latent space.
            latent_mode: If True: Select the most likely latent state.
            all_latent_separated: Samples each latent mode individually without merging them into a GMM.
            full_dist: Samples all latent states and merges them into a GMM as output.
        """
        self.latent = latent
        self.num_samples = num_samples
        self.latent_mode = latent_mode
        self.full_dist = full_dist
        self.all_latent_separated = all_latent_separated

    def run(self, mode):
        """Generate a number of samples from the prior distribution.

        Args:
            mode: Mode in which the model is operated. E.g. Train, or Predict.

        Return:
            latent_samples: The generated latent samples.
            num_samples: Number of the generated samples.
            num_components: Number og GMM components.
        """
        return self.latent.sample_p(
            self.num_samples,
            mode,
            most_likely_z=self.latent_mode,
            full_dist=self.full_dist,
            all_z_sep=self.all_latent_separated,
        )


class ExplainableMultimodalGenerativeCVAE(model.MultimodalGenerativeCVAE):
    """Represents the graph neural network for every node type."""

    def __init__(
        self,
        environment,
        node_type,
        model_registrar,
        hyperparams,
        device,
        edge_types,
        log_writer=None,
    ):
        """Instantiates the model parameters.

        Args:
            environment: The environment object containing the data for all scenes.
            node_type: The node type of this model.
            model_registrar: The object containing the parameters of the network.
            hyperparams: A dictionary for hyper-parameters that are needed for constructing the model.
            device: The device of the memory (cpu or cuda).
            edge_types: A list of all possible edge types, [(from_node_type, to_node_type)].
            log_writer: The log writer object which logs in tensorboard during training.
        """
        super().__init__(
            environment, node_type, model_registrar, hyperparams, device, edge_types, log_writer
        )
        dynamic_class = getattr(model_components, hyperparams["dynamic"][node_type]["name"])
        dynamic_limits = hyperparams["dynamic"][node_type]["limits"]
        self.dynamic = dynamic_class(
            environment.scenes[0].dt,
            dynamic_limits,
            device,
            model_registrar,
            self.x_size,
            node_type,
        )

    def replace_layer(self, layer_name, module):
        """Replace the module of a specific layer name."""
        module.load_state_dict(self.model_registrar.model_dict[layer_name].state_dict())
        self.model_registrar.model_dict[layer_name] = module
        self.node_modules[layer_name] = self.model_registrar.model_dict[layer_name]

    def create_node_models(self):
        """Create the network architecture of the model."""
        super().create_node_models()
        if self.check_if_hyperparameter_exists("edge_variant"):
            self.replace_layer(
                self.node_type + "/decoder/state_action",
                nn.Sequential(nn.Linear(self.state_length, 32)),
            )
            self.add_submodule(
                self.node_type + "/decoder/state_action_two",
                model_if_absent=nn.Sequential(nn.Linear(32, self.pred_state_length)),
            )
        self.replace_layer(
            self.node_type + "/edge_influence_encoder",
            model_components.AdditiveAttention(
                encoder_hidden_state_dim=self.hyperparams["enc_rnn_dim_edge_influence"],
                decoder_hidden_state_dim=self.hyperparams["enc_rnn_dim_history"],
            ),
        )

    def initialize_dynamics(self, history):
        """Initialize the dynamics of the model by setting the current position and velocity.

        Both the inputs and outputs of the model are standardized such that all trajectories
        are relative to the target agent. The predicted relative trajectories are then converted
        back to the absolute coordinates using this dynamic object.

            Args
                history: The history state of the target agent over time.
        """
        initial_dynamics = dict()
        initial_dynamics["pos"] = history[:, -1, 0:2]
        initial_dynamics["vel"] = history[:, -1, 2:4]
        self.dynamic.set_initial_condition(initial_dynamics)

    def check_if_hyperparameter_exists(self, name):
        """Check the existence of a hyperparameter and its boolean value.

        Args:
            name: The name of the hyperparameter.

        Return:
            True If the hyperparameter exists and is set to True,
            otherwise False.
        """
        return name in self.hyperparams.keys() and self.hyperparams[name]

    def stack_neighbors(self, neighbors, batch_size):
        """Stack the trajectories of the target agent neighbors.

        Args:
            neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
            batch_size: Number of samples in the batch.

        Return:
            The stacked trajectories of neighbors or a zero trajectory if no neighbors were found.
        """
        max_hl = self.hyperparams["maximum_history_length"]
        if neighbors:
            stacked_neighbors = [torch.stack(x) for x in neighbors]
            return torch.stack(stacked_neighbors, dim=0).to(self.device)
        return torch.zeros((batch_size, 1, max_hl + 1, self.state_length), device=self.device)

    def decode_initial_state_and_prediction(
        self,
        encoded_input,
        latent_stacked,
        standardized_current_state,
        num_samples,
        num_components=1,
    ):
        """Decode the initial state/prediction conditioned on history, neighbors and latent samples.

        Args:
            encoded_input: A feature vector encoding both the history and neighbors.
            latent_stacked: A tensor stacking multiple latent samples.
            standardized_current_state: The standardized last observed state of the target agent.
            num_samples: Number of samples to draw from the latent space.
            num_components: Number of GMM components to be predicted at every time step.

        Return:
            initial_state: The initial predicted hidden state of the next time step.
            initial_predicton: The initial prediction for the next time step
            encoded_condition: The feature vector used as condition.
        """
        latent_stacked = torch.reshape(latent_stacked, (-1, self.latent.z_dim))
        encoded_condition = torch.cat(
            [latent_stacked, encoded_input.repeat(num_samples * num_components, 1)], dim=1
        )
        initial_state = self.node_modules[self.node_type + "/decoder/initial_h"](encoded_condition)
        initial_prediction = self.node_modules[self.node_type + "/decoder/state_action"](
            standardized_current_state
        )
        if self.check_if_hyperparameter_exists("edge_variant"):
            initial_prediction = self.node_modules[self.node_type + "/decoder/state_action_two"](
                initial_prediction
            )
        return initial_state, initial_prediction, encoded_condition

    def get_log_mixture_component_weight(self, mode, num_samples, num_components, correlation):
        """Get the log of the weight for every Gaussian mixture component.

        Args:
            mode: Mode in which the model is operated. E.g. Train, or Predict.
            num_samples: Number of samples to draw from the latent space.
            num_components: Number of GMM components to be predicted at every time step.
            correlation: The correlation matrix of the uncertainty matrix.

        Return:
            log_pi: The log of the mixture weights.
        """
        if num_components > 1:
            if mode == model.model_utils.ModeKeys.PREDICT:
                return self.latent.p_dist.logits.repeat(num_samples, 1, 1)
            return self.latent.q_dist.logits.repeat(num_samples, 1, 1)
        return torch.ones_like(
            correlation.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1)
        )

    def reshape_mean_or_sigma(self, gmm_parameter, num_samples, num_components):
        """Reshape the means or log_sigmas of the predicted gmm distribution.

        Args:
            gmm_parameter: Either the means or log_sigmas of a GMM.
            num_samples: Number of samples to draw from the latent space.
            num_components: Number of GMM components to be predicted at every time step.

        Return:
            Reshaped tensor.
        """
        return (
            gmm_parameter.reshape(num_samples, num_components, -1, 2)
            .permute(0, 2, 1, 3)
            .reshape(-1, 2 * num_components)
        )

    def reshape_correlation(self, gmm_parameter, num_samples, num_components):
        """Reshape the correlation matrix of the gmm.

        Args:
            gmm_parameter: Either the means or log_sigmas of a GMM.
            num_samples: Number of samples to draw from the latent space.
            num_components: Number of GMM components to be predicted at every time step.

        Return:
            Reshaped tensor.
        """
        return (
            gmm_parameter.reshape(num_samples, num_components, -1)
            .permute(0, 2, 1)
            .reshape(-1, num_components)
        )

    def decode_future_trajectory(
        self,
        mode,
        initial_state,
        initial_prediction,
        encoded_condition,
        gmm_parameters_decoder,
        prediction_horizon,
        num_samples,
        num_components=1,
    ):
        """Decode the parameters of a Gaussian mixture model for multiple future time steps.

        Args:
            mode: Mode in which the model is operated. E.g. Train, or Predict.
            state: The initial predicted hidden state of the next time step.
            initial_prediction: The initial prediction for the next time step
            encoded_condition: The feature vector used as condition.
            gmm_parameters_decoder: A function to predict the parameters of the GMM for the next timestep.
            prediction_horizon: Number of time steps to predict in the future.
            num_samples: Number of samples to draw from the latent space.
            num_components: Number of GMM components to be predicted at every time step.

        Return:
            predicted_gmm: GMM2D object for all parameters of the mixture model of the predicted future trajectory.
        """
        log_pis, means, log_sigmas, correlations = [], [], [], []
        decoder_input = torch.cat(
            [encoded_condition, initial_prediction.repeat(num_samples * num_components, 1)], dim=1
        )
        state = initial_state
        for future_timestep in range(prediction_horizon):
            mean, log_sigma, correlation, state, current_prediction = gmm_parameters_decoder.run(
                mode=mode, input=decoder_input, previous_state=state
            )
            decoder_input = torch.cat([encoded_condition, current_prediction], dim=1)
            log_pi = self.get_log_mixture_component_weight(
                mode, num_samples, num_components, correlation
            )

            means.append(self.reshape_mean_or_sigma(mean, num_samples, num_components))
            log_sigmas.append(self.reshape_mean_or_sigma(log_sigma, num_samples, num_components))
            correlations.append(self.reshape_correlation(correlation, num_samples, num_components))
            log_pis.append(log_pi)

        log_pis = torch.reshape(
            torch.stack(log_pis, dim=1), [num_samples, -1, prediction_horizon, num_components]
        )
        means = torch.reshape(
            torch.stack(means, dim=1),
            [num_samples, -1, prediction_horizon, num_components * self.pred_state_length],
        )
        log_sigmas = torch.reshape(
            torch.stack(log_sigmas, dim=1),
            [num_samples, -1, prediction_horizon, num_components * self.pred_state_length],
        )
        correlations = torch.reshape(
            torch.stack(correlations, dim=1), [num_samples, -1, prediction_horizon, num_components]
        )

        return model_components.GMM2D(log_pis, means, log_sigmas, correlations)

    def combine_neighbors(self, agent_data, edge_type):
        """Combine (sum) all trajectories of neighbors of the same type.

        Args:
            agent_data: The data object of the target agent.
            edge_type: (from_node_type, to_node_type).

        Return:
            The combined trajectory.
        """
        edge_states_list = list()
        for i, neighbor_states in enumerate(agent_data.neighbors_standardized[edge_type]):
            if len(neighbor_states) == 0:
                edge_state = torch.zeros(
                    (1, self.hyperparams["maximum_history_length"] + 1, neighbor_state_length),
                    device=self.device,
                )
            else:
                stacked_neighbors = torch.stack(neighbor_states, dim=0)
                stacked_masks = (
                    (torch.stack(agent_data.neighbors_mask[edge_type][i], dim=0) > 0)
                    .int()
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                edge_state = (stacked_neighbors * stacked_masks).to(self.device)
            edge_states_list.append(torch.sum(edge_state, dim=0))
        return torch.stack(edge_states_list, dim=0)

    def combine_edge_masks(self, agent_data, edge_type):
        """Combine (sum) all edge masks of neighbors of the same type.

        Args:
            agent_data: The data object of the target agent.
            edge_type: (from_node_type, to_node_type).

        Return:
            The combined edge mask.
        """
        edge_mask_list = list()
        for edge_value in agent_data.neighbors_edge_value[edge_type]:
            edge_mask_list.append(
                torch.clamp(torch.sum(edge_value.to(self.device), dim=0, keepdim=True), max=1.0)
            )
        return torch.stack(edge_mask_list, dim=0)

    def encode_edge(self, mode, edge_type, agent_data):
        """Encode the all neighbors from the same type.

        Args:
            mode: Mode in which the model is operated. E.g. Train, or Predict.
            edge_type: (from_node_type, to_node_type).
            agent_data: The data object of the target agent.

        Return:
            A feature vector encoding all neighbors of the same type.
        """
        agent_data_shape = agent_data.history_standardized.shape
        neighbor_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
        )
        combined_neighbors = torch.zeros(
            (agent_data_shape[0], agent_data_shape[1], neighbor_state_length)
        ).to(self.device)
        combined_edge_masks = torch.ones((agent_data_shape[0], 1)).to(self.device)
        if agent_data.neighbors:
            combined_neighbors = self.combine_neighbors(agent_data, edge_type)
            combined_edge_masks = self.combine_edge_masks(agent_data, edge_type)

        joint_history = torch.cat([combined_neighbors, agent_data.history_standardized], dim=-1)
        outputs, _ = model.model_utils.run_lstm_on_variable_length_seqs(
            self.node_modules[
                environment.scene_graph.DirectedEdge.get_str_from_types(*edge_type)
                + "/edge_encoder"
            ],
            original_seqs=joint_history,
            lower_indices=agent_data.first_history_indices,
        )
        outputs = F.dropout(
            outputs,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == model.model_utils.ModeKeys.TRAIN),
        )
        last_index_per_sequence = -(agent_data.first_history_indices + 1)
        edge_encoding = outputs[
            torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence
        ]
        return (edge_encoding * combined_edge_masks).unsqueeze(1)

    def encode_edge_late_fusion(self, mode, edge_type, agent_data):
        agent_data_shape = agent_data.history_standardized.shape
        neighbor_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
        )
        combined_neighbors = torch.zeros(
            (
                agent_data_shape[0],
                1,
                self.hyperparams["maximum_history_length"] + 1,
                neighbor_state_length,
            )
        ).to(self.device)
        if agent_data.neighbors_standardized:
            stacked_neighbors = [
                torch.stack(x) for x in agent_data.neighbors_standardized[edge_type]
            ]
            combined_neighbors = torch.stack(stacked_neighbors, dim=0).to(self.device)

        num_neighbors = combined_neighbors.shape[1]
        repeated_history_standardized = agent_data.history_standardized.unsqueeze(1).repeat(
            (1, num_neighbors, 1, 1)
        )
        joint_history = torch.cat([combined_neighbors, repeated_history_standardized], dim=-1)
        results = []
        for neighbor_index in range(num_neighbors):
            outputs, _ = model.model_utils.run_lstm_on_variable_length_seqs(
                self.node_modules[
                    environment.scene_graph.DirectedEdge.get_str_from_types(*edge_type)
                    + "/edge_encoder"
                ],
                original_seqs=joint_history[:, neighbor_index, :, :],
                lower_indices=agent_data.first_history_indices,
            )
            outputs = F.dropout(
                outputs,
                p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                training=(mode == model.model_utils.ModeKeys.TRAIN),
            )
            last_index_per_sequence = -(agent_data.first_history_indices + 1)
            results.append(
                outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
            )
        return torch.stack(results, dim=1)

    def encode_neighbors(self, mode, agent_data, node_history_encoded):
        """Encode the information from all neighbors of the target agent.

        First encode the edges according to the neighbors' types (i.e, all pedestrians-pedestrians),
        then aggregate all edges to get the total edge influence feature vector
        to be used later in the decoder.

            Args:
                mode: Mode in which the model is operated. E.g. Train, or Predict.
                agent_data: The data object of the target agent.
                node_history_encoded: The feature encoding the history of the target agent.

            Return:
                A feature vector encoding all neighbors history trajectories.
        """
        node_edges_encoded = list()
        node_edges_encoded_mask = list()
        for edge_type in self.edge_types:
            if self.check_if_hyperparameter_exists("edge_variant"):
                encoded_edges_type = self.encode_edge_late_fusion(mode, edge_type, agent_data)
            else:
                encoded_edges_type = self.encode_edge(mode, edge_type, agent_data)
            node_edges_encoded.append(encoded_edges_type)
            if agent_data.neighbors_mask:
                stacked_masks = [torch.stack(x) for x in agent_data.neighbors_mask[edge_type]]
                node_edges_encoded_mask.append(torch.stack(stacked_masks, dim=0).to(self.device))
            else:
                node_edges_encoded_mask.append(
                    torch.ones_like(encoded_edges_type[..., 0], device=self.device)
                )
        node_edges_encoded = torch.cat(node_edges_encoded, dim=1)
        encoded_edges_mask = torch.cat(node_edges_encoded_mask, dim=1)
        combined_edges, _ = self.node_modules[self.node_type + "/edge_influence_encoder"](
            encoder_states=node_edges_encoded,
            decoder_state=node_history_encoded,
            mask=encoded_edges_mask.detach(),
        )
        return F.dropout(
            combined_edges,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == model.model_utils.ModeKeys.TRAIN),
        )

    def encode_map(self, mode, map):
        encoded_map = self.node_modules[self.node_type + "/map_encoder"](
            map * 2.0 - 1.0, (mode == model.model_utils.ModeKeys.TRAIN)
        )
        do = self.hyperparams["map_encoder"][self.node_type]["dropout"]
        return F.dropout(encoded_map, do, training=(mode == model.model_utils.ModeKeys.TRAIN))

    def obtain_encoded_tensors(self, mode, agent_data):
        """Encodes input and output tensors of the target node.

        The input tensors contains the history trajectory of the target agent and all neighbors' trajectories.
        The output tensor contains the future trajectory of the target agent, which is only needed in training mode.

            Args:
                mode: Mode in which the model is operated. E.g. Train, or Predict.
                agent_data: The data object of the target agent.

            Return:
                encoded_input: A feature vector encoding both the history and neighbors.
                encoded_future: A feature vector encoding future trajectory of the target.
        """
        node_history_encoded = self.encode_node_history(
            mode, agent_data.history_standardized, agent_data.first_history_indices
        )
        encoded_concat_list = list()
        if self.check_if_hyperparameter_exists("edge_encoding"):
            total_edge_influence = self.encode_neighbors(mode, agent_data, node_history_encoded)
            encoded_concat_list.append(total_edge_influence)
        encoded_concat_list.append(node_history_encoded)
        if (
            self.check_if_hyperparameter_exists("use_map_encoding")
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            encoded_concat_list.append(self.encode_map(mode, agent_data.map))

        encoded_input = torch.cat(encoded_concat_list, dim=1)

        encoded_future = None
        if mode == model.model_utils.ModeKeys.TRAIN:
            encoded_future = self.encode_node_future(
                mode, agent_data.history_standardized[:, -1], agent_data.future_standardized
            )
        return encoded_input, encoded_future

    def loss(self, agent_data, prediction_horizon):
        """Calculates the training loss for a batch.

        The loss function is a combination of the negative log-likelihood, kl-divergence and infoVAE mutual information.

            Args:
                agent_data: The data object of the target agent.
                prediction_horizon: Number of timesteps to predict in the future.

            Return:
                A scalar tensor of the training loss.
        """
        mode = model.model_utils.ModeKeys.TRAIN
        self.initialize_dynamics(agent_data.history)

        encoded_input, encoded_future = self.obtain_encoded_tensors(mode, agent_data)
        latent_samples, kl_divergence = self.encoder(mode, encoded_input, encoded_future)

        num_samples = self.hyperparams["k"]
        num_components = self.hyperparams["N"] * self.hyperparams["K"]
        (
            initial_state,
            initial_prediction,
            encoded_condition,
        ) = self.decode_initial_state_and_prediction(
            encoded_input,
            latent_samples,
            agent_data.history_standardized[:, -1],
            num_samples,
            num_components,
        )
        lstm_cell = self.node_modules[self.node_type + "/decoder/rnn_cell"]
        single_timestep_decoder = SingleTimeStepDecoder(self.project_to_GMM_params, lstm_cell)
        predicted_gmm = self.decode_future_trajectory(
            mode,
            initial_state,
            initial_prediction,
            encoded_condition,
            single_timestep_decoder,
            prediction_horizon,
            self.hyperparams["k"],
            num_components,
        )
        if self.hyperparams["dynamic"][self.node_type]["distribution"]:
            predicted_gmm = self.dynamic.integrate_distribution(predicted_gmm, encoded_input)

        log_likelihood = torch.clamp(
            predicted_gmm.log_prob(agent_data.future), max=self.hyperparams["log_p_yt_xz_max"]
        )
        log_likelihood = torch.sum(log_likelihood, dim=2)
        log_likelihood = torch.mean(log_likelihood)
        mutual_inf_p = model.model_utils.mutual_inf_mc(self.latent.p_dist)

        return -(log_likelihood - self.kl_weight * kl_divergence + 1.0 * mutual_inf_p)

    def predict(
        self, agent_data, prediction_horizon, prior_latent_sampler, gmm_mode=False, resolution=0.1
    ):
        """
        Predicts the future of a batch of nodes.

            Args:
                agent_data: The data object of the target agent.
                prediction_horizon: Number of timesteps to predict in the future.
                prior_latent_sampler: Prior latent sampler.
                gmm_mode: If True, the most likelihood sample of the predicted GMM is computed.
                resolution: The resolution of the grid when estimating the most likelihood sample.

            Return:
                predicted_gmm: GMM2D object for all parameters of the mixture model of the predicted future trajectory.
                predicted_samples: Multiple predicted trajectories of the target agent.
        """
        mode = model.model_utils.ModeKeys.PREDICT
        self.initialize_dynamics(agent_data.history)
        encoded_input, encoded_future = self.obtain_encoded_tensors(mode, agent_data)
        self.latent.p_dist = self.p_z_x(mode, encoded_input)
        latent_samples, num_samples, num_components = prior_latent_sampler.run(mode=mode)

        (
            initial_state,
            initial_prediction,
            encoded_condition,
        ) = self.decode_initial_state_and_prediction(
            encoded_input,
            latent_samples,
            agent_data.history_standardized[:, -1],
            num_samples,
            num_components,
        )
        lstm_cell = self.node_modules[self.node_type + "/decoder/rnn_cell"]
        single_timestep_decoder = SingleTimeStepDecoder(
            self.project_to_GMM_params, lstm_cell, gmm_mode=gmm_mode, resolution=resolution
        )
        predicted_gmm = self.decode_future_trajectory(
            mode,
            initial_state,
            initial_prediction,
            encoded_condition,
            single_timestep_decoder,
            prediction_horizon,
            num_samples,
            num_components,
        )
        if gmm_mode:
            predicted_samples = predicted_gmm.mode(resolution=resolution)
        else:
            predicted_samples = predicted_gmm.rsample()
        predicted_samples = self.dynamic.integrate_samples(predicted_samples, encoded_input)

        if self.hyperparams["dynamic"][self.node_type]["distribution"]:
            predicted_gmm = self.dynamic.integrate_distribution(predicted_gmm, encoded_input)

        return predicted_gmm, predicted_samples


class ExplainableTrajectron(model.Trajectron):
    """Represents the full framework containing a separate model for every node type."""

    def __init__(self, model_registrar, hyperparams, log_writer, device):
        super().__init__(model_registrar, hyperparams, log_writer, device)

    def set_environment(self, environment):
        """Create the models for all node types of an environment.

        Args:
            environment: The environment object containing all nodes.
        """
        self.environment = environment
        self.node_models_dict.clear()
        edge_types = environment.get_edge_types()
        for node_type in environment.NodeType:
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = ExplainableMultimodalGenerativeCVAE(
                    environment,
                    node_type,
                    self.model_registrar,
                    self.hyperparams,
                    self.device,
                    edge_types,
                    log_writer=self.log_writer,
                )

    def loss(self, batch, node_type):
        """Compute the training loss for a single batch given the model of a specific node type.

        Args:
            batch: A list of input/output data.
            node_type: The node type of that batch (e.g, PEDESTRIAN, VEHICLE, etc).

        Returns:
            The training loss as scalar.
        """
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
        ) = batch

        history = history.to(self.device)
        future = future.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        history_standardized = history_standardized.to(self.device)
        future_standardized = future_standardized.to(self.device)
        agent_data = AgentData(
            history,
            history_standardized,
            first_history_indices,
            third_party_dataset.restore(neighbors),
            third_party_dataset.restore(neighbors_standardized),
            third_party_dataset.restore(neighbors_edge_value),
            third_party_dataset.restore(neighbors_mask),
            third_party_dataset.restore(neighbors_nodes),
            future,
            future_standardized,
            map,
        )
        model = self.node_models_dict[node_type]
        return model.loss(agent_data, self.ph)

    def predict(
        self,
        scene,
        timesteps,
        prediction_horizon,
        prior_latent_sampler,
        min_future_timesteps=0,
        min_history_timesteps=1,
        gmm_mode=False,
        resolution=0.1,
        target_node_type=None,
        without_neighbors=False,
    ):
        """Predict the future for all nodes in a specific scene and within time range.

        Args:
            scene: The scene from where we get all nodes.
            timesteps: A range of the time interval for the nodes to be selected.
            prediction_horizon: Number of timesteps to predict in the future.
            prior_latent_sampler: Prior latent sampler.
            min_future_timesteps: The minimum future timesteps for the nodes to be selected.
            min_history_timesteps: The minimum history timesteps for the nodes to be selected.
            gmm_mode: If True, the most likelihood sample of the predicted GMM is computed.
            resolution: The resolution of the grid when estimating the most likelihood sample.
            target_node_type: Predict only for this target node type.
            without_neighbors: Ignore all neighbors when predicting.

        Returns:
            The training loss as scalar.
        """
        predictions_dict = {}
        for node_type in self.environment.NodeType:
            if target_node_type != node_type or node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]
            prior_latent_sampler.latent = model.latent
            data_parameters = dataset.DataParameters(
                self.state,
                self.pred_state,
                min_history_timesteps,
                self.max_ht,
                min_future_timesteps,
                model.edge_types,
                self.hyperparams,
            )
            batch = dataset.get_scene_data(
                self.environment, scene, timesteps, node_type, data_parameters
            )
            if batch is None:
                continue
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

            history = history.to(self.device)
            history_standardized = history_standardized.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)
            if without_neighbors:
                agent_data = AgentData(
                    history,
                    history_standardized,
                    first_history_indices,
                    None,
                    None,
                    None,
                    None,
                    None,
                    map=map,
                )
            else:
                agent_data = AgentData(
                    history,
                    history_standardized,
                    first_history_indices,
                    neighbors,
                    neighbors_standardized,
                    neighbors_edge_value,
                    neighbors_mask,
                    neighbors_nodes,
                    map=map,
                )

            # predict multiple trajectories
            _, predicted_samples = model.predict(
                agent_data, prediction_horizon, prior_latent_sampler, gmm_mode, resolution
            )
            predicted_samples_array = predicted_samples.cpu().detach().numpy()

            # predict the full multimodal distribution
            prior_latent_sampler.num_samples = 1
            prior_latent_sampler.full_dist = True
            predicted_gmm, _ = model.predict(
                agent_data, prediction_horizon, prior_latent_sampler, True, resolution
            )

            for i, node in enumerate(nodes):
                predictions_dict[node] = {
                    "samples": predicted_samples_array[:, [i]],
                    "GMM": predicted_gmm.get_for_node(i),
                }
        return predictions_dict
