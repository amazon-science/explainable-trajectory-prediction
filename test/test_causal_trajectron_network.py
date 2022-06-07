# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import dill
import model.model_registrar
import torch

from explainable_trajectory_prediction.trajectron import network


HISTORY_LENGTH = 8
PREDICTION_HORIZON = 12
STATE_LENGTH = 6
PREDICTION_STATE_LENGTH = 2


def _load_environment():
    configuration_file = "test/data/config.json"
    with open(configuration_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)
    train_data_path = "test/data/test.pkl"
    with open(train_data_path, "rb") as f:
        data_environment = dill.load(f, encoding="latin1")
    return hyperparams, data_environment


def _get_dummy_agent_data(batch_size, num_neighbors, node_type):
    history = torch.zeros((batch_size, HISTORY_LENGTH, STATE_LENGTH))
    history_st = torch.zeros_like(history)
    first_history_indices = torch.tensor([0 for _ in range(batch_size)])
    neighbors = {
        (node_type, node_type): [
            [torch.zeros((HISTORY_LENGTH, STATE_LENGTH)) for _ in range(num_neighbors)]
            for _ in range(batch_size)
        ]
    }
    neighbors_mask = {
        (node_type, node_type): [
            [torch.tensor(1.0) for _ in range(num_neighbors)]
            for _ in range(batch_size)
        ]
    }
    neighbors_edge_value = {
        (node_type, node_type): [torch.tensor([1.0]) for _ in range(batch_size)]
    }
    neighbors_nodes = {(node_type, node_type): [None for _ in range(batch_size)]}
    future = torch.zeros((batch_size, PREDICTION_HORIZON, PREDICTION_STATE_LENGTH))
    future_standardized = torch.zeros((batch_size, PREDICTION_HORIZON, PREDICTION_STATE_LENGTH))
    return network.AgentData(
        history,
        history_st,
        first_history_indices,
        neighbors,
        neighbors,
        neighbors_edge_value,
        neighbors_mask,
        neighbors_nodes,
        future,
        future_standardized,
    )


def test_model_init():
    hyperparams, data_environment = _load_environment()
    node_type = data_environment.NodeType[0]
    edge_types = data_environment.get_edge_types()
    device = "cpu"
    model_registrar = model.model_registrar.ModelRegistrar(model_dir="test/data", device=device)
    assert (
        network.ExplainableMultimodalGenerativeCVAE(
            data_environment, node_type, model_registrar, hyperparams, device, edge_types
        )
        is not None
    )


def test_model_loss():
    torch.set_default_dtype(torch.float32)
    hyperparams, data_environment = _load_environment()
    node_type = data_environment.NodeType[0]
    agent_data = _get_dummy_agent_data(256, 2, node_type)
    edge_types = data_environment.get_edge_types()
    model_registrar = model.model_registrar.ModelRegistrar(model_dir="test/data", device="cpu")
    explainable_network = network.ExplainableMultimodalGenerativeCVAE(
        data_environment, node_type, model_registrar, hyperparams, "cpu", edge_types
    )
    explainable_network.set_annealing_params()
    assert explainable_network.loss(agent_data, 12) is not None


def test_model_predict():
    hyperparams, data_environment = _load_environment()
    node_type = data_environment.NodeType[0]
    agent_data = _get_dummy_agent_data(1, 2, node_type)
    edge_types = data_environment.get_edge_types()
    model_registrar = model.model_registrar.ModelRegistrar(model_dir="test/data", device="cpu")
    explainable_network = network.ExplainableMultimodalGenerativeCVAE(
        data_environment, node_type, model_registrar, hyperparams, "cpu", edge_types
    )
    prior_sampler = network.PriorSampler(latent=explainable_network.latent, num_samples=1)
    assert explainable_network.predict(agent_data, 12, prior_sampler) is not None
