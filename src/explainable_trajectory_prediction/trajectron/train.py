# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trainer class of Trajectron++ framework.

    Some of the functions are based on the training script under
    thirdparty/Trajectron_plus_plus/trajectron/train.py
    of the following repository:
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage."""
import copy
import json
import os
import pathlib
import time

import dill
import model.dataset as third_party_dataset
import model.model_registrar as model_registrar
import numpy as np
import tensorboardX
import torch
import tqdm

from explainable_trajectory_prediction.trajectron import network
from explainable_trajectory_prediction.trajectron import dataset
from explainable_trajectory_prediction.trajectron import utilities


class Trainer(object):
    def __init__(self, train_parameters, model_parameters):
        """Initialize the trainer class.

        Args:
            train_parameters: The parsed argument from the command line.
            model_parameters: A dictionary of model-specific parameters
        """
        self.train_parameters = train_parameters
        self.model_parameters = model_parameters
        self._initialize_device_and_seed()
        self._create_model_directory_and_log()
        self.log_writer = tensorboardX.SummaryWriter(log_dir=self.model_directory)
        self._load_train_environment()
        self._create_train_dataset_loader()
        self._calculate_scene_graph()
        self.model_registrar = model_registrar.ModelRegistrar(
            self.model_directory, self.train_parameters.device
        )
        self._create_model()
        self._create_optimizer_and_scheduler()

    def _initialize_device_and_seed(self):
        """Initialize the device used for training and set the random seed."""
        if not torch.cuda.is_available():
            self.train_parameters.device = torch.device("cpu")

        if self.train_parameters.seed:
            utilities.initialize_device_and_seed(self.train_parameters.seed)

    def _create_model_directory_and_log(self):
        """Create the directory to store the trained models and write its config file."""
        self.model_directory = os.path.join(
            self.train_parameters.log_dir,
            "models_"
            + time.strftime("%d_%b_%Y_%H_%M_%S_", time.localtime())
            + self.train_parameters.log_tag,
        )
        pathlib.Path(self.model_directory).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.model_directory, "config.json"), "w") as config_json:
            json.dump(self.model_parameters, config_json)

    def _override_radius(self):
        if "override_attention_radius" in self.model_parameters:
            utilities.override_radius_parameters(
                self.train_environment.attention_radius,
                self.model_parameters["override_attention_radius"],
            )
        self.train_environment.neighbors_radius = copy.deepcopy(
            self.train_environment.attention_radius
        )
        if "override_neighbors_radius" in self.model_parameters:
            utilities.override_radius_parameters(
                self.train_environment.neighbors_radius,
                self.model_parameters["override_neighbors_radius"],
            )

    def _load_train_environment(self):
        """Load the training environment containing the dataset from a pickle file."""
        train_data_path = os.path.join(
            self.train_parameters.data_dir, self.train_parameters.train_data_dict
        )
        with open(train_data_path, "rb") as file:
            self.train_environment = dill.load(file, encoding="latin1")
        self._override_radius()

    def _create_train_dataset_loader(self):
        """Create a torch dataset loaded for the given environment."""
        train_dataset = dataset.EnvironmentDataset(
            self.train_environment,
            self.model_parameters["state"],
            self.model_parameters["pred_state"],
            scene_freq_mult=self.model_parameters["scene_freq_mult_train"],
            node_freq_mult=self.model_parameters["node_freq_mult_train"],
            hyperparams=self.model_parameters,
            min_history_timesteps=self.model_parameters["minimum_history_length"],
            min_future_timesteps=self.model_parameters["prediction_horizon"],
            return_robot=True,
        )
        train_dataset.augment = self.model_parameters["augment"]
        self.train_data_loader = dict()
        for node_type_data_set in train_dataset:
            if len(node_type_data_set) == 0:
                continue
            node_type_dataloader = torch.utils.data.DataLoader(
                node_type_data_set,
                collate_fn=third_party_dataset.collate,
                pin_memory=self.train_parameters.device != "cpu",
                batch_size=self.model_parameters["batch_size"],
                shuffle=True,
                num_workers=self.train_parameters.preprocess_workers,
            )
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    def _calculate_scene_graph(self):
        """Calculate the scene graph of the dataset offline."""
        if self.model_parameters["offline_scene_graph"] == "yes":
            for i, scene in enumerate(self.train_environment.scenes):
                scene.calculate_scene_graph(
                    self.train_environment.neighbors_radius,
                    self.model_parameters["edge_addition_filter"],
                    self.model_parameters["edge_removal_filter"],
                )

    def _create_model(self):
        """Create the trajectron model for training."""
        self.trajectron = network.ExplainableTrajectron(
            self.model_registrar,
            self.model_parameters,
            self.log_writer,
            self.train_parameters.device,
        )
        self.trajectron.set_environment(self.train_environment)
        self.trajectron.set_annealing_params()

    def _create_optimizer_and_scheduler(self):
        """Create the torch optimizer and learning rate scheduler."""
        self.optimizer = dict()
        self.lr_scheduler = dict()
        for node_type in self.train_environment.NodeType:
            if node_type not in self.model_parameters["pred_state"]:
                continue
            self.optimizer[node_type] = torch.optim.Adam(
                [
                    {
                        "params": self.model_registrar.get_all_but_name_match(
                            "map_encoder"
                        ).parameters()
                    },
                    {
                        "params": self.model_registrar.get_name_match("map_encoder").parameters(),
                        "lr": 0.0008,
                    },
                ],
                lr=self.model_parameters["learning_rate"],
            )
            if self.model_parameters["learning_rate_style"] == "const":
                self.lr_scheduler[node_type] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer[node_type], gamma=1.0
                )
            elif self.model_parameters["learning_rate_style"] == "exp":
                self.lr_scheduler[node_type] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer[node_type], gamma=self.model_parameters["learning_decay_rate"]
                )

    def fit_batch(self, batch, current_iteration, node_type):
        """Train for a single batch.

        Args:
            batch: The data batch.
            current_iteration: The current iteration of the training.
            node_type: The node type of the training model.
        """
        self.trajectron.set_curr_iter(current_iteration)
        self.trajectron.step_annealers(node_type)
        self.optimizer[node_type].zero_grad()
        train_loss = self.trajectron.loss(batch, node_type)
        train_loss.backward()

        if self.model_parameters["grad_clip"] is not None:
            torch.nn.utils.clip_grad_value_(
                self.model_registrar.parameters(), self.model_parameters["grad_clip"]
            )
        self.optimizer[node_type].step()
        self.lr_scheduler[node_type].step()

        self.log_writer.add_scalar(
            f"{node_type}/train/learning_rate",
            self.lr_scheduler[node_type].get_last_lr()[0],
            current_iteration,
        )
        self.log_writer.add_scalar(f"{node_type}/train/loss", train_loss, current_iteration)
        return train_loss.item()

    def fit(self):
        curr_iter_node_type = {node_type: 0 for node_type in self.train_data_loader.keys()}
        for epoch in range(self.train_parameters.train_epochs):
            for node_type, data_loader in self.train_data_loader.items():
                current_iteration = curr_iter_node_type[node_type]
                progress_bar = tqdm.tqdm(data_loader, ncols=80)
                for batch in progress_bar:
                    train_loss = self.fit_batch(batch, current_iteration, node_type)
                    progress_bar.set_description(f"Epoch {epoch}, {node_type} L: {train_loss:.2f}")
                    current_iteration += 1
                curr_iter_node_type[node_type] = current_iteration

            if (epoch + 1) % self.train_parameters.save_every == 0:
                self.model_registrar.save_models(epoch + 1)
