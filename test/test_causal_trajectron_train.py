# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import types

from explainable_trajectory_prediction.trajectron import train


def test_trainer(tmp_path):
    """Tests if model training saves a model to disk."""
    parameters = types.SimpleNamespace(
        device="cpu",
        preprocess_workers=0,
        data_dir="test/data",
        train_data_dict="test.pkl",
        log_dir=tmp_path,
        log_tag="",
        train_epochs=1,
        save_every=1,
        seed=123,
    )
    with open("test/data/config.json", "r", encoding="utf-8") as config_json:
        model_parameters = json.load(config_json)
    trainer = train.Trainer(parameters, model_parameters)
    trainer.fit()
    assert os.path.exists("%s/model_registrar-1.pt" % trainer.model_directory)
