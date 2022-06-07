# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import types

import numpy as np

from explainable_trajectory_prediction.trajectron import test


def test_tester():
    """Tests if model testing returns valid errors."""
    parameters = types.SimpleNamespace(
        device="cpu",
        model="models/trajectron/eth-ucy/eth_orig",
        checkpoint=100,
        data="test/data/test.pkl",
        without_neighbours=False,
        resolution=0.01,
    )
    tester = test.Tester(
        parameters.data,
        parameters.model,
        parameters.checkpoint,
        parameters.device,
    )
    results = tester.run(parameters.without_neighbours, parameters.resolution)
    assert "PEDESTRIAN" in results.keys() and len(results.keys()) == 1
    for metric in ["ade", "fde", "kde", "nll"]:
        assert metric in results["PEDESTRIAN"].keys()
    assert len(results["PEDESTRIAN"].keys()) == 4
    assert 0.55 <= np.mean(results["PEDESTRIAN"]["ade"]) <= 0.65
    assert 2.0 <= np.mean(results["PEDESTRIAN"]["nll"]) <= 2.50
