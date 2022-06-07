# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import types

import pytest

from explainable_trajectory_prediction.trajectron import shapley_values
from explainable_trajectory_prediction.trajectron import test


@pytest.mark.parametrize("metric_name", ["ade", "nll"])
@pytest.mark.parametrize("variant_name", ["zero", "random"])
def test_shapley_values(metric_name, variant_name):
    """Tests the Shapley values estimation."""
    parameters = types.SimpleNamespace(
        device="cpu",
        model="models/trajectron/eth-ucy/eth_orig",
        checkpoint=100,
        data="test/data/test.pkl",
        node_type="PEDESTRIAN",
        random_node_types=["PEDESTRIAN"],
        scene_index=0,
        metric=shapley_values.create_metric(metric_name),
        variant=shapley_values.create_variant(variant_name),
    )
    tester = test.Tester(
        parameters.data,
        parameters.model,
        parameters.checkpoint,
        parameters.device,
    )
    shapley_values_estimator = shapley_values.ShapleyValues(
        tester, parameters.node_type, parameters.random_node_types, parameters.metric
    )
    test_scene = tester.environment.scenes[parameters.scene_index]
    get_replacement_trajectory = lambda edge_type: parameters.variant(
        tester.environment.scenes,
        shapley_values_estimator.data_parameters,
        tester.environment,
        edge_type,
    )
    results = shapley_values_estimator.run(test_scene, get_replacement_trajectory, 10)
    assert results is not None
    assert len(results.keys()) > 0
    for node_results in results.values():
        assert "past" in node_results[9]["shapley_values"].keys()
