# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import types

from explainable_trajectory_prediction.pecnet import dataset
from explainable_trajectory_prediction.pecnet import test
from explainable_trajectory_prediction.trajectron import utilities


def test_tester():
    """Tests if model testing returns valid errors."""
    parameters = types.SimpleNamespace(
        load_file="thirdparty/PECNet/saved_models/PECNET_social_model1.pt",
        dataset_path="thirdparty/PECNet/social_pool_data/test_all_4096_0_100.pickle",
        number_of_runs=10,
        without_neighbours=False,
    )
    tester = test.Tester(utilities.get_device(), parameters.load_file)
    min_ade, min_fde = tester.run(
        parameters.dataset_path,
        parameters.number_of_runs,
        dataset.get_sdd_data,
        parameters.without_neighbours,
    )
    assert 9.0 <= min_ade <= 10.0
    assert 15.0 <= min_fde <= 16.5
