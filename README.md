# Explainable Trajectory Prediction

This repository contains the code for explainable trajectory prediction based on Shapley values.


### Data Preparation
The processing code/files of the ETH-UCY, nuScenes and SDD can be found at:

- ETH-UCY, SDD for Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus/blob/master/experiments/pedestrians/process_data.py

- nuScenes for Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus/blob/master/experiments/nuScenes/process_data.py

- SDD for PECNet: https://github.com/HarshayuGirase/PECNet/tree/master/social_pool_data

To process the SportVU dataset for the two frameworks:

```
python tools/prepare_data.py
python tools/process_trajectron_data.py
python tools/process_pecnet_data.py
```

The raw data can be accessed at:
https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs

### Thirdparty
Download the following frameworks and place them under:

- thirdparty/Trajectron-plus-plus: https://github.com/StanfordASL/Trajectron-plus-plus
- thidparty/PECNet: https://github.com/HarshayuGirase/PECNet

### Pre-trained Models
All pre-trained models are provided under `models/`.

### Models Testing
Examples to test models:
``` 
# Testing Traj++Edge on ETH
env PYTHONPATH=src python bin/test_trajectron.py models/trajectron/eth-ucy/eth_edge 100 data/eth-ucy/eth_test.pkl cpu 
# Testing PECNet on SDD
env PYTHONPATH=src python bin/test_pecnet.py thirdparty/PECNet/saved_models/PECNET_social_model1.pt thirdparty/PECNet/social_pool_data/test_all_4096_0_100.pickle
```

To test the models without interaction, append ```--without_neighbours``` to the call.

### Model Training
An example call to train Trajectron++Edge:
``` 
# Training Traj++Edge on SportVU 
env PYTHONPATH=src python bin/train_trajectron.py --config models/trajectron/sport/edge/config.json --data_dir data/sport --train_data_dict trajectron_train.pkl --log_dir logs --log_tag sport_traj++Edge --train_epochs 20 --save_every 5 --deeper_action --late_fusion
```

### Shapley Values Estimation
Example calls to compute Shapley values:

``` 
# Shapley values of Traj++Edge on the first scene of SportVU
env PYTHONPATH=src python bin/compute_shapley_values_trajectron.py models/trajectron/sport/edge 20 data/sport/trajectron_test.pkl cuda:0 home nll zero 0 results/sport/Traj++Edge --random_node_types home guest

# Shapley values of PECNet on the first scene of SportVU
env PYTHONPATH=src python bin/compute_shapley_values_pecnet.py models/pecnet/sport.pt data/sport/pecnet_test.pkl 0 results/sport/pecnet
```
The above calls estimate the Shapley values per scene, to merge the results over all scenes of a dataset:

``` 
python tools/merge_results.py results/sport Traj++Edge nll results
```

### Plotting
An example call to plot the aggregated Shapley values by comparing two models:

``` 
python tools/plot_shapley_values.py --names Traj++ Traj++Edge --paths results/Traj++_nll.pkl results/Traj++Edge_nll.pkl --output_path results/SportVU_nll.png
```

To plot the local analysis (per scenario) of the SportVU:

``` 
python tools/plot_scenarios.py results/sport/Traj++Edge_home_0_nll.pkl results/court.png
```

where the second argument point to the image of the scene and can be accessed at:
https://github.com/linouk23/NBA-Player-Movements/

### Unit Tests

The following script checks a set of unit tests.

```
./local_run_tests.sh 
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.