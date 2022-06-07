import argparse
import dill
import matplotlib.pyplot as plt
import numpy as np

BAR_WIDTH = 0.12
DPI_RESOLUTION = 200
FONT_SIZE = 12


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--paths", nargs="+")
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def aggregate_local_contributions(shapley_values):
    result = []
    for key, value in shapley_values.items():
        if "past" not in key and "random" not in key:
            result.append(value)
    if result:
        return np.max(result)
    return 0


def get_contributions(files):
    result = [dict() for _ in range(len(files))]
    for file_index, file_path in enumerate(files):
        agent_contribution = {"past": [], "interaction": [], "random": []}
        with open(file_path, "rb") as file_reader:
            shapley_values = dill.load(file_reader)
            for sv_scene in shapley_values.values():
                for node_type, sv_node_type in sv_scene.items():
                    for sv_node in sv_node_type.values():
                        for sv_timestep in sv_node.values():
                            agent_contribution["interaction"].append(
                                aggregate_local_contributions(sv_timestep)
                            )
                            agent_contribution["past"].append(sv_timestep["past"])
                            for node_key, node_value in sv_timestep.items():
                                if "random" in node_key:
                                    agent_contribution["random"].append(node_value)
        result[file_index]["Agent"] = agent_contribution
    return result


def plot_shapley_values(contributions, legend, output_path):
    num_models = len(contributions)
    node_types = contributions[0].keys()
    for index, node_type in enumerate(node_types):
        plt.figure(dpi=DPI_RESOLUTION)
        axis = plt.gca()
        labels = contributions[0][node_type].keys()
        labels_locations = np.arange(len(labels)) / 2.0
        for model_index in range(num_models):
            first_center = labels_locations - ((num_models - 1) / 2) * BAR_WIDTH
            axis.bar(
                first_center + model_index * BAR_WIDTH,
                [np.mean(x) for x in contributions[model_index][node_type].values()],
                BAR_WIDTH,
                label=legend[model_index],
            )
        axis.set_xticks(labels_locations)
        axis.set_xticklabels(labels, fontsize=FONT_SIZE)
        axis.legend()
        plt.savefig(output_path)


if __name__ == "__main__":
    parameters = parse_arguments()
    if len(parameters.names) != len(parameters.paths):
        raise ValueError("Argument names and paths should have the same length.")
    contributions = get_contributions(parameters.paths)
    plot_shapley_values(contributions, parameters.names, parameters.output_path)
