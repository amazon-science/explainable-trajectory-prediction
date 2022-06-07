"""Merge the Shapley values results into a single file for visualization."""
import argparse
import glob
import os

import dill


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("model", help="model name", type=str)
    parser.add_argument("metric", help="metric name", type=str)
    parser.add_argument("output_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    parameters = parse_arguments()
    files = glob.glob(
        parameters.input_path + "/%s_*_%s.pkl" % (parameters.model, parameters.metric)
    )
    result = {}
    for file in files:
        _, node_type, scene_index, _ = os.path.basename(file).split("_")
        if scene_index not in result.keys():
            result[scene_index] = {}
        with open(file, "rb") as file_reader:
            data = dill.load(file_reader)
            result[scene_index][node_type] = data
    output_file_path = os.path.join(
        parameters.output_path, "%s_%s.pkl" % (parameters.model, parameters.metric)
    )
    if os.path.exists(output_file_path):
        raise FileExistsError(f"Output file already exists {output_file_path}.")
    with open(output_file_path, "wb") as file_writer:
        dill.dump(result, file_writer)
