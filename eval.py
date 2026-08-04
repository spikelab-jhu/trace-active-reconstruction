"""Evaluate rendering quality of saved Gaussian maps."""

import json
import os
import warnings

import hydra
import numpy as np
import torch

from mapping.gaussian_map import GaussianMap
from simulator import get_simulator
from utils.evaluation_tool import EvaluationTool


warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="./config", config_name="eval")
def main(cfg):
    experiment_path = os.path.join(
        cfg.experiment.output_dir,
        str(cfg.experiment.exp_id),
        cfg.scene.scene_name,
        cfg.planner.planner_name,
        str(cfg.experiment.run_id),
    )
    record_info_file = os.path.join(experiment_path, "map", "record_info.txt")
    if not os.path.exists(record_info_file):
        raise FileNotFoundError(f"Missing mission record: {record_info_file}")

    record_info = np.atleast_2d(np.loadtxt(record_info_file))
    if bool(cfg.get("eval_final_only", False)):
        record_info = record_info[-1:]

    id_list = record_info[:, 0]
    timer_list = record_info[:, 1]
    path_list = record_info[:, 2]
    map_list = []

    for map_id_raw in id_list:
        map_id = int(map_id_raw)
        map_file = os.path.join(experiment_path, "map", f"map_{map_id:03}.th")
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Missing Gaussian map: {map_file}")
        gaussian_map = GaussianMap(None, device)
        gaussian_map.load(map_file)
        map_list.append(gaussian_map)

    simulator = get_simulator(cfg)
    print("\n---------- start evaluation ----------")
    eval_tool = EvaluationTool(
        map_list,
        cfg.test_folder,
        experiment_path,
        device,
        simulator=simulator,
    )
    eval_result = eval_tool.eval()

    eval_result["step"] = id_list.tolist()
    eval_result["time"] = timer_list.tolist()
    eval_result["path_length"] = path_list.tolist()

    result_file = os.path.join(experiment_path, "final_result.json")
    result_data = {}
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as handle:
            result_data = json.load(handle)
    result_data.update(eval_result)
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(result_data, handle, indent=2)
        handle.write("\n")
    print(f"Saved metrics to {result_file}")


if __name__ == "__main__":
    main()
