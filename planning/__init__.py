from .random import Random
from .ergodic_horizon import ErgodicHorizon


def get_planner(cfg, device):
    planner_cfg = cfg.planner
    if planner_cfg.type == "random":
        return Random(planner_cfg, device)
    elif planner_cfg.type == "ergodic":
        return ErgodicHorizon(planner_cfg, device)
    else:
        raise NotImplementedError
