import numpy as np
import pandas as pd

from typing import Optional, Union, List, Tuple, Literal
from pathlib import Path
from functools import partial
from egowalk_dataset.datasets.gnm.cutters import (AbstractTrajectoryCutter,
                                                  apply_cutter)
from egowalk_dataset.misc.constants import (DEFAULT_DATA_PATH,
                                            BASE_PARQUET_DIR)
from egowalk_dataset.datasets.trajectory.trajectory import EgoWalkTrajectory
from egowalk_dataset.util.parallel import do_parallel
from egowalk_dataset.util.math import to_relative_frame


def _get_gnm_tuple(obs_idx: int,
                   segment_timestamps: List[int],
                   segment_traj_bev: np.ndarray,
                   context_length: int,
                   goal_offset: Union[int, Tuple[int, int]],
                   goal_offset_mode: Literal["fixed", "sampled", "stochastic"],
                   action_length: int,
                   context_step: int,
                   action_step: int):
    if isinstance(goal_offset, int):
        goal_offset = (goal_offset, goal_offset)
    if (len(segment_timestamps) - obs_idx - 1) < goal_offset[0]:
        return None, None, None
    goal_offset_min = goal_offset[0]
    goal_offset_max = min(goal_offset[1], len(segment_timestamps) - obs_idx - 1)
    if goal_offset_mode == "fixed":
        result_goal_offset = (goal_offset_min, goal_offset_max)
    elif goal_offset_mode == "sampled":
        result_goal_offset = np.random.randint(goal_offset_min, 
                                               goal_offset_max + 1)
        result_goal_offset = (result_goal_offset, result_goal_offset)
    else:
        result_goal_offset = (goal_offset_min,
                              goal_offset_max)


    context_timestamps = [segment_timestamps[obs_idx]]
    for i in range(1, context_length + 1):
        idx = max(obs_idx - i * context_step, 0)
        context_timestamps.append(segment_timestamps[idx])
    context_timestamps.reverse()
    
    actions = [segment_traj_bev[obs_idx]]
    for i in range(1, action_length + 1):
        idx = min(obs_idx + i * action_step, len(segment_timestamps) - 1)
        actions.append(segment_traj_bev[idx])
    actions = np.stack(actions, axis=0)
    actions = to_relative_frame(actions)[1:, :]

    return context_timestamps, actions, result_goal_offset


def _filter_actions(actions: np.ndarray) -> np.ndarray:
    filtered_actions = actions.copy()
    for i in range(1, len(filtered_actions)):
        mask = np.isnan(filtered_actions[i])
        if np.any(mask):
            filtered_actions[i, mask] = filtered_actions[i-1, mask]
    for i in range(len(filtered_actions)-2, -1, -1):
        mask = np.isnan(filtered_actions[i])
        if np.any(mask):
            filtered_actions[i, mask] = filtered_actions[i+1, mask]
    return filtered_actions


def _index_single_traj(traj_name: Path,
                       root: Path,
                       cutters: List[AbstractTrajectoryCutter],
                       window_step: int,
                       context_length: int,
                       goal_offset: Union[int, Tuple[int, int]],
                       goal_offset_mode: Literal["fixed", "sampled", "stochastic"],
                       action_length: int,
                       context_step: int,
                       action_step: int) -> None:
    traj = EgoWalkTrajectory.from_dataset(name=traj_name,
                                          data_path=root)

    timestamps = traj.odometry.valid_timestamps
    traj_bev = traj.odometry.get_bev(filter_valid=True)

    result = {
        "trajectory": [],
        "obs_idxs": [],
        "goal_idx": [],
        "action": []
    }

    segments = apply_cutter(trajectory=traj_bev,
                            cutter=cutters)
    for segment in segments:
        segment_timestamps = timestamps[segment[0]:segment[1]]
        segment_traj_bev = traj_bev[segment[0]:segment[1]]
        for i in range(0, len(segment_timestamps), window_step):
            context_timestamps, actions, result_goal_offset = _get_gnm_tuple(obs_idx=i,
                                                                      segment_timestamps=segment_timestamps,
                                                                      segment_traj_bev=segment_traj_bev,
                                                                      context_length=context_length,
                                                                      goal_offset=goal_offset,
                                                                      goal_offset_mode=goal_offset_mode,
                                                                      action_length=action_length,
                                                                      context_step=context_step,
                                                                      action_step=action_step)
            if context_timestamps is None:
                continue
            
            context_idxs = [traj.rgb.timestamp_to_idx(t) for t in context_timestamps]
            goal_idx = (traj.rgb.timestamp_to_idx(segment_timestamps[i + result_goal_offset[0]]),
                        traj.rgb.timestamp_to_idx(segment_timestamps[i + result_goal_offset[1]]))
            actions = [[float(e[0]), float(e[1]), float(e[2])] for e in actions]

            result["trajectory"].append(traj_name)
            result["obs_idxs"].append(context_idxs)
            result["goal_idx"].append(goal_idx)
            result["action"].append(actions)
    
    return result


def _index_single_traj_text(traj_name: Path,
                       root: Path,
                       caption_type: str,
                       context_length: int,
                       action_length: int,
                       context_step: int,
                       action_step: int,
                       window_step: int,
                       n_window_steps: int) -> None:
    traj = EgoWalkTrajectory.from_dataset(name=traj_name,
                                          data_path=root)
    text_df = pd.read_parquet(root / "annotations" / caption_type / f"{traj_name}__annotations_{caption_type}.parquet")

    timestamps = traj.odometry.all_timestamps
    traj_bev = traj.odometry.get_bev(filter_valid=False)

    result = {
        "trajectory": [],
        "obs_idxs": [],
        "goal_caption": [],
        "goal_bbox": [],
        "action": []
    }

    for _, row in text_df.iterrows():
        traj_name = row["trajectory"]
        caption = row["caption"]
        frame_idx = row["frame"]
        bbox = (row["box_x"], row["box_y"], row["box_w"], row["box_h"])

        for i in range(0, n_window_steps * window_step, window_step):
            if frame_idx + i > len(timestamps) - 1:
                break
            context_timestamps, actions, _ = _get_gnm_tuple(obs_idx=frame_idx + i,
                                                                        segment_timestamps=timestamps,
                                                                        segment_traj_bev=traj_bev,
                                                                        context_length=context_length,
                                                                        goal_offset=0,
                                                                        goal_offset_mode="fixed",
                                                                        action_length=action_length,
                                                                        context_step=context_step,
                                                                        action_step=action_step)
            if np.isnan(actions).all():
                continue
            if np.isnan(actions).any():
                actions = _filter_actions(actions)

            context_idxs = [traj.rgb.timestamp_to_idx(t) for t in context_timestamps]
            actions = [[float(e[0]), float(e[1]), float(e[2])] for e in actions]

            result["trajectory"].append(traj_name)
            result["obs_idxs"].append(context_idxs)
            result["goal_caption"].append(caption)
            result["goal_bbox"].append(bbox)
            result["action"].append(actions)
        
    return result


def index_gnm(cutters: List[AbstractTrajectoryCutter],
              window_step: int,
              context_length: int,
              goal_offset: Union[int, Tuple[int, int]],
              goal_offset_mode: Literal["fixed", "sampled", "stochastic"],
              action_length: int,
              context_step: int = 1,
              action_step: int = 1,
              data_path: Union[str, Path] = DEFAULT_DATA_PATH,
              trajectories: Optional[List[str]] = None,
              n_workers: int = 0,
              use_tqdm: bool = True):
    root = Path(data_path)
    
    if trajectories is None:
        parquet_dir = root / BASE_PARQUET_DIR
        if not parquet_dir.exists():
            raise FileNotFoundError(f"No dataset found in {root}")
        traj_names = sorted([e.stem for e in parquet_dir.glob("*.parquet")])
        if len(traj_names) == 0:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    else:
        traj_names = trajectories
    
    task_fn = partial(_index_single_traj,
                      root=root,
                      cutters=cutters,
                      window_step=window_step,
                      context_length=context_length,
                      goal_offset=goal_offset,
                      goal_offset_mode=goal_offset_mode,
                      action_length=action_length,
                      context_step=context_step,
                      action_step=action_step)
    
    traj_results = do_parallel(task_fn,
                          traj_names,
                          n_workers=n_workers, 
                          use_tqdm=use_tqdm)
    traj_results = sorted(traj_results, key=lambda x: x["trajectory"])
    
    final_result = {k: [] for k in traj_results[0].keys()}
    for result in traj_results:
        for k, v in result.items():
            final_result[k].extend(v)
    
    return final_result


def index_gnm_text(caption_type: str,
              context_length: int,
              action_length: int,
              context_step: int = 1,
              action_step: int = 1,
              window_step: int = 1,
              n_window_steps: int = 1,
              data_path: Union[str, Path] = DEFAULT_DATA_PATH,
              trajectories: Optional[List[str]] = None,
              n_workers: int = 0,
              use_tqdm: bool = True):
    root = Path(data_path)
    
    if trajectories is None:
        parquet_dir = root / BASE_PARQUET_DIR
        if not parquet_dir.exists():
            raise FileNotFoundError(f"No dataset found in {root}")
        traj_names = sorted([e.stem for e in parquet_dir.glob("*.parquet")])
        if len(traj_names) == 0:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    else:
        traj_names = trajectories
    
    task_fn = partial(_index_single_traj_text,
                      caption_type=caption_type,
                      root=root,
                      context_length=context_length,
                      action_length=action_length,
                      context_step=context_step,
                      action_step=action_step,
                      window_step=window_step,
                      n_window_steps=n_window_steps)
    
    traj_results = do_parallel(task_fn,
                          traj_names,
                          n_workers=n_workers, 
                          use_tqdm=use_tqdm)
    traj_results = sorted(traj_results, key=lambda x: x["trajectory"])
    
    final_result = {k: [] for k in traj_results[0].keys()}
    for result in traj_results:
        for k, v in result.items():
            final_result[k].extend(v)
    
    return final_result
