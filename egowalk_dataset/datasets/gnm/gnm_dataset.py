import torch
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional, Callable, Literal, Tuple, List
from egowalk_dataset.util.video import decode_frame
from egowalk_dataset.misc.indexing import IndependentSequence
from egowalk_dataset.misc.constants import (DEFAULT_DATA_PATH,
                                            BASE_RGB_DIR,
                                            BASE_DEPTH_DIR,
                                            BASE_VIDEO_DIR,
                                            RGB_VIDEO_EXTENSION,
                                            DEPTH_VIDEO_EXTENSION)


@dataclass
class GNMTuple:
    trajectory_name: str
    obs_idxs: List[int]
    goal_idx: Optional[int]
    action: List[float]
    goal_bbox: Optional[Tuple[float, float, float, float]]
    goal_caption: Optional[str]


class GNMFeature(ABC):

    def __init__(self,
                 name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> Any:
        pass


class GNMImageFeature(GNMFeature):

    def __init__(self,
                 name: str,
                 fmt: Literal["rgb", "depth"],
                 field: Literal["obs", "goal"],
                 indices: Optional[Union[Tuple[int, ...], int]] = None,
                 transform: Optional[Callable[[np.ndarray],
                                              Union[np.ndarray, torch.Tensor]]] = None):
        assert fmt in ("rgb", "depth"), f"Invalid format {fmt}"
        super(GNMImageFeature, self).__init__(name)
        self._fmt = fmt
        self._field = field
        self._indices = indices
        self._transform = transform

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> Union[np.ndarray, torch.Tensor]:
        traj_name = gnm_tuple.trajectory_name
        if self._fmt == "rgb":
            video = root / BASE_VIDEO_DIR / BASE_RGB_DIR / \
                f"{traj_name}__rgb.{RGB_VIDEO_EXTENSION}"
        else:
            video = root / BASE_VIDEO_DIR / BASE_DEPTH_DIR / \
                f"{traj_name}__depth.{DEPTH_VIDEO_EXTENSION}"

        if self._field == "obs":
            video_idxs = gnm_tuple.obs_idxs
            if self._indices is not None:
                if isinstance(self._indices, int):
                    video_idxs = [video_idxs[self._indices]]
                    single = True
                else:
                    video_idxs = [video_idxs[i] for i in self._indices]
                    single = False
            else:
                single = False
        else:
            video_idxs = [gnm_tuple.goal_idx]
            single = True

        frames = decode_frame(container=video,
                              frame_idx=IndependentSequence(video_idxs),
                              fmt=self._fmt)

        if self._transform is not None:
            frames = [self._transform(frame) for frame in frames]

        if not single:
            if isinstance(frames[0], np.ndarray):
                frames = np.stack(frames, axis=0)
            elif isinstance(frames[0], torch.Tensor):
                frames = torch.stack(frames, dim=0)
            return frames
        else:
            return frames[0]


class GNMRGBFeature(GNMImageFeature):

    def __init__(self,
                 name: str,
                 field: Literal["obs", "goal"],
                 indices: Optional[Union[Tuple[int, ...], int]] = None,
                 transform: Optional[Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]] = None):
        super(GNMRGBFeature, self).__init__(
            name, "rgb", field, indices, transform)


class GNMDepthFeature(GNMImageFeature):

    def __init__(self,
                 name: str,
                 field: Literal["obs", "goal"],
                 indices: Optional[Union[Tuple[int, ...], int]] = None,
                 transform: Optional[Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]] = None):
        super(GNMDepthFeature, self).__init__(
            name, "depth", field, indices, transform)


class GNMWaypointFeature(GNMFeature):

    def __init__(self,
                 name: str,
                 angle_format: Literal["none", "yaw", "sincos"] = "none",
                 return_tensors: Literal["np", "pt"] = "pt"):
        super(GNMWaypointFeature, self).__init__(name)
        self._angle_format = angle_format
        self._return_tensors = return_tensors

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> Any:
        actions = np.array(gnm_tuple.action)
        waypoints = actions[:, :2]
        if self._angle_format == "none":
            pass
        elif self._angle_format == "yaw":
            angles = actions[:, 2]
            waypoints = np.concatenate([waypoints, angles[:, None]], axis=1)
        elif self._angle_format == "sincos":
            angles = actions[:, 2]
            sines = np.sin(angles)
            cosines = np.cos(angles)
            waypoints = np.concatenate([waypoints, sines[:, None],
                                        cosines[:, None]], axis=1)
        else:
            raise ValueError(f"Invalid angle format {self._angle_format}")

        if self._return_tensors == "np":
            return waypoints
        elif self._return_tensors == "pt":
            return torch.from_numpy(waypoints).float()
        else:
            raise ValueError(f"Invalid return tensors {self._return_tensors}")


class GNMCaptionFeature(GNMFeature):

    def __init__(self,
                 name: str,
                 transform: Optional[Callable[[str], 
                                              Union[np.ndarray, 
                                                    torch.Tensor]]] = None):
        super(GNMCaptionFeature, self).__init__(name)
        self._transform = transform

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> Any:
        caption = gnm_tuple.goal_caption
        if self._transform is not None:
            caption = self._transform(caption)
        return caption


class GNMBBoxFeature(GNMFeature):

    def __init__(self,
                 name: str,
                 return_tensors: Literal["np", "pt"] = "pt"):
        super(GNMBBoxFeature, self).__init__(name)
        self._return_tensors = return_tensors

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> Any:
        bbox = np.array(gnm_tuple.goal_bbox)
        if self._return_tensors == "np":
            return bbox
        elif self._return_tensors == "pt":
            return torch.from_numpy(bbox).float()
        else:
            raise ValueError(f"Invalid return tensors {self._return_tensors}")


class GNMDataset(torch.utils.data.Dataset):

    def __init__(self,
                 index: Dict[str, Any],
                 features: List[GNMFeature],
                 data_path: Union[str, Path] = DEFAULT_DATA_PATH):
        super(GNMDataset, self).__init__()
        self._index = index
        self._features = features
        self._root = Path(data_path)

    def __len__(self):
        return len(self._index["trajectory"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gnm_tuple_kwargs = {}

        current_obs_idx = self._index["obs_idxs"][idx][-1]
        action = self._index["action"][idx]

        if "goal_idx" in self._index:
            # Assume it is the vision-only GNM dataset
            goal_idx = self._index["goal_idx"][idx]
            if goal_idx[0] == goal_idx[1]:
                goal_idx = goal_idx[0]
            else:
                goal_idx = np.random.randint(goal_idx[0], goal_idx[1] + 1)
            gnm_tuple_kwargs["goal_idx"] = goal_idx
            gnm_tuple_kwargs["goal_caption"] = None
            gnm_tuple_kwargs["goal_bbox"] = None

            action_length = len(action)
            if action_length > (goal_idx - current_obs_idx):
                action = action[:goal_idx - current_obs_idx]
                action = action + [action[-1]] * \
                    (action_length - (goal_idx - current_obs_idx))

        else:
            # Assume it is the text-only GNM dataset
            gnm_tuple_kwargs["goal_caption"] = self._index["goal_caption"][idx]
            gnm_tuple_kwargs["goal_bbox"] = self._index["goal_bbox"][idx]
            gnm_tuple_kwargs["goal_idx"] = None

        gnm_tuple_kwargs["action"] = action
        gnm_tuple_kwargs["trajectory_name"] = self._index["trajectory"][idx]
        gnm_tuple_kwargs["obs_idxs"] = self._index["obs_idxs"][idx]

        gnm_tuple = GNMTuple(**gnm_tuple_kwargs)
        features = {feature.name: feature(self._root, gnm_tuple)
                    for feature in self._features}
        return features


class DefaultGNMDataset(GNMDataset):

    def __init__(self,
                 index: Dict[str, Any],
                 image_transform: Optional[Callable[[np.ndarray],
                                                    Union[np.ndarray, torch.Tensor]]] = None,
                 angle_format: Literal["none", "yaw", "sincos"] = "none",
                 root: Optional[Union[str, Path]] = None):
        obs_feature = GNMRGBFeature(name="obs",
                                    field="obs",
                                    transform=image_transform)
        goal_feature = GNMRGBFeature(name="goal",
                                     field="goal",
                                     transform=image_transform)
        action_feature = GNMWaypointFeature(name="action",
                                            angle_format=angle_format)
        features = [obs_feature, goal_feature, action_feature]
        super(DefaultGNMDataset, self).__init__(index, features, root)
