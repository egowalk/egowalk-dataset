import numpy as np

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, List, Dict, Any
from pathlib import Path
from datasets import Dataset
from egowalk_dataset.misc.constants import (BASE_PARQUET_DIR,
                                            BASE_VIDEO_DIR,
                                            BASE_RGB_DIR,
                                            BASE_DEPTH_DIR,
                                            RGB_VIDEO_EXTENSION,
                                            DEPTH_VIDEO_EXTENSION)
from egowalk_dataset.util.video import decode_frame
from egowalk_dataset.datasets.trajectory.geometry import quaternion_to_yaw
from egowalk_dataset.datasets.base.base_metadata import EgoWalkBaseMetadata


T = TypeVar("T")


def prepare_rgb_files(trajectories: List[str],
                      files_list: List[str]) -> List[str]:
    for traj in trajectories:
        video_file = f"{BASE_VIDEO_DIR}/{BASE_RGB_DIR}/{traj}__rgb.{RGB_VIDEO_EXTENSION}"
        if video_file not in files_list:
            files_list.append(video_file)
    return files_list


def prepare_depth_files(trajectories: List[str],
                        files_list: List[str]) -> List[str]:
    for traj in trajectories:
        video_file = f"{BASE_VIDEO_DIR}/{BASE_DEPTH_DIR}/{traj}__depth.{DEPTH_VIDEO_EXTENSION}"
        if video_file not in files_list:
            files_list.append(video_file)
    return files_list


def prepare_parquet_files(trajectories: List[str],
                          files_list: List[str]) -> List[str]:
    for traj in trajectories:
        parquet_file = f"{BASE_PARQUET_DIR}/{traj}.parquet"
        if parquet_file not in files_list:
            files_list.append(parquet_file)
    return files_list


class AbstractBaseModality(ABC, Generic[T]):

    def __init__(self,
                 requires_rgb: bool,
                 requires_depth: bool):
        self._requires_rgb = requires_rgb
        self._requires_depth = requires_depth

    @property
    def requires_rgb(self) -> bool:
        return self._requires_rgb

    @property
    def requires_depth(self) -> bool:
        return self._requires_depth

    @property
    def prepare_files_list(self,
                           trajectories: List[str],
                           files_list: List[str]) -> List[str]:
        pass

    @abstractmethod
    def read(self,
             idx: int,
             root: Path,
             record: Dict[str, Any],
             hf_dataset: Dataset,
             metadata: EgoWalkBaseMetadata) -> T:
        pass


class RGBBaseModality(AbstractBaseModality[np.ndarray]):

    def __init__(self):
        super(RGBBaseModality, self).__init__(requires_rgb=True,
                                              requires_depth=False)

    def prepare_files_list(self,
                           trajectories: List[str],
                           files_list: List[str]) -> List[str]:
        return prepare_rgb_files(trajectories, files_list)

    def read(self,
             idx: int,
             root: Path,
             record: Dict[str, Any],
             hf_dataset: Dataset,
             metadata: EgoWalkBaseMetadata) -> np.ndarray:
        traj_name = record["trajectory"]
        frame = record["frame"]
        video_file = root / BASE_VIDEO_DIR / \
            BASE_RGB_DIR / f"{traj_name}__rgb.{RGB_VIDEO_EXTENSION}"

        rgb_frame = decode_frame(container=video_file,
                                 frame_idx=frame,
                                 fmt="rgb")

        return rgb_frame


class DepthBaseModality(AbstractBaseModality[np.ndarray]):

    def __init__(self):
        super(DepthBaseModality, self).__init__(requires_rgb=False,
                                                requires_depth=True)

    def prepare_files_list(self,
                           trajectories: List[str],
                           files_list: List[str]) -> List[str]:
        return prepare_depth_files(trajectories, files_list)

    def read(self,
             idx: int,
             root: Path,
             record: Dict[str, Any],
             hf_dataset: Dataset,
             metadata: EgoWalkBaseMetadata) -> np.ndarray:
        traj_name = record["trajectory"]
        frame = record["frame"]
        video_file = root / BASE_VIDEO_DIR / \
            BASE_DEPTH_DIR / f"{traj_name}__depth.{DEPTH_VIDEO_EXTENSION}"

        depth_frame = decode_frame(container=video_file,
                                   frame_idx=frame,
                                   fmt="depth")

        return depth_frame


class Pose3DBaseModality(AbstractBaseModality[np.ndarray]):

    def __init__(self):
        super(Pose3DBaseModality, self).__init__(requires_rgb=False,
                                                 requires_depth=False)

    def prepare_files_list(self,
                           trajectories: List[str],
                           files_list: List[str]) -> List[str]:
        return prepare_parquet_files(trajectories, files_list)

    def read(self,
             idx: int,
             root: Path,
             record: Dict[str, Any],
             hf_dataset: Dataset,
             metadata: EgoWalkBaseMetadata) -> np.ndarray:
        cart_x = record["cart_x"]
        if cart_x is None:
            return np.array([np.nan] * 7)
        cart_y = record["cart_y"]
        cart_z = record["cart_z"]
        quat_x = record["quat_x"]
        quat_y = record["quat_y"]
        quat_z = record["quat_z"]
        quat_w = record["quat_w"]
        return np.array([cart_x, cart_y, cart_z, quat_x, quat_y, quat_z, quat_w])


class Pose2DBaseModality(AbstractBaseModality[np.ndarray]):

    def __init__(self):
        super(Pose2DBaseModality, self).__init__(requires_rgb=False,
                                                 requires_depth=False)

    def prepare_files_list(self,
                           trajectories: List[str],
                           files_list: List[str]) -> List[str]:
        return prepare_parquet_files(trajectories, files_list)

    def read(self,
             idx: int,
             root: Path,
             record: Dict[str, Any],
             hf_dataset: Dataset,
             metadata: EgoWalkBaseMetadata) -> np.ndarray:
        cart_x = record["cart_x"]
        if cart_x is None:
            return np.array([np.nan] * 3)
        cart_y = record["cart_y"]
        quat_x = record["quat_x"]
        quat_y = record["quat_y"]
        quat_z = record["quat_z"]
        quat_w = record["quat_w"]
        yaw = quaternion_to_yaw(quat_x, quat_y, quat_z, quat_w)
        return np.array([cart_x, cart_y, yaw])


class HeightBaseModality(AbstractBaseModality[float]):

    def __init__(self):
        super(HeightBaseModality, self).__init__(requires_rgb=False,
                                                 requires_depth=False)

    def prepare_files_list(self,
                           trajectories: List[str],
                           files_list: List[str]) -> List[str]:
        return files_list

    def read(self,
             idx: int,
             root: Path,
             record: Dict[str, Any],
             hf_dataset: Dataset,
             metadata: EgoWalkBaseMetadata) -> float:
        return metadata.heights[record["trajectory"]]
