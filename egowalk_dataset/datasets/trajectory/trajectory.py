from __future__ import annotations

import json
import numpy as np
import pandas as pd
import av

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path
from egowalk_dataset.datasets.trajectory.geometry import (Pose3D, 
                                                 Vector3D, 
                                                 Quaternion3D, 
                                                 Pose2D)
from egowalk_dataset.util.video import FormatType, decode_frame
from egowalk_dataset.misc.constants import (DEFAULT_DATA_PATH,
                                             RGB_VIDEO_EXTENSION,
                                             DEPTH_VIDEO_EXTENSION,
                                             BASE_RGB_DIR,
                                             BASE_DEPTH_DIR,
                                             BASE_METADATA_DIR,
                                             BASE_PARQUET_DIR,
                                             BASE_VIDEO_DIR)


class AbstractVideoChannel(ABC):

    def __init__(self,
                 parquet_file: Union[str, Path],
                 video_file: Union[str, Path],
                 fmt: FormatType,
                 timestamps: Optional[List[int]] = None):
        parquet_file = Path(parquet_file)
        df = pd.read_parquet(parquet_file)

        timestamps_to_idx = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            timestamp = int(row['timestamp'])
            if timestamps is not None and timestamp not in timestamps:
                continue
            timestamps_to_idx[timestamp] = idx

        self._timestamps_to_idx = timestamps_to_idx
        self._video_file = Path(video_file)
        self._format = fmt

    @abstractmethod
    def __getitem__(self, idx: int) -> np.ndarray:
        pass

    def __len__(self) -> int:
        return len(self._timestamps_to_idx)
    
    @property
    def timestamps(self) -> List[int]:
        return sorted(self._timestamps_to_idx.keys())
    
    @property
    def video_file(self) -> Path:
        return self._video_file

    def timestamp_to_idx(self, timestamp: int) -> int:
        if timestamp not in self._timestamps_to_idx:
            raise ValueError(f"Timestamp {timestamp} not found in trajectory")
        return self._timestamps_to_idx[timestamp]
    
    def at(self, timestamp: int) -> Optional[np.ndarray]:
        if timestamp not in self._timestamps_to_idx:
            raise ValueError(f"Timestamp {timestamp} not found in trajectory")
        return self[self._timestamps_to_idx[timestamp]]

    def close(self) -> None:
        pass


class BasicVideoChannel(AbstractVideoChannel):

    def __init__(self,
                 parquet_file: Union[str, Path],
                 video_file: Union[str, Path],
                 fmt: FormatType,
                 timestamps: Optional[List[int]] = None):
        super(BasicVideoChannel, self).__init__(parquet_file, 
                                                video_file, 
                                                fmt, 
                                                timestamps)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return decode_frame(container=self._video_file,
                            frame_idx=idx,
                            fmt=self._format)
    

class KeepOpenVideoChannel(AbstractVideoChannel):

    def __init__(self,
                 parquet_file: Union[str, Path],
                 video_file: Union[str, Path],
                 fmt: FormatType,
                 timestamps: Optional[List[int]] = None):
        super(KeepOpenVideoChannel, self).__init__(parquet_file, 
                                                   video_file, 
                                                   fmt, 
                                                   timestamps)
        self._container = av.open(self._video_file, mode="r")
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return decode_frame(container=self._container,
                            frame_idx=idx,
                            fmt=self._format)
    
    def close(self) -> None:
        self._container.close()


class OdometryChannel:

    def __init__(self, 
                 parquet_file: Union[str, Path],
                 timestamps: Optional[List[int]] = None) -> None:
        parquet_file = Path(parquet_file)
        df = pd.read_parquet(parquet_file)
        poses = []
        timestamp_to_idx = {}
        valid_timestamps = []        
        for idx, (_, row) in enumerate(df.iterrows()):
            timestamp = int(row['timestamp'])
            if timestamps is not None and timestamp not in timestamps:
                continue
            
            cart_x = row['cart_x']
            if np.isnan(cart_x):
                poses.append(None)
            else:
                cart_x = float(cart_x)
                cart_y = float(row['cart_y'])
                cart_z = float(row['cart_z'])
                quat_x = float(row['quat_x'])
                quat_y = float(row['quat_y'])
                quat_z = float(row['quat_z'])
                quat_w = float(row['quat_w'])
                poses.append(Pose3D(Vector3D((cart_x, cart_y, cart_z)), 
                                    Quaternion3D((quat_x, quat_y, quat_z, quat_w))))
                valid_timestamps.append(timestamp)
            
            timestamp_to_idx[timestamp] = idx
        
        self._poses = poses
        self._timestamp_to_idx = timestamp_to_idx
        self._valid_timestamps = valid_timestamps

    def __len__(self) -> int:
        return len(self._poses)
    
    def __getitem__(self, idx: int) -> Optional[Pose3D]:
        return self._poses[idx]

    @property
    def poses(self) -> List[Optional[Pose3D]]:
        return self._poses

    @property
    def valid_timestamps(self) -> List[int]:
        return self._valid_timestamps
    
    @property
    def all_timestamps(self) -> List[int]:
        return sorted(self._timestamp_to_idx.keys())
    
    def at(self, timestamp: int) -> Optional[Pose3D]:
        if timestamp not in self._timestamp_to_idx:
            raise ValueError(f"Timestamp {timestamp} not found in trajectory")
        return self._poses[self._timestamp_to_idx[timestamp]]
    
    def get_bev(self,
                filter_valid: bool = False) -> np.ndarray:
        if filter_valid:
            traj = [Pose2D.from_3d(e).to_array()
                    for e in self._poses if e is not None]
        else:
            traj = [Pose2D.from_3d(e).to_array() \
                    if e is not None else np.array([np.nan] * 3)
                    for e in self._poses]
        return np.stack(traj, axis=0)


class EgoWalkTrajectory:

    def __init__(self,
                 rgb_video_file: Union[str, Path],
                 depth_video_file: Union[str, Path],
                 parquet_file: Union[str, Path],
                 camera_height: float,
                 timestamps: Optional[List[int]] = None,
                 keep_video_open: bool = False):
        if not keep_video_open:
            self._rgb = BasicVideoChannel(parquet_file=parquet_file,
                                          video_file=rgb_video_file,
                                          fmt="rgb",
                                          timestamps=timestamps)
            self._depth = BasicVideoChannel(parquet_file=parquet_file,
                                            video_file=depth_video_file,
                                            fmt="depth",
                                            timestamps=timestamps)
        else:
            self._rgb = KeepOpenVideoChannel(parquet_file=parquet_file,
                                             video_file=rgb_video_file,
                                             fmt="rgb",
                                            timestamps=timestamps)
            self._depth = KeepOpenVideoChannel(parquet_file=parquet_file,
                                               video_file=depth_video_file,
                                               fmt="depth",
                                               timestamps=timestamps)
        self._odometry = OdometryChannel(parquet_file=parquet_file,
                                         timestamps=timestamps)
        self._camera_height = camera_height

    @classmethod
    def from_dataset(cls,
                     name: str,
                     data_path: Union[str, Path] = DEFAULT_DATA_PATH,
                     timestamps: Optional[List[int]] = None,
                     keep_video_open: bool = False) -> EgoWalkTrajectory:
        root = Path(data_path)
        with open(root / BASE_METADATA_DIR / "heights.json", "r") as f:
            height = float(json.load(f)[name])
        return cls(rgb_video_file=root / BASE_VIDEO_DIR /BASE_RGB_DIR / f"{name}__rgb.{RGB_VIDEO_EXTENSION}",
                   depth_video_file=root / BASE_VIDEO_DIR/ BASE_DEPTH_DIR / f"{name}__depth.{DEPTH_VIDEO_EXTENSION}",
                   parquet_file=root / BASE_PARQUET_DIR / f"{name}.parquet",
                   camera_height=height,
                   timestamps=timestamps,
                   keep_video_open=keep_video_open)

    @property
    def rgb(self) -> AbstractVideoChannel:
        return self._rgb
    
    @property
    def depth(self) -> AbstractVideoChannel:
        return self._depth
    
    @property
    def odometry(self) -> OdometryChannel:
        return self._odometry
    
    @property
    def camera_height(self) -> float:
        return self._camera_height
    
    def close(self) -> None:
        self._rgb.close()
        self._depth.close()
