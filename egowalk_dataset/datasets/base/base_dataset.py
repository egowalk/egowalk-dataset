import torch

from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from egowalk_dataset.misc.constants import (HF_EGOWALK_HOME,
                                            BASE_PARQUET_DIR)
from egowalk_dataset.datasets.base.base_modalities import (AbstractBaseModality,
                                                      RGBBaseModality,
                                                      DepthBaseModality,
                                                      Pose3DBaseModality,
                                                      Pose2DBaseModality,
                                                      HeightBaseModality,
                                                      prepare_parquet_files)
from egowalk_dataset.datasets.base.base_modalities import EgoWalkBaseMetadata


DEFAULT_MODALITY_RGB = "rgb"
DEFAULT_MODALITY_DEPTH = "depth"
DEFAULT_MODALITY_POSE_3D = "pose_3d"
DEFAULT_MODALITY_POSE_2D = "pose_2d"
DEFAULT_MODALITY_HEIGHT = "height"


class EgoWalkBaseDataset(torch.utils.data.Dataset):

    def __init__(self,
                 repo_id: Optional[str] = None,
                 root: Optional[Union[str, Path]] = None,
                 trajectories: Optional[List[str]] = None,
                 modalities: Optional[Union[List[str], Dict[str, AbstractBaseModality]]] = None,
                 force_cache_sync: bool = False):
        if repo_id is None:
            assert root is not None, "Either repo_id or root must be provided"
        if root is None:
            root = HF_EGOWALK_HOME / repo_id
        else:
            root = Path(root)
        self._root = root
        self._repo_id = repo_id

        self._meta = EgoWalkBaseMetadata(repo_id=repo_id,
                                         root=root,
                                         force_cache_sync=force_cache_sync)
        self._modalities = self._get_modalities(modalities)

        if trajectories is None:
            trajectories = self._meta.trajectories.copy()
        self._trajectories = trajectories

        self._hf_dataset: Dataset = None
        self._traj_index: Dict[str, Tuple[int, int]] = None

        files_list = self._prepare_files_list(trajectories)
        if repo_id is None:
            self._load_hf_dataset(files_list)
        else:
            try:
                if force_cache_sync:
                    raise FileNotFoundError
                assert all((root / file).is_file() for file in files_list)
                self._load_hf_dataset(files_list)
            except (AssertionError, FileNotFoundError, NotADirectoryError):
                self._download_data(files_list)
                self._load_hf_dataset(files_list)

        self._build_trajectories_index()

    @property
    def meta(self) -> EgoWalkBaseMetadata:
        return self._meta
    
    @property
    def repo_id(self) -> Optional[str]:
        return self._repo_id
    
    @property
    def root(self) -> Path:
        return self._root
    
    @property
    def modalities(self) -> List[str]:
        return list(self._modalities.keys())

    @property
    def trajectories(self) -> List[str]:
        return self._trajectories

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self._hf_dataset[idx]
        result = {}
        for name, modality in self._modalities.items():
            result[name] = modality.read(idx=idx,
                                         root=self._root,
                                         record=record,
                                         hf_dataset=self._hf_dataset,
                                         metadata=self._meta)
        return result
    
    def __len__(self) -> int:
        return len(self._hf_dataset)
    
    def get_trajectory_item(self,
                            trajectory: str,
                            idx: int) -> Dict[str, Any]:
        start, end = self._traj_index[trajectory]
        abs_idx = start + idx
        if abs_idx > end or abs_idx < start:
            raise IndexError(f"Index {idx} is out of bounds for trajectory {trajectory}")
        return self[abs_idx]
    
    def get_trajectory_length(self,
                             trajectory: str) -> int:
        start, end = self._traj_index[trajectory]
        return end - start + 1

    def _get_modalities(self,
                        modalities: Optional[Union[List[str], Dict[str, AbstractBaseModality]]]) -> Dict[str, AbstractBaseModality]:
        if modalities is None:
            modalities = [DEFAULT_MODALITY_RGB, 
                          DEFAULT_MODALITY_DEPTH, 
                          DEFAULT_MODALITY_POSE_2D]
        
        if isinstance(modalities, dict):
            return modalities
        
        result = {}
        for modality in modalities:
            if modality == DEFAULT_MODALITY_RGB:
                result[modality] = RGBBaseModality()
            elif modality == DEFAULT_MODALITY_DEPTH:
                result[modality] = DepthBaseModality()
            elif modality == DEFAULT_MODALITY_POSE_3D:
                result[modality] = Pose3DBaseModality()
            elif modality == DEFAULT_MODALITY_POSE_2D:
                result[modality] = Pose2DBaseModality()
            elif modality == DEFAULT_MODALITY_HEIGHT:
                result[modality] = HeightBaseModality()
            else:
                raise ValueError(f"Unknown default modality: {modality}")
        
        return result

    def _needs_rgb(self,
                   modalities: Dict[str, AbstractBaseModality]) -> bool:
        return any(modality.requires_rgb for modality in modalities.values())
    
    def _needs_depth(self,
                     modalities: Dict[str, AbstractBaseModality]) -> bool:
        return any(modality.requires_depth for modality in modalities.values())
    
    def _prepare_files_list(self,
                            trajectories: Optional[List[str]] = None) -> Optional[List[str]]:
        if trajectories is None:
            return None
        files_list = prepare_parquet_files(trajectories, [])
        for modality in self._modalities.values():
            files_list = modality.prepare_files_list(trajectories, files_list)
        return files_list
    
    def _load_hf_dataset(self, files_list: List[str]):
        files_list = [str(self._root / e) for e in files_list if e.split("/")[-2] == BASE_PARQUET_DIR]
        self._hf_dataset = load_dataset("parquet",
                                        data_files=files_list,
                                        split="train")
        
    def _download_data(self, files_list: List[str]):
        snapshot_download(
            self._repo_id,
            repo_type="dataset",
            local_dir=self._root,
            allow_patterns=files_list,
        )

    def _build_trajectories_index(self):
        traj_index = {}
        previous_name = None
        current_start = 0

        for i in range(len(self._hf_dataset)):
            record = self._hf_dataset[i]
            name = record["trajectory"]
            
            if previous_name is None:
                # Case 1: first trajectory
                previous_name = name   

            elif previous_name != name:
                # Case 2: new trajectory appeared
                traj_index[previous_name] = (current_start, i - 1)
                current_start = i
                previous_name = name

        traj_index[previous_name] = (current_start, len(self._hf_dataset) - 1)
        
        self._traj_index = traj_index
