import json
from typing import Optional, Union, List, Dict
from pathlib import Path
from huggingface_hub import snapshot_download
from egowalk_dataset.misc.constants import (HF_EGOWALK_HOME,
                                            BASE_METADATA_DIR,
                                            METADATA_TRAJECTORIES_FILE,
                                            METADATA_HEIGHTS_FILE,
                                            METADATA_INFO_FILE,
                                            METADATA_RGB_CAMERA_FILE)
from egowalk_dataset.camera.camera_params import CameraParameters


class EgoWalkBaseMetadata:

    def __init__(self,
                 repo_id: Optional[str] = None,
                 root: Optional[Union[str, Path]] = None,
                 force_cache_sync: bool = False):
        if repo_id is None:
            assert root is not None, "Either repo_id or root must be provided"
        if root is None:
            root = HF_EGOWALK_HOME / repo_id
        else:
            root = Path(root)
        self._root = root
        self._repo_id = repo_id

        self._trajectories: List[str] = None
        self._heights: Dict[str, float] = None
        self._fps: float = None
        self._rgb_camera_params: Optional[CameraParameters] = None

        if repo_id is None:
            # Repo ID is not provided, just load data from the file system
            self._load_metadata()
        else:
            # Repo ID is provided, need sync it
            try:
                if force_cache_sync:
                    raise FileNotFoundError
                self._load_metadata()
            except (FileNotFoundError, NotADirectoryError):
                (self._root / BASE_METADATA_DIR).mkdir(exist_ok=True, parents=True)
                self._download_metadata()
                self._load_metadata()

    @property
    def repo_id(self) -> str:
        return self._repo_id
    
    @property
    def root(self) -> Path:
        return self._root
    
    @property
    def trajectories(self) -> List[str]:
        return self._trajectories
    
    @property
    def heights(self) -> Dict[str, float]:
        return self._heights

    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def rgb_camera_params(self) -> Optional[CameraParameters]:
        return self._rgb_camera_params

    def _load_metadata(self):
        with open(self._root / BASE_METADATA_DIR / METADATA_TRAJECTORIES_FILE, "r") as f:
            self._trajectories = json.load(f)
        with open(self._root / BASE_METADATA_DIR / METADATA_HEIGHTS_FILE, "r") as f:
            self._heights = json.load(f)

        with open(self._root / BASE_METADATA_DIR / METADATA_INFO_FILE, "r") as f:
            info = json.load(f)
        self._fps = info["fps"]

        camera_file = self._root / BASE_METADATA_DIR / METADATA_RGB_CAMERA_FILE
        with open(camera_file, "r") as f:
            camera_params = json.load(f)
        if len(camera_params) > 0:
            camera_params = CameraParameters(**camera_params)
        else:
            camera_params = None
        self._rgb_camera_params = camera_params

    def _download_metadata(self):
        snapshot_download(self._repo_id,
                          repo_type="dataset",
                          local_dir=self._root,
                          allow_patterns=f"{BASE_METADATA_DIR}/*")
