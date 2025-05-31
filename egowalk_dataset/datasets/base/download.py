from pathlib import Path
from typing import Optional, List
from huggingface_hub import snapshot_download
from egowalk_dataset.misc.constants import (DEFAULT_REPO_ID,
                                            HF_EGOWALK_HOME,
                                            BASE_PARQUET_DIR,
                                            BASE_ANNOTATION_DIR,
                                            BASE_VIDEO_DIR,
                                            BASE_DEPTH_DIR,
                                            BASE_RGB_DIR,
                                            RGB_VIDEO_EXTENSION,
                                            DEPTH_VIDEO_EXTENSION)
from egowalk_dataset.datasets.base.base_metadata import EgoWalkBaseMetadata


def _collect_video_files(trajectories: List[str],
                         video_type: str) -> List[str]:
    result = []
    for traj in trajectories:
        if video_type == "rgb":
            video_file = f"{BASE_VIDEO_DIR}/{BASE_RGB_DIR}/{traj}__{video_type}.{RGB_VIDEO_EXTENSION}"
        elif video_type == "depth":
            video_file = f"{BASE_VIDEO_DIR}/{BASE_DEPTH_DIR}/{traj}__{video_type}.{DEPTH_VIDEO_EXTENSION}"
        else:
            raise ValueError(f"Invalid video type: {video_type}")
        result.append(video_file)
    return result


def download_dataset(download_rgb: bool,
                     download_depth: bool,
                     repo_id: str = DEFAULT_REPO_ID,
                     root: Optional[str] = None,
                     trajectories: Optional[List[str]] = None,
                     force_cache_sync: bool = False) -> None:
    # Determine download directory if not provided
    if root is None:
        root = HF_EGOWALK_HOME / repo_id
    else:
        root = Path(root)
    
    # Load metadata (it is lightweight)
    metadata = EgoWalkBaseMetadata(repo_id=repo_id,
                                   root=root,
                                   force_cache_sync=force_cache_sync)
    
    # If no trajectories are provided, use all trajectories
    if trajectories is None:
        trajectories = metadata.trajectories.copy()
    
    # Collect files to download
    files_list = [
        f"{BASE_PARQUET_DIR}/*",
        f"{BASE_ANNOTATION_DIR}/*",
    ]
    if download_rgb:
        files_list.extend(_collect_video_files(trajectories, "rgb"))
    if download_depth:
        files_list.extend(_collect_video_files(trajectories, "depth"))

    # Do actual download
    snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=files_list,
        )
