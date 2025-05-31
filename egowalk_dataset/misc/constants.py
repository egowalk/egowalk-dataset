import os

from pathlib import Path
from huggingface_hub.constants import HF_HOME


HF_EGOWALK_HOME = Path(os.getenv("HF_EGOWALK_HOME", Path(HF_HOME) / "egowalk")).expanduser()

DEFAULT_REPO_ID = "EgoWalk/trajectories"

DEFAULT_DATA_PATH  = HF_EGOWALK_HOME / DEFAULT_REPO_ID

BASE_METADATA_DIR = "meta"
BASE_PARQUET_DIR = "data"
BASE_VIDEO_DIR = "video"
BASE_RGB_DIR = "rgb"
BASE_DEPTH_DIR = "depth"
BASE_ANNOTATION_DIR = "annotations"

METADATA_TRAJECTORIES_FILE = "trajectories.json"
METADATA_HEIGHTS_FILE = "heights.json"
METADATA_INFO_FILE = "info.json"
METADATA_RGB_CAMERA_FILE = "camera_rgb.json"

RGB_VIDEO_EXTENSION = "mp4"
DEPTH_VIDEO_EXTENSION = "mkv"
