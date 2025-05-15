import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class AbstractTrajectoryCutter(ABC):

    @abstractmethod
    def __call__(self, pose: np.ndarray, pose_next: np.ndarray) -> np.ndarray:
        pass


class CompositeCutter(AbstractTrajectoryCutter):
    
    def __init__(self, cutters: List[AbstractTrajectoryCutter]) -> None:
        super(CompositeCutter, self).__init__()
        self._cutters = [e for e in cutters]
    
    def __call__(self, pose: np.ndarray, pose_next: np.ndarray) -> np.ndarray:
        for cutter in self._cutters:
            result = cutter(pose, pose_next)
            if result:
                return True
        return False


class StuckCutter(AbstractTrajectoryCutter):

    def __init__(self, eps: float = 1e-2) -> None:
        super(StuckCutter, self).__init__()
        self._eps = eps

    def __call__(self, pose: np.ndarray, pose_next: np.ndarray) -> bool:
        distance = np.linalg.norm(pose_next[:2] - pose[:2])
        return distance < self._eps


class BackwardCutter(AbstractTrajectoryCutter):

    def __init__(self, 
                 backwards_eps: float = 1e-2, 
                 stuck_eps: float = 1e-2, 
                 ignore_stuck: bool = True) -> None:
        super(BackwardCutter, self).__init__()
        self._backwards_eps = backwards_eps
        self._stuck_eps = stuck_eps
        self._ignore_stuck = ignore_stuck

    def __call__(self, pose: np.ndarray, pose_next: np.ndarray) -> bool:
        diff = pose_next[:2] - pose[:2]
        dx, dy = diff
        yaw = pose[2]
        if dx * np.cos(yaw) + dy * np.sin(yaw) < self._backwards_eps:
            return True
        if np.linalg.norm(diff) < self._stuck_eps:
            return not self._ignore_stuck
        return False


class SpikesCutter(AbstractTrajectoryCutter):

    def __init__(self, spike_threshold: float = 3.) -> None:
        super(SpikesCutter, self).__init__()
        self._spike_threshold = spike_threshold

    def __call__(self, pose: np.ndarray, pose_next: np.ndarray) -> bool:
        return np.linalg.norm(pose_next[:2] - pose[:2]) > self._spike_threshold


def apply_cutter(trajectory: np.ndarray, cutter: Optional[AbstractTrajectoryCutter]) -> List[Tuple[int, int]]:
    assert len(trajectory.shape) == 2 and trajectory.shape[1] == 3, \
        f"Trajectory shape must be (N, 3), got {trajectory.shape}"
    if cutter is None:
        return [(0, trajectory.shape[0])]
    if isinstance(cutter, list):
        cutter = CompositeCutter(cutter)
    segments = []
    last_start_idx = 0
    for i in range(1, trajectory.shape[0]):
        if cutter(trajectory[i-1], trajectory[i]):
            segments.append((last_start_idx, i))
            last_start_idx = i
    segments.append((last_start_idx, trajectory.shape[0]))
    return segments
