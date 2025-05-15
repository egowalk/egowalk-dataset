import numpy as np


class CameraParameters:

    def __init__(self,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 k1: float,
                 k2: float,
                 k3: float,
                 k4: float,
                 k5: float,
                 k6: float,
                 p1: float,
                 p2: float) -> None:
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._k4 = k4
        self._k5 = k5
        self._k6 = k6
        self._p1 = p1
        self._p2 = p2

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self._fx, 0.0, self._cx],
                         [0.0, self._fy, self._cy],
                         [0.0, 0.0, 1.0]])

    @property
    def distortion_coefficients(self) -> np.ndarray:
        return np.array([self._k1, self._k2, self._p1,
                         self._p2, self._k3, self._k4, self._k5, self._k6])
