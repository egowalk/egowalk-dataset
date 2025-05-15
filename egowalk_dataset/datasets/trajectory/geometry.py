from __future__ import annotations

import numpy as np

from typing import Tuple


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


class Vector3D:

    def __init__(self, xyz: Tuple[float, float, float]):
        self._x = xyz[0]
        self._y = xyz[1]
        self._z = xyz[2]

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    def to_array(self) -> np.ndarray:
        return np.array([self._x, self._y, self._z])

    def __str__(self) -> str:
        return f"({self._x}, {self._y}, {self._z})"

    def __repr__(self):
        return f"{{'x': {self._x}, 'y': {self._y}, 'z': {self._z}}}"


class Quaternion3D:

    def __init__(self, xyzw: Tuple[float, float, float, float]):
        self._x = xyzw[0]
        self._y = xyzw[1]
        self._z = xyzw[2]
        self._w = xyzw[3]

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def w(self) -> float:
        return self._w

    def to_array(self) -> np.ndarray:
        return np.array([self._x, self._y, self._z, self._w])

    def __str__(self) -> str:
        return f"({self._x}, {self._y}, {self._z}, {self._w})"

    def __repr__(self):
        return f"{{'x': {self._x}, 'y': {self._y}, 'z': {self._z}, 'w': {self._w}}}"


class Pose3D:

    def __init__(self,
                 position: Vector3D,
                 orientation: Quaternion3D):
        self._position = position
        self._orientation = orientation

    @property
    def position(self) -> Vector3D:
        return self._position

    @property
    def orientation(self) -> Quaternion3D:
        return self._orientation

    def __str__(self):
        return f"Position: {str(self._position)}, Orientation: {str(self._orientation)}"

    def __repr__(self):
        return f"{{'position': {self._position.__repr__()}, 'orientation': {self._orientation.__repr__()}}}"


class Pose2D:

    def __init__(self,
                 x: float,
                 y: float,
                 yaw: float):
        self._x = x
        self._y = y
        self._yaw = yaw

    @staticmethod
    def from_3d(pose: Pose3D) -> Pose2D:
        return Pose2D(pose.position.x,
                      pose.position.y,
                      quaternion_to_yaw(pose.orientation.x,
                                        pose.orientation.y,
                                        pose.orientation.z,
                                        pose.orientation.w))

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def yaw(self) -> float:
        return self._yaw

    def to_array(self) -> np.ndarray:
        return np.array([self._x, self._y, self._yaw])

    def __str__(self) -> str:
        return f"({self._x}, {self._y}, {self._yaw})"

    def __repr__(self):
        return f"{{'x': {self._x}, 'y': {self._y}, 'yaw': {self._yaw}}}"
