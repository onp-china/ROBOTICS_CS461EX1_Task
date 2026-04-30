from __future__ import annotations

import functools
from typing import Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> tuple[np.ndarray, torch.device | None, torch.dtype | None]:
    if isinstance(x, np.ndarray):
        return x, None, None
    return x.detach().cpu().numpy(), x.device, x.dtype


def _from_numpy(
    array: np.ndarray,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        return array
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def _normalize_vector(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    norm = np.clip(norm, a_min=1e-8, a_max=None)
    return array / norm


def _rotation_6d_to_matrix(array: np.ndarray) -> np.ndarray:
    a1 = array[..., :3]
    a2 = array[..., 3:6]
    b1 = _normalize_vector(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_vector(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _matrix_to_rotation_6d(array: np.ndarray) -> np.ndarray:
    return array[..., :2, :].reshape(*array.shape[:-2], 6)


def _axis_angle_to_matrix(array: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(array.reshape(-1, 3)).as_matrix().reshape(*array.shape[:-1], 3, 3)


def _matrix_to_axis_angle(array: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(array.reshape(-1, 3, 3)).as_rotvec().reshape(*array.shape[:-2], 3)


def _quaternion_to_matrix(array: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(array.reshape(-1, 4)).as_matrix().reshape(*array.shape[:-1], 3, 3)


def _matrix_to_quaternion(array: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(array.reshape(-1, 3, 3)).as_quat().reshape(*array.shape[:-2], 4)


def _euler_to_matrix(array: np.ndarray, convention: str) -> np.ndarray:
    return Rotation.from_euler(convention, array.reshape(-1, 3)).as_matrix().reshape(*array.shape[:-1], 3, 3)


def _matrix_to_euler(array: np.ndarray, convention: str) -> np.ndarray:
    return Rotation.from_matrix(array.reshape(-1, 3, 3)).as_euler(convention).reshape(*array.shape[:-2], 3)


class RotationTransformer:
    valid_reps = [
        "axis_angle",
        "euler_angles",
        "quaternion",
        "rotation_6d",
        "matrix",
    ]

    def __init__(
        self,
        from_rep: str = "axis_angle",
        to_rep: str = "rotation_6d",
        from_convention: str | None = None,
        to_convention: str | None = None,
    ):
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            funcs = self._converter_pair(from_rep, from_convention)
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = self._converter_pair(to_rep, to_convention)
            forward_funcs.append(funcs[1])
            inverse_funcs.append(funcs[0])

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs[::-1]

    @staticmethod
    def _converter_pair(rep: str, convention: str | None):
        if rep == "axis_angle":
            return _axis_angle_to_matrix, _matrix_to_axis_angle
        if rep == "quaternion":
            return _quaternion_to_matrix, _matrix_to_quaternion
        if rep == "rotation_6d":
            return _rotation_6d_to_matrix, _matrix_to_rotation_6d
        if rep == "euler_angles":
            assert convention is not None
            return (
                functools.partial(_euler_to_matrix, convention=convention),
                functools.partial(_matrix_to_euler, convention=convention),
            )
        raise ValueError(f"Unsupported representation: {rep}")

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        array, device, dtype = _to_numpy(x)
        result = array
        for func in funcs:
            result = func(result)
        result = result.astype(array.dtype, copy=False)
        return _from_numpy(result, device, dtype)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)
