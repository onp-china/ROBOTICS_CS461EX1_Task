"""Minimal mujoco_py compatibility shim for robomimic on modern mujoco setups.

This project runs robosuite with the `mujoco` Python package, but robomimic's
robosuite wrapper still imports `mujoco_py` only to reference
`mujoco_py.builder.MujocoException`.
"""


class _MujocoException(Exception):
    """Placeholder exception matching mujoco_py.builder.MujocoException."""


class _Builder:
    MujocoException = _MujocoException


builder = _Builder()
