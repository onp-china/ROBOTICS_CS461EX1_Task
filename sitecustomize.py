"""Runtime compatibility patches for this Windows training environment."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "glfw")

try:
    import robosuite.utils.mjcf_utils as mjcf_utils

    if not hasattr(mjcf_utils, "postprocess_model_xml"):
        mjcf_utils.postprocess_model_xml = lambda xml: xml
except Exception:
    pass
