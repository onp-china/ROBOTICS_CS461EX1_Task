from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview'
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        self._last_raw_obs = None
        
        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        self._last_raw_obs = raw_obs
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs

    def get_raw_observation(self):
        if self._last_raw_obs is None:
            self._last_raw_obs = self.env.get_observation()
        return self._last_raw_obs

    def get_eef_position(self):
        raw_obs = self.get_raw_observation()
        value = raw_obs.get("robot0_eef_pos")
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def get_object_position(self):
        raw_obs = self.get_raw_observation()
        value = raw_obs.get("object")
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] < 3:
            return None
        return value[:3]

    def get_gripper_openness(self):
        raw_obs = self.get_raw_observation()
        value = raw_obs.get("robot0_gripper_qpos")
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] == 0:
            return None
        return float(np.mean(value))

    def is_task_success(self):
        candidates = [
            getattr(self.env, "is_success", None),
            getattr(self.env, "_check_success", None),
            getattr(self.env, "check_success", None),
        ]
        for candidate in candidates:
            if callable(candidate):
                try:
                    result = candidate()
                except TypeError:
                    continue
                if isinstance(result, dict):
                    for key in ("task", "success", "is_success"):
                        value = result.get(key)
                        if value is not None:
                            return bool(value)
                if result is not None:
                    return bool(result)
        return None

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self._last_raw_obs = raw_obs
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
