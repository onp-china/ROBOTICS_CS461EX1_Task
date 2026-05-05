import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
        self._rollout_debug_start_object_pos = None
        self._rollout_debug_end_object_pos = None
        self._rollout_debug_min_distance = float("inf")
        self._rollout_debug_task_success = False
        self._rollout_debug_ready_to_grasp = False
        self._rollout_pregrasp_distance_threshold = 0.03
        self._rollout_pregrasp_gripper_open_threshold = 0.02
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))
        self._rollout_debug_start_object_pos = None
        self._rollout_debug_end_object_pos = None
        self._rollout_debug_min_distance = float("inf")
        self._rollout_debug_task_success = False
        self._rollout_debug_ready_to_grasp = False
        self._update_rollout_debug_metrics()

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)
            self._update_rollout_debug_metrics()

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

    def get_rollout_debug_metrics(self):
        min_distance = self._rollout_debug_min_distance
        displacement = self._compute_object_displacement()
        return {
            "min_eef_object_distance": float(min_distance) if np.isfinite(min_distance) else float("nan"),
            "object_displacement": float(displacement) if displacement is not None else 0.0,
            "task_success": bool(self._rollout_debug_task_success),
            "ready_to_grasp": bool(self._rollout_debug_ready_to_grasp),
        }

    def _unwrap_lowdim_env(self):
        env = self.env
        while hasattr(env, "env"):
            if (
                hasattr(env, "get_eef_position")
                and hasattr(env, "get_object_position")
                and hasattr(env, "get_gripper_openness")
                and hasattr(env, "is_task_success")
            ):
                return env
            env = env.env
        if (
            hasattr(env, "get_eef_position")
            and hasattr(env, "get_object_position")
            and hasattr(env, "get_gripper_openness")
            and hasattr(env, "is_task_success")
        ):
            return env
        return None

    def _update_rollout_debug_metrics(self):
        env = self._unwrap_lowdim_env()
        if env is None:
            return
        eef_pos = env.get_eef_position()
        object_pos = env.get_object_position()
        gripper_openness = env.get_gripper_openness()
        task_success = env.is_task_success()
        if object_pos is not None:
            object_pos = np.asarray(object_pos, dtype=np.float32).reshape(-1)
            if self._rollout_debug_start_object_pos is None:
                self._rollout_debug_start_object_pos = object_pos.copy()
            self._rollout_debug_end_object_pos = object_pos.copy()
        if task_success is not None:
            self._rollout_debug_task_success = self._rollout_debug_task_success or bool(task_success)
        if eef_pos is None or object_pos is None:
            return
        eef_pos = np.asarray(eef_pos, dtype=np.float32).reshape(-1)
        if eef_pos.shape[0] < 3 or object_pos.shape[0] < 3:
            return
        distance = float(np.linalg.norm(eef_pos[:3] - object_pos[:3]))
        self._rollout_debug_min_distance = min(self._rollout_debug_min_distance, distance)
        if gripper_openness is not None:
            ready = (
                distance <= self._rollout_pregrasp_distance_threshold
                and float(gripper_openness) >= self._rollout_pregrasp_gripper_open_threshold
            )
            self._rollout_debug_ready_to_grasp = self._rollout_debug_ready_to_grasp or bool(ready)

    def _compute_object_displacement(self):
        if self._rollout_debug_start_object_pos is None or self._rollout_debug_end_object_pos is None:
            return None
        if self._rollout_debug_start_object_pos.shape[0] < 3 or self._rollout_debug_end_object_pos.shape[0] < 3:
            return None
        return float(np.linalg.norm(
            self._rollout_debug_end_object_pos[:3] - self._rollout_debug_start_object_pos[:3]
        ))
