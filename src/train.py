import gymnasium as gym
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import numpy as np
import pickle
import os
from stable_baselines3.common.vec_env import VecNormalize
from collections import deque


class ProjectAgent:
    def __init__(self, vec_normalize_path="src/vec_normalize.pkl", stack_size=4):
        self.model = PPO.load("src/ppo_hiv_normalized")
        self.vec_normalize_path = vec_normalize_path
        self.obs_rms = None
        self.stack_size = stack_size
        self.state_deque = deque(maxlen=stack_size)

        self.reset_state_deque()

        # Load observation normalization statistics if the file exists
        if os.path.exists(self.vec_normalize_path):
            self.obs_rms = self.load_normalization_stats(vec_normalize_path)



    def load_normalization_stats(self, vec_normalize_path):
        # Load only the relevant statistics from the .pkl file
        with open(vec_normalize_path, "rb") as file_:
            data = pickle.load(file_)
        return data.obs_rms

    def act(self, observation, use_random=False):
        # Update observation deque
        self.state_deque.append(observation)
        stacked_obs = np.array(self.state_deque)

        # Manually normalize the stacked observation (if needed)
        if self.obs_rms is not None:
            stacked_obs = np.clip((stacked_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)

        # Get the action from the loaded model
        action, _ = self.model.predict(stacked_obs, deterministic=True)
        return action

    def reset_state_deque(self):
        # Initialize the deque with zeros
        for _ in range(self.stack_size):
            self.state_deque.append(np.zeros(6, dtype=np.float32))
    
    def load(self):
        pass