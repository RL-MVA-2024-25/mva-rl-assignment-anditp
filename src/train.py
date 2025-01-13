import gymnasium as gym
from env_hiv import HIVPatient
from stable_baselines3 import PPO
import numpy as np
import pickle
import os

class ProjectAgent:
    def __init__(self, vec_normalize_path="src/vec_normalize.pkl"):
        # Load the trained PPO model
        self.model = PPO.load("src/ppo_hiv_normalized")
        self.vec_normalize_path = vec_normalize_path
        self.obs_rms = None

        # Load observation normalization statistics if the file exists
        if os.path.exists(self.vec_normalize_path):
            self.obs_rms = self.load_normalization_stats(vec_normalize_path)

    def load_normalization_stats(self, vec_normalize_path):
        # Load only the relevant statistics from the .pkl file
        with open(vec_normalize_path, "rb") as file_:
            data = pickle.load(file_)
        return data.obs_rms

    def act(self, observation, use_random=False):
        # Manually normalize the observation (if needed)
        if self.obs_rms is not None:
            observation = np.clip((observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)

        # Get the action from the loaded model
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def load(self):
        pass