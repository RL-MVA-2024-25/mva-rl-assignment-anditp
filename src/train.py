import gymnasium as gym
from env_hiv import HIVPatient
from stable_baselines3 import PPO
import numpy as np
import pickle
import os

class ProjectAgent:
    def __init__(self, vec_normalize_path="src/vec_normalize.pkl"):
        self.model = PPO.load("src/ppo_hiv_normalized")
        self.vec_normalize_path = vec_normalize_path
        self.obs_rms = None

        if os.path.exists(self.vec_normalize_path):
            self.obs_rms = self.load_normalization_stats(vec_normalize_path)

    def load_normalization_stats(self, vec_normalize_path):
        with open(vec_normalize_path, "rb") as file_:
            data = pickle.load(file_)
        return data.obs_rms

    def act(self, observation, use_random=False):
        if self.obs_rms is not None:
            observation = np.clip((observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)

        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path):
        pass
    
    def load(self):
        pass