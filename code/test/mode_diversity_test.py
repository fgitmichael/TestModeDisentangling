import os
import numpy as np
import torch
import cv2

import rlkit.torch.sac.diayn
from .mode_actions_sampler import ModeActionSampler
from network import ModeDisentanglingNetwork
from env import OrdinaryEnvForPytorch


class DisentanglingDiversityTester:
    def __init__(self,
                 latent_model_path,
                 env,
                 seed=0,
                 ):
        # Latent model
        self.latent_model = torch.load(latent_model_path).eval()
        print("Model loaded")

        # Environment
        self.env = env
        self.action_repeat = self.env.action_repeat

        # Seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Sampling mode-conditioned actions from the latent model
        self.mode_action_sampler = ModeActionSampler(self.latent_model, device=self.device)

        # Other
        self.steps = 0
        self.episodes = 0

    def _get_state_obs_enc(self, state_obs):
        with torch.no_grad():
            state_obs_enc = self.latent_model.encoder(
                torch.from_numpy(state_obs).unsqueeze(dim=0).float().to(self.device)
            )
        return state_obs_enc

    def show_skill(self):
        # Env reset
        obs = self.env.reset()
        self.env.render()

        episode_steps = 0
        done = False
        while not done:
            action = self.mode_action_sampler(self._get_state_obs_enc(obs))
            obs, _, done, _ = self.env.step(action.cpu().numpy()[0])
            self.env.render()

            self.steps += self.action_repeat
            episode_steps += 1

    def run(self):
        skills = torch.Tensor([i for i in np.arange(-2, 2, 0.1)]).unsqueeze(1)
        for skill in skills:
            self.mode_action_sampler.reset(mode=skill.unsqueeze(0))
            self.show_skill()
            print('new_mode' + str(skill))



