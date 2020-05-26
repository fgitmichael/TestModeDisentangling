import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import rlkit.torch.sac.diayn
from .mode_actions_sampler import ModeActionSampler
from network import ModeDisentanglingNetwork
from env import DmControlEnvForPytorchBothObstype


class DisentanglingTester:
    def __init__(self,
                 latent_model_path,
                 env,
                 seed,
                 video_dir,
                 ):

        # Latent model
        self.latent_model = torch.load(latent_model_path).eval()
        print("Model loaded")

        # Environment
        self.env = env
        assert isinstance(self.env, DmControlEnvForPytorchBothObstype), \
            'Both observation types (pixel and state representantion are needed' \
            ' to create the test video. ' \
            'Take DmControlForPytorchBothObstype env-class'
        assert self.env.obs_type == 'pixels'
        self.action_repeat = self.env.action_repeat

        # Seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Directories
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)

        # Video
        self.video_cnt = 0
        self.video_name = 'skill'
        self.video = None
        self._reset_video_writer(video_name=self.video_name + str(self.video_cnt))

        # Sampling mode conditioned actions from the latent model
        self.mode_action_sampler = ModeActionSampler(self.latent_model, device=self.device)

        # Other
        self.steps = 0
        self.episodes = 0

    def _create_video_name(self):
        return self.video_name + str(self.video_cnt)

    def _reset_video_writer(self, video_name):
        video_path = os.path.join(self.video_dir, video_name)
        video_path += '.avi'
        rows = self.env.observation_space.shape[1]
        cols = self.env.observation_space.shape[2]
        self.video = cv2.VideoWriter(video_path,
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     25,
                                     (rows, cols),
                                     True)

        self.video_cnt += 1


    def _write_img_to_video(self, img):
        # (H, W, num_channels) seems to be needed by cvtColor
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        bgr_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.video.write(bgr_img)

    def _save_video(self):
        self.video.release()

    def _get_state_obs_enc(self):
        state_obs = self.env.get_state_obs()
        with torch.no_grad():
            state_obs_enc = self.latent_model.encoder(
                torch.from_numpy(state_obs).unsqueeze(dim=0).float().to(self.device)
        )
        return state_obs_enc

    def generate_skill_autoregressive(self):
        # Env reset
        pixel_obs = self.env.reset()
        state_obs = self._get_state_obs_enc()
        self._write_img_to_video(pixel_obs)

        # Counters
        self.episodes += 1
        episode_steps = 0

        # Done Flag
        done = False

        while not done:
            action = self.mode_action_sampler(state_obs)
            pixel_obs, _, done, _ = self.env.step(action.cpu().numpy())
            state_obs = self._get_state_obs_enc()

            self.steps += self.action_repeat
            episode_steps += self.action_repeat

            self._write_img_to_video(pixel_obs)

    def run(self, num_skills=10):
        for skill in range(num_skills):
            # Resets
            self.mode_action_sampler.reset()
            self._reset_video_writer(video_name=self._create_video_name())

            # Make video
            self.generate_skill_autoregressive()
            self._save_video()

