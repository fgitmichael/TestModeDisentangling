import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


from memory.memory_disentangling import MyMemoryDisentangling
from latent_model_trainer import LatentTrainer
from network.mode_disentangling import ModeDisentanglingNetwork
from utils import calc_kl_divergence, update_params, RunningMeanStats

# Needed for the loaded skill policy (Do not delete!)
import rlkit.torch.sac.diayn


class DisentanglingTrainer(LatentTrainer):
    def __init__(self,
                 env,
                 log_dir,
                 state_rep,
                 skill_policy_path,
                 seed,
                 run_id,
                 feature_dim=256,
                 num_sequences=40,
                 cuda=False,
                 ):
        parent_kwargs = dict(
            num_steps=3000000,
            initial_latent_steps=100000,
            batch_size=256,
            latent_batch_size=64,
            num_sequences=num_sequences,
            latent_lr=0.0001,
            feature_dim=feature_dim,
            latent1_dim=32,
            latent2_dim=256,
            hidden_units=[256, 256],
            hidden_rnn_dim=100,
            rnn_layers=1,
            memory_size=1e5,
            leaky_slope=0.2,
            grad_clip=None,
            start_steps=10000,
            training_log_interval=100,
            learning_log_interval=100,
            cuda=cuda,
            seed=seed)

        # Other
        self.run_id = run_id
        self.state_rep = state_rep

        # Comment for summery writer
        summary_comment = self.run_id

        # Environment
        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.action_repeat = self.env.action_repeat

        # Seed
        torch.manual_seed(parent_kwargs['seed'])
        np.random.seed(parent_kwargs['seed'])
        self.env.seed(parent_kwargs['seed'])

        # Device
        self.device = torch.device(
            "cuda" if parent_kwargs['cuda'] and torch.cuda.is_available() else "cpu"
        )

        # Latent Network
        self.latent = ModeDisentanglingNetwork(
            self.observation_shape,
            self.action_shape,
            feature_dim=parent_kwargs['feature_dim'],
            latent1_dim=parent_kwargs['latent1_dim'],
            latent2_dim=parent_kwargs['latent2_dim'],
            mode_dim=200,
            hidden_units=parent_kwargs['hidden_units'],
            rnn_layers=parent_kwargs['rnn_layers'],
            hidden_rnn_dim=parent_kwargs['hidden_rnn_dim'],
            leaky_slope=parent_kwargs['leaky_slope'],
            state_rep=state_rep
        ).to(self.device)

        # Load pretrained DIAYN skill policy
        data = torch.load(skill_policy_path)
        self.policy = data['evaluation/policy']
        print("Policy loaded")

        # Optimization
        self.latent_optim = Adam(self.latent.parameters(), lr=parent_kwargs['latent_lr'])

        # Memory
        self.memory = MyMemoryDisentangling(
            capacity=parent_kwargs['memory_size'],
            num_sequences=parent_kwargs['num_sequences'],
            observation_shape=self.observation_shape,
            action_shape=self.action_shape,
            device=self.device
        )

        # Log directories
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.images_dir = os.path.join(log_dir, 'images')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        # Summary writer with conversion of hparams
        # (certain types are not aloud for hparam-storage)
        self.writer = SummaryWriter(os.path.join(self.summary_dir, summary_comment),
                                    filename_suffix=self.run_id)
        hparam_dict = parent_kwargs.copy()
        for k, v in hparam_dict.items():
            if isinstance(v, type(None)):
                hparam_dict[k] = 'None'
            if isinstance(v, list):
                hparam_dict[k] = torch.Tensor(v)
        hparam_dict['hidden_units'] = torch.Tensor(parent_kwargs['hidden_units'])
        self.writer.add_hparams(hparam_dict=hparam_dict,
                                metric_dict={})

        # Set hyperparameters
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.initial_latent_steps = parent_kwargs['initial_latent_steps']
        self.num_sequences = num_sequences
        self.num_steps = parent_kwargs['num_steps']
        self.batch_size = parent_kwargs['batch_size']
        self.latent_batch_size = parent_kwargs['latent_batch_size']
        self.start_steps = parent_kwargs['start_steps']
        self.grad_clip = parent_kwargs['grad_clip']
        self.training_log_interval = parent_kwargs['training_log_interval']
        self.learning_log_interval = parent_kwargs['learning_log_interval']

    def get_skill_action_pixel(self):
        obs_state_space = self.env.get_state_obs()
        action, info = self.policy.get_action(obs_state_space)
        return action

    def get_skill_action_state_rep(self, observation):
        action, info = self.policy.get_action(observation)
        return action

    def set_policy_skill(self, skill):
        self.policy.stochastic_policy.skill = skill

    def train_episode(self):
        self.episodes += 1
        episode_steps = 0
        episode_reward = 0.
        done = False
        state = self.env.reset()
        self.memory.set_initial_state(state)
        skill = np.random.randint(self.policy.stochastic_policy.skill_dim)
        self.set_policy_skill(skill)

        next_state = state
        while not done:
            action = self.get_skill_action_state_rep(next_state) if self.state_rep \
                else self.get_skill_action_pixel()
            next_state, reward, done, _ = self.env.step(action)
            self.steps += self.action_repeat
            episode_steps += self.action_repeat
            episode_reward += reward

            self.memory.append(action, reward, next_state, done)

            if self.is_update():
                if self.learning_steps < self.initial_latent_steps:
                    print('-' * 60)
                    print('Learning the disentangled model only...')
                    for _ in tqdm(range(self.initial_latent_steps)):
                        self.learning_steps += 1
                        self.learn_latent()
                    print('Finished learning the disentangled model')
                    print('-' * 60)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'skill: {skill:<4}  ')

        self.save_models()

    def learn_latent(self):
        # Sample sequence
        images_seq, actions_seq, rewards_seq, dones_seq = \
            self.memory.sample_latent(self.latent_batch_size)

        # Calc loss
        latent_loss = self.calc_latent_loss(
            images_seq, actions_seq, rewards_seq, dones_seq)

        # Backprop
        update_params(
            self.latent_optim, self.latent, latent_loss, self.grad_clip)

        # Write net params
        if self._is_log(self.learning_log_interval // 2):
            self.latent.write_net_params(self.writer, self.learning_steps)

    def calc_latent_loss(self, images_seq, actions_seq, rewards_seq,
                         dones_seq):
        # Get features from images
        features_seq = self.latent.encoder(images_seq)

        # Sample from posterior dynamics
        ((latent1_post_samples, latent2_post_samples, mode_post_sample),
            (latent1_post_dists, latent2_post_dists, mode_post_dist)) = \
            self.latent.sample_posterior(actions_seq=actions_seq,
                                         features_seq=features_seq)

        # Sample from prior dynamics
        ((latent1_pri_samples, latent2_pri_samples, mode_pri_sample),
            (latent1_pri_dists, latent2_pri_dists, mode_pri_dist)) = \
            self.latent.sample_prior(features_seq)

        # KL divergence losses
        kld_losses = calc_kl_divergence([mode_post_dist], [mode_pri_dist]) + \
            calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

        # Log likelihood loss of generated actions
        mode_post_sample = mode_post_sample.unsqueeze(1)
        mode_post_samples = mode_post_sample.expand(
            mode_post_sample.size(0), latent1_post_samples.size(1), mode_post_sample.size(2))
        actions_seq_dists = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples, mode_post_samples])
        log_likelihood_loss = actions_seq_dists.log_prob(actions_seq).mean(dim=0).sum()

        # Loss
        latent_loss = kld_losses - log_likelihood_loss

        # Logging
        if self._is_log(self.learning_log_interval):

            # Reconstruction error
            reconst_error = (actions_seq - actions_seq_dists.loc) \
                .pow(2).mean(dim=(0, 1)).sum()
            self._summary_log('stats/reconst_error', reconst_error)
            print('reconstruction error: %f', reconst_error)

            # KL divergence
            mode_kldiv = calc_kl_divergence([mode_post_dist], [mode_pri_dist])
            seq_kldiv = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)
            kldiv = mode_kldiv + seq_kldiv
            self._summary_log('stats/mode_kldiv', mode_kldiv)
            self._summary_log('stats/seq_kldiv', seq_kldiv)
            self._summary_log('stats/kldiv', kldiv)

            # Log Likelyhood
            self._summary_log('stats/log-likelyhood', log_likelihood_loss)

            # Loss
            self._summary_log('loss/network', latent_loss)

            # Save Model
            self.latent.save(os.path.join(self.model_dir, 'model.pth'))

            # Reconstruction test
            rand_batch = np.random.choice(actions_seq.size(0))
            action_dim = actions_seq.size(2)
            gt_actions = actions_seq[rand_batch].detach().cpu()
            post_actions = actions_seq_dists.loc[rand_batch].detach().cpu()
            for dim in range(action_dim):
                plt.interactive(False)
                plt.plot(gt_actions[:, dim].numpy())
                plt.plot(post_actions[:, dim].numpy())
                fig = plt.gcf()
                self.writer.add_figure('Reconstruction test dim' + str(dim), fig,
                                       global_step=self.learning_steps)
                plt.clf()

            #with torch.no_grad():
            #    pri_actions = self.latent.decoder(
            #        [latent1_pri_samples[:1], latent2_pri_samples[:1]]
            #    ).loc[0].detach().cpu()
            #    cond_pri_samples, _ = self.latent.sample_prior(
            #        features_seq[:1], actions_seq[:1, 0]
            #    )
            #    cond_pri_actions = self.latent.decoder(
            #        cond_pri_samples.loc[0].detach().cpu()
            #    )

            #actions = torch.cat(
            #    [gt_actions, post_actions, cond_pri_actions, pri_actions], dim=-2
            #)

        return latent_loss

    def save_models(self):
        path_name = os.path.join(self.model_dir, self.run_id)
        self.latent.save(path_name + 'model_state_dict.pth')
        torch.save(self.latent, path_name + 'whole_model.pth')
        #np.save(self.memory, os.path.join(self.model_dir, 'memory.pth'))

    def _is_log(self, log_interval):
        return True if self.learning_steps % log_interval == 0 else False

    def _summary_log(self, data_name, data):
        if type(data) == torch.Tensor:
            data = data.detach().cpu().item()
        self.writer.add_scalar(data_name, data, self.learning_steps)
