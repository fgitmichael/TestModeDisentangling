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
from InfoGradEstimation.MIGE import entropy_surrogate, SpectralScoreEstimator

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
                 feature_dim=5,
                 num_sequences=80,
                 cuda=False,
                 ):
        parent_kwargs = dict(
            num_steps=3000000,
            initial_latent_steps=100000,
            batch_size=256,
            latent_batch_size=128,
            num_sequences=num_sequences,
            latent_lr=0.0001,
            feature_dim=feature_dim, # Note: Only used if state_rep == False
            latent1_dim=4,
            latent2_dim=12,
            mode_dim=1,
            hidden_units=[26, 26],
            hidden_rnn_dim=34,
            rnn_layers=2,
            memory_size=1e5,
            leaky_slope=0.2,
            grad_clip=None,
            start_steps=5000,
            training_log_interval=100,
            learning_log_interval=50,
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
            mode_dim=parent_kwargs['mode_dim'],
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

        # MI-Gradient score estimators
        self.spectral_j = SpectralScoreEstimator(n_eigen_threshold=0.99)
        self.spectral_m = SpectralScoreEstimator(n_eigen_threshold=0.99)

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
        skill = np.random.choice([3, 4, 5, 7, 9], 1).item()
        self.set_policy_skill(skill)

        next_state = state
        while not done and episode_steps <= self.num_sequences + 2:
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
        if self._is_log(self.learning_log_interval * 5):
            self.latent.write_net_params(self.writer, self.learning_steps)

    def calc_latent_loss(self,
                         images_seq,
                         actions_seq,
                         rewards_seq,
                         dones_seq):
        # Get features from images
        features_seq = self.latent.encoder(images_seq)

        # Sample from posterior dynamics
        ((latent1_post_samples, latent2_post_samples, mode_post_samples),
            (latent1_post_dists, latent2_post_dists, mode_post_dist)) = \
            self.latent.sample_posterior(actions_seq=actions_seq,
                                         features_seq=features_seq)

        # Sample from prior dynamics
        ((latent1_pri_samples, latent2_pri_samples, mode_pri_samples),
            (latent1_pri_dists, latent2_pri_dists, mode_pri_dist)) = \
            self.latent.sample_prior(features_seq)

        # KL divergence losses
        latent_kld = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)
        latent1_dim = latent1_post_samples.size(2)
        seq_length = latent1_post_samples.size(1)
        latent_kld /= latent1_dim
        latent_kld /= seq_length

        mode_kld = calc_kl_divergence([mode_post_dist], [mode_pri_dist])
        mode_dim = mode_post_samples.size(2)
        mode_kld /= mode_dim
        kld_losses = mode_kld + latent_kld

        # Log likelihood loss of generated actions
        actions_seq_dists = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples, mode_post_samples])
        log_likelihood = actions_seq_dists.log_prob(actions_seq).mean(dim=0).mean()

        # Log likelihood loss of generated actions with latent dynamic priors and mode
        # posterior
        actions_seq_dists_mode = self.latent.decoder(
            [latent1_pri_samples.detach(), latent2_pri_samples.detach(), mode_post_samples])
        ll_dyn_pri_mode_post = actions_seq_dists_mode.log_prob(actions_seq).mean(dim=0).mean()

        # Log likelihood loss of generated actions with latent dynamic posteriors and
        # mode prior
        action_seq_dists_dyn = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples, mode_pri_samples]
        )
        ll_dyn_post_mode_pri = action_seq_dists_dyn.log_prob(actions_seq).mean(dim=0).mean()

        # Maximum Mean Discrepancy (MMD)
        mode_pri_sample = mode_pri_samples[:, 0, :]
        mode_post_sample = mode_post_samples[:, 0, :]
        mmd_mode = self.compute_mmd_tutorial(mode_pri_sample, mode_post_sample)
        mmd_latent = 0
        #latent1_post_samples_trans = latent1_post_samples.transpose(0, 1)
        #latent1_pri_samples_trans = latent1_pri_samples.transpose(0, 1)
        #for idx in range(latent1_post_samples_trans.size(0)):
        #    mmd_latent += self.compute_mmd_tutorial(latent1_pri_samples_trans[idx],
        #                                            latent1_post_samples_trans[idx])
        latent1_post_samples_trans = latent1_post_samples.\
            view(latent1_post_samples.size(0), -1)
        latent1_pri_samples_trans = latent1_pri_samples.\
            view(latent1_pri_samples.size(0), -1)
        mmd_latent = self.compute_mmd_tutorial(
            latent1_pri_samples_trans, latent1_post_samples_trans)
        mmd_mode_weighted = mmd_mode
        mmd_latent_weighted = mmd_latent * 100
        mmd_loss = mmd_latent_weighted + mmd_mode_weighted

        # MI-Gradient
        batch_size = mode_post_samples.size(0)
        features_actions_seq = torch.cat([features_seq[:, :-1, :],
                                          actions_seq], dim=2)
        xs = features_actions_seq.view(batch_size, -1)
        ys = mode_post_samples.view(batch_size, -1)
        xs_ys = torch.cat([xs, ys], dim=1)
        gradient_estimator = entropy_surrogate(self.spectral_j, xs_ys) \
                             - entropy_surrogate(self.spectral_m, ys)


        # Loss
        reg_weight = 1000.
        alpha = 1.
        kld_info_weighted = (1. - alpha) * kld_losses
        mmd_info_weighted = (alpha + reg_weight - 1.) * mmd_loss
        #latent_loss = kld_info_weighted - log_likelihood + mmd_info_weighted
        latent_loss = -log_likelihood + latent_kld + mmd_mode_weighted - gradient_estimator
        latent_loss *= 100

        # Logging
        if self._is_log(self.learning_log_interval):

            # Reconstruction error
            reconst_error = (actions_seq - actions_seq_dists.loc) \
                .pow(2).mean(dim=(0, 1)).sum()
            self._summary_log('stats/reconst_error', reconst_error)
            print('reconstruction error: %f', reconst_error)

            reconst_err_mode_post = (actions_seq - actions_seq_dists_mode.loc)\
                .pow(2).mean(dim=(0,1)).sum()
            reconst_err_dyn_post = (actions_seq - action_seq_dists_dyn.loc)\
                .pow(2).mean(dim=(0, 1)).sum()
            self._summary_log('stats/reconst_error mode post', reconst_err_mode_post)
            self._summary_log('stats/reconst_error dyn post', reconst_err_dyn_post)

            # KL divergence
            mode_kldiv_standard = calc_kl_divergence([mode_post_dist], [mode_pri_dist])
            seq_kldiv_standard = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)
            kldiv_standard = mode_kldiv_standard + seq_kldiv_standard

            self._summary_log('stats_kldiv_standard/kldiv_standard', kldiv_standard)
            self._summary_log('stats_kldiv_standard/mode_kldiv_standard', mode_kldiv_standard)
            self._summary_log('stats_kldiv_standard/seq_kldiv_standard', seq_kldiv_standard)

            self._summary_log('stats_kldiv/mode_kldiv_used_for_loss', mode_kld)
            self._summary_log('stats_kldiv/latent_kldiv_used_for_loss', latent_kld)
            self._summary_log('stats_kldiv/klddiv used for loss', kld_losses)

            # Log Likelyhood
            self._summary_log('stats/log-likelyhood', log_likelihood)
            self._summary_log('stats/log-likelyhood dyn pri,  mode post', ll_dyn_pri_mode_post)
            self._summary_log('stats/log-likelyhood dyn post, mode pri', ll_dyn_post_mode_pri)

            # MMD
            self._summary_log('stats_mmd/mmd_weighted', mmd_info_weighted)
            self._summary_log('stats_mmd/kld_weighted', kld_info_weighted)
            self._summary_log('stats_mmd/mmd_mode_weighted', mmd_mode_weighted)
            self._summary_log('stats_mmd/mmd_latent_weighted', mmd_latent_weighted)

            # MI-Grad
            self._summary_log('stats_mi/mi_grad_est', gradient_estimator)

            # Loss
            self._summary_log('loss/network', latent_loss)

            # Save Model
            self.latent.save(os.path.join(self.model_dir, 'model.pth'))

            # Reconstruction Test
            rand_batch_idx = np.random.choice(actions_seq.size(0))
            self._reconstruction_post_test(rand_batch_idx,
                                           actions_seq,
                                           actions_seq_dists)
            self._reconstruction_mode_post_test(rand_batch_idx,
                                                actions_seq,
                                                mode_post_samples,
                                                latent1_pri_samples,
                                                latent2_pri_samples)
            self._reconstruction_dyn_post_test(rand_batch_idx,
                                               actions_seq,
                                               latent1_post_samples,
                                               latent2_post_samples,
                                               mode_pri_samples)

        return latent_loss

    def _reconstruction_post_test(self,rand_batch_idx, actions_seq, actions_seq_dists):
        """
        Test reconstruction of inferred posterior
        Args:
            rand_batch_idx      : which part of batch to use
            actions_seq         : actions sequence sampled from data
            actions_seq_dists   : distribution of inferred posterior actions (reconstruction)
        """
        # Reconstruction test
        rand_batch = rand_batch_idx
        action_dim = actions_seq.size(2)
        gt_actions = actions_seq[rand_batch].detach().cpu()
        post_actions = actions_seq_dists.loc[rand_batch].detach().cpu()
        for dim in range(action_dim):
            fig = self._reconstruction_test_plot(dim, gt_actions, post_actions)
            self.writer.add_figure('Reconst_post_test/reconst test dim' + str(dim), fig,
                                   global_step=self.learning_steps)
            plt.clf()

    def _reconstruction_mode_post_test(self,
                                       rand_batch_idx,
                                       actions_seq,
                                       latent1_pri_samples,
                                       latent2_pri_samples,
                                       mode_post_samples):
        """
        Test if mode inference works
        Args:
            rand_batch_idx      : which part of batch to use
            actions_seq         : actions sequence sampled from data
            mode_post_samples   : Sample from the inferred mode posterior distribution
            latent1_pri_samples : Samples from the dynamics prior
            latent2_pri_samples : Samples from the dynamics prior
        """
        # Use random sample from batch
        rand_batch = rand_batch_idx

        # Decode
        actions_seq_dists = self.latent.decoder([latent1_pri_samples[rand_batch],
                                                 latent2_pri_samples[rand_batch],
                                                 mode_post_samples[rand_batch]])

        # Reconstruction test
        action_dim = actions_seq.size(2)
        gt_actions = actions_seq[rand_batch].detach().cpu()
        post_actions = actions_seq_dists.loc.detach().cpu()

        # Plot
        for dim in range(action_dim):
            fig = self._reconstruction_test_plot(dim, gt_actions, post_actions)
            self.writer.add_figure('Reconst_mode_post_test/reconst test dim' + str(dim),
                                   fig,
                                   global_step=self.learning_steps)

    def _reconstruction_dyn_post_test(self,
                                      rand_batch_idx,
                                      actions_seq,
                                      latent1_post_samples,
                                      latent2_post_samples,
                                      mode_pri_samples):
        """
        Test the influence of the latent dynamics inference
        """
        # Use random sample from batch
        rand_batch = rand_batch_idx

        # Decode
        actions_seq_dists = self.latent.decoder([latent1_post_samples[rand_batch],
                                                 latent2_post_samples[rand_batch],
                                                 mode_pri_samples[rand_batch]])

        # Reconstruction test
        action_dim = actions_seq.size(2)
        gt_actions = actions_seq[rand_batch].detach().cpu()
        post_actions = actions_seq_dists.loc.detach().cpu()

        # Plot
        for dim in range(action_dim):
            fig = self._reconstruction_test_plot(dim, gt_actions, post_actions)
            self.writer.add_figure('Reconst_dyn_post_test/reconst test dim' + str(dim),
                                   fig,
                                   global_step=self.learning_steps)

    def _reconstruction_test_plot(self, dim, gt_actions, post_actions):
        plt.interactive(False)
        axes = plt.gca()
        axes.set_ylim([-1.5, 1.5])
        plt.plot(gt_actions[:, dim].numpy())
        plt.plot(post_actions[:, dim].numpy())
        fig = plt.gcf()
        return fig

    def compute_kernel_tutorial(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd_tutorial(self, x, y):
        assert x.shape == y.shape
        x_kernel = self.compute_kernel_tutorial(x, x)
        y_kernel = self.compute_kernel_tutorial(y, y)
        xy_kernel = self.compute_kernel_tutorial(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

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
