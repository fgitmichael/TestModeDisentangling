import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from .base import BaseNetwork, create_linear_network,\
    weights_init_xavier
from .latent import Gaussian, ConstantGaussian, \
    Decoder, Encoder, EncoderStateRep


class LogvarGaussian(Gaussian):
    """
    Only difference to Gaussian should be the way the variance is
    calculated
    """

    def __init__(self, **kwargs):
        # Call base constructor
        super(LogvarGaussian, self).__init__(**kwargs)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, logstd = torch.chunk(x, 2, dim=-1)
            std = torch.exp(logstd) + 1e5

        return Normal(loc=mean, scale=std)


class BiRnn(BaseNetwork):

    def __init__(self,
                 input_dim,
                 hidden_rnn_dim,
                 rnn_layers=1,
                 learn_initial_state=True):
        super(BiRnn, self).__init__()

        # RNN
        # Note: batch_first=True means input and output dims are treated as
        #       (batch, seq, feature)
        self.input_dim = input_dim
        self.hidden_rnn_dim = hidden_rnn_dim
        self.f_rnn = nn.GRU(self.input_dim, self.hidden_rnn_dim,
                            num_layers=rnn_layers,
                            bidirectional=True)

        # Noisy hidden init state
        # Note: Only works with GRU right now
        self.learn_init = learn_initial_state
        if self.learn_init:
            # Initial state (dim: num_layers * num_directions, batch, hidden_size)
            self.init_network = Gaussian(input_dim=1,
                                         output_dim=self.f_rnn.hidden_size,
                                         hidden_units=[256])

    def forward(self, x):
        num_sequence = x.size(0)
        batch_size = x.size(1)

        # Initial state (dim: num_layers * num_directions, batch, hidden_size)
        if self.learn_init:
            num_directions = 2 if self.f_rnn.bidirectional else 1
            init_input = torch.ones(self.f_rnn.num_layers * num_directions,
                                    batch_size,
                                    1).to(x.device)
            hidden_init = self.init_network(init_input).rsample()

            # LSTM recursion and extraction of the ends of the two directions
            # (front: end of the forward pass, back: end of the backward pass)
            rnn_out, _ = self.f_rnn(x, hidden_init)
        else:
            # Don't use learned initial state and rely on pytorch init
            rnn_out, _ = self.f_rnn(x)

        # Split into the two directions
        (forward_out, backward_out) = torch.chunk(rnn_out, 2, dim=2)

        # Get the ends of the two directions
        front = forward_out[num_sequence - 1, :, :]
        back = backward_out[0, :, :]

        # Stack along hidden_dim and return
        return torch.cat([front, back], dim=1)


# TODO: Move this class as inner class to ModeDisentanglingNetwork as it is
#      too sophisticated
class ModeEncoder(BaseNetwork):

    def __init__(self,
                 feature_shape,
                 action_shape,
                 output_dim,  # typically mode_dim
                 hidden_rnn_dim,
                 hidden_units,
                 rnn_layers
                 ):
        super(ModeEncoder, self).__init__()

        self.f_rnn_features = BiRnn(feature_shape,
                                    hidden_rnn_dim=hidden_rnn_dim,
                                    rnn_layers=rnn_layers)
        self.f_rnn_actions = BiRnn(action_shape,
                                   hidden_rnn_dim=hidden_rnn_dim,
                                   rnn_layers=rnn_layers)

        # Concatenation of 2*hidden_rnn_dim from the features rnn and
        # 2*hidden_rnn_dim from actions rnn, hence input dim is 4*hidden_rnn_dim
        self.f_dist = Gaussian(input_dim=4 * hidden_rnn_dim,
                               output_dim=output_dim,
                               hidden_units=hidden_units)

    def forward(self, features_seq, actions_seq):
        feat_res = self.f_rnn_features(features_seq)
        act_res = self.f_rnn_actions(actions_seq)
        rnn_result = torch.cat([feat_res, act_res], dim=1)

        # Feed result into Gaussian layer
        return self.f_dist(rnn_result)


class ModeEncoderCombined(BaseNetwork):

    def __init__(self,
                 feature_shape,
                 action_shape,
                 output_dim, # typicall mode_dim
                 hidden_rnn_dim,
                 hidden_units,
                 rnn_layers):
        super(BaseNetwork, self).__init__()

        self.rnn = BiRnn(feature_shape + action_shape,
                           hidden_rnn_dim=hidden_rnn_dim,
                           rnn_layers=rnn_layers)

        self.mode_dist = Gaussian(input_dim=2*hidden_rnn_dim,
                                  output_dim=output_dim,
                                  hidden_units=hidden_units)


    def forward(self, features_seq, actions_seq):
        # State-seq-len is always shorter by one than action_seq_len, but to stack the
        # sequence need to have the same length. Solution: learn the missing element in the state seq
        # Solution: discard last action
        assert features_seq.size(0)+1 == actions_seq.size(0)
        actions_seq = actions_seq[:-1, :, :]

        seq = torch.cat([features_seq, actions_seq], dim=2)

        rnn_result = self.rnn(seq)

        return self.mode_dist(rnn_result)


class ModeDisentanglingNetwork(BaseNetwork):

    def __init__(self,
                 observation_shape,
                 action_shape,
                 state_rep,
                 feature_dim,
                 latent1_dim,
                 latent2_dim,
                 mode_dim,
                 hidden_units,
                 hidden_rnn_dim,
                 rnn_layers,
                 leaky_slope=0.2):
        super(ModeDisentanglingNetwork, self).__init__()

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            latent1_dim, latent2_dim, hidden_units, leaky_slope=leaky_slope)
        # p(z1(t+1) | z2(t), x(t))
        self.latent1_prior = Gaussian(
            latent2_dim + feature_dim, latent1_dim, hidden_units,
            leaky_slope=leaky_slope)
        # p(z2(t+1) | z1(t+1), z2(t), x(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + feature_dim, latent2_dim,
            hidden_units, leaky_slope=leaky_slope)
        # p(m) = N(0,I)
        self.mode_prior = ConstantGaussian(mode_dim)

        # q(z1(0) | action(0))
        self.latent1_init_posterior = Gaussian(
            action_shape[0], latent1_dim, hidden_units, leaky_slope=leaky_slope)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | action(t+1), z2(t), x(t))
        self.latent1_posterior = Gaussian(
            action_shape[0] + latent2_dim + feature_dim, latent1_dim,
            hidden_units, leaky_slope=leaky_slope)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior
        # q(m | features(1:T-1), actions(1:T))
        self.mode_posterior = ModeEncoderCombined(feature_dim,
                                                  action_shape[0],
                                                  output_dim=mode_dim,
                                                  hidden_rnn_dim=hidden_rnn_dim,
                                                  hidden_units=hidden_units,
                                                  rnn_layers=rnn_layers,
                                                  )

        # feat(t) = x(t) : This encoding is performed deterministically.
        if state_rep:
            # State representation
            self.encoder = EncoderStateRep(observation_shape[0], feature_dim)
        else:
            # Conv-nets for pixel observations
            self.encoder = Encoder(observation_shape[0], feature_dim,
                                   leaky_slope=leaky_slope)

        # p(u(t) | z2(t), z1(t), m)
        # TODO: Test if fixed variance performs better than learned variance (as in VAE)
        self.decoder = Gaussian(
            latent1_dim + latent2_dim + mode_dim,
            action_shape[0],
            hidden_units,
            std=1e-1,
            leaky_slope=leaky_slope)

    def sample_prior(self, features_seq, init_actions=None):
        """
        Sample from prior dynamics (with conditioning on initial actions)

        Args:
            features_seq      : (N, S, *feature_shape) tensor of feature sequences
            init_actions     : (N, *action_shape) tensor of initial actions or None
        Returns:
            latent1_samples  : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples  : (N, S+1, L2) tensor of sampled latent vectors.
            mode_samples     : (N, S+1, mode_dim) tensor of sampled mode vectors.
            latent1_dists    : (S+1) length list of (N, L1) distributions.
            latent2_dists    : (S+1) length list of (N, L2) distributions.

        """
        num_sequences = features_seq.size(1)
        features_seq = torch.transpose(features_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                # Condition on initial actions
                if init_actions is not None:
                    # q(z1(0) | action(0))
                    latent1_dist = self.latent1_init_posterior(init_actions)
                    latent1_sample = latent1_dist.rsample()
                    # q(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_posterior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

                # Not conditionning
                else:
                    # p(z1(0)) = N(0,I)
                    latent1_dist = self.latent1_init_prior(features_seq[t])
                    latent1_sample = latent1_dist.rsample()
                    # p(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_prior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

            else:
                # p(z1(t) | z2(t-1), feature(t-1))
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t - 1], features_seq[t - 1]])
                latent1_sample = latent1_dist.rsample()
                # p(z2(t) | z1(t), z2(t-1), feature(t-1))
                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t - 1], features_seq[t - 1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        mode_dist = self.mode_prior(features_seq[0])
        mode_sample = mode_dist.rsample()
        mode_samples = self._broadcast_mode_samples(mode_sample, latent1_samples)

        return (latent1_samples, latent2_samples, mode_samples), \
               (latent1_dists, latent2_dists, mode_dist)

    def sample_posterior(self, actions_seq, features_seq):
        """
        Sample from posterior dynamics and mode

        Args:
            actions_seq  : (N, S+1, *action_space) tensor of action sequences
            features_seq : (N, S, 256) tensor of feature sequences
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors
            mode_samples    : (N, S+1, mode_dim) tensor of sampled modes
            latent1_dists   : (S+1) length list of (N, L1) distributions
            latent2_dists   : (S+1) length list of (N, L2) distributions
            mode_dist       : scalar vector of (N, mode_dim) distributions
        """
        num_sequences = features_seq.size(1)
        actions_seq = torch.transpose(actions_seq, 0, 1)
        features_seq = torch.transpose(features_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                # q(z1(0) | action(0))
                latent1_dist = self.latent1_init_posterior(actions_seq[t])
                latent1_sample = latent1_dist.rsample()
                # q(z2(0) | z1(0))
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()
            else:
                # q(z1(t) | action(t), z2(t-1), features(t-1))
                latent1_dist = self.latent1_posterior(
                    [actions_seq[t], latent2_samples[t - 1], features_seq[t - 1]])
                latent1_sample = latent1_dist.rsample()
                # q(z2(t) | z1(t), z2(t-1), features(t-1))
                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t - 1], features_seq[t - 1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        assert features_seq.size(0) == num_sequences
        assert actions_seq.size(0) == num_sequences + 1
        mode_dist = self.mode_posterior(features_seq=features_seq,
                                        actions_seq=actions_seq)
        mode_sample = mode_dist.rsample()
        mode_samples = self._broadcast_mode_samples(mode_sample, latent1_samples)

        return (latent1_samples, latent2_samples, mode_samples), \
               (latent1_dists, latent2_dists, mode_dist)

    def _broadcast_mode_samples(self, mode_sample, latent1_samples):
        """
        Args:
        mode_sample        : (N, 1, mode_dim) Tensor
                             sample of mode variable that will be repeated to the length
                             of the sequence in latent1_samples
        latent1_samples    : (N, S+1, L1) Tensor
                             Used to get the dimension

        Returns:
        mode_samples       : (N, S+1, mode_dim) Tensor
        """
        assert len(latent1_samples.shape) == 3, 'latent1_samples is not a three dimensional' \
                                                'Tensor'

        mode_sample = mode_sample.unsqueeze(1)
        mode_samples = mode_sample.expand(
            mode_sample.size(0), latent1_samples.size(1), mode_sample.size(2))

        return mode_samples

    def sample_mode(self):
        """
        Sample from mode prior
        """
        batch_size = 1
        return self.mode_prior(torch.zeros(batch_size, 1)).sample()

        return action


