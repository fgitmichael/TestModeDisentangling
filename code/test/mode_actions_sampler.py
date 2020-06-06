import torch


from network import ModeDisentanglingNetwork


class ModeActionSampler:

    def __init__(self,
                 latent_network,
                 device='cuda',
                 mode_init=None):

        self.latent_network = latent_network
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialized in reset
        # mode           : mode_dim sized tensor
        self.mode = None

        # latent1_state  : latent1_dim sized tensor of timestep t-1
        self.latent1_state = None

        # latent2_state  : latent2_dim sized tensor of timestep t-1
        self.latent2_state = None

        # Init
        self.reset(mode=mode_init)

    def reset(self, mode=None):
        if mode is None:
            # Set random mode
            self.mode = self.latent_network.sample_mode().\
                to(self.device)
        else:
            self.mode = mode.to(self.device)

        # Latent states
        batch_size = 1
        vector = torch.zeros(batch_size, 1).to(self.device)
        self.latent1_state = self.latent_network. \
            latent1_init_prior(vector).sample().\
            to(self.device)

        self.latent2_state = self.latent_network. \
            latent2_init_prior(self.latent1_state).sample().\
            to(self.device)


    def __call__(self, feature):
        """
        Sample action conditioned on arg mode for auto-regressive sampling

        Args:
            feature        : feature_dim sized tensor of timestep t
        Returns:
            action         : action_dim sized tensor
        """
        with torch.no_grad():
            # Propagate latent state for one step
            # p(z1(t) | z2(t-1), feature(t-1))
            latent1_dist_t = self.latent_network.latent1_prior(
                [self.latent2_state, feature])
            self.latent1_state = latent1_dist_t.sample()
            # p(z2(t) | z1(t), z2(t-1), feature(t-1))
            latent2_dist_t = self.latent_network.latent2_prior(
                [self.latent1_state, self.latent2_state, feature])
            self.latent2_state = latent2_dist_t.sample()

            # Generate action
            action_dist = self.latent_network.decoder(
                latent1_sample=self.latent1_state,
                latent2_sample=self.latent2_state,
                mode_sample=self.mode)

        return action_dist.loc
