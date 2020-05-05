from memory.lazy import LazySequenceBuff, LazyMemory
from collections import deque
import numpy as np
import torch
'''
Inheritence of the Buffers of 'lazy' for the use of disentnagling
Minor changes:
    - As actions are supposed to be generated, now sequence buffer of actions is 
      longer than state buffer (sample_posterior needs this set up)
'''

class MySeqBufferDisentangling(LazySequenceBuff):
    def __init__(self, num_sequences=8):
        super(MySeqBufferDisentangling, self).__init__(num_sequences)

    def reset(self):
        self.memory = {
            'state': deque(maxlen=self.num_sequences),
            'action': deque(maxlen=self.num_sequences + 1),
            'reward': deque(maxlen=self.num_sequences),
            'done': deque(maxlen=self.num_sequences)}

    def __len__(self):
        return len(self.memory['action'])

class MyMemoryDisentangling(LazyMemory):
    def __init__(self, **kwargs):
        super(MyMemoryDisentangling, self).__init__(**kwargs)

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        self.buff = MySeqBufferDisentangling(num_sequences=self.num_sequences)

    def sample_latent(self, batch_size):
        '''
        Returns:
            states_seq   : (N, S, *observation_shape) shaped tensor.
            actions_seq  : (N, S+1, *action_shape) shaped tensor.
            rewards_seq  : (N, S, 1) shaped tensor.
            dones_seq    : (N, S, 1) shaped tensor.
        '''
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states_seq = np.empty((
            batch_size, self.num_sequences, *self.observation_shape),
            dtype=np.uint8)
        actions_seq = np.empty((
            batch_size, self.num_sequences + 1, *self.action_shape),
            dtype=np.float32)
        rewards_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.float32)
        dones_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.bool)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states_seq[i, ...] = self['state'][index]
            actions_seq[i, ...] = self['action'][index]
            rewards_seq[i, ...] = self['reward'][index]
            dones_seq[i, ...] = self['done'][index]

        states_seq = torch.ByteTensor(states_seq).to(self.device).float()/255.
        actions_seq = torch.FloatTensor(actions_seq).to(self.device)
        rewards_seq = torch.FloatTensor(rewards_seq).to(self.device)
        dones_seq = torch.BoolTensor(dones_seq).to(self.device).float()

        return states_seq, actions_seq, rewards_seq, dones_seq

