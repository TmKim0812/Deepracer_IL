"""
Expert demonstration dataset.

Each sample represents one timestep of expert driving:
  - image   : (C, H, W) camera frame
  - state   : (state_dim,) vehicle state (speed, heading, etc.)
  - prev_action : (action_dim,) action taken at t-1
  - action  : (action_dim,) expert action at t  [steering, throttle]

When no real data is available, DummyExpertDataset generates random
tensors that match the expected shapes so the rest of the pipeline can
be developed and tested end-to-end.
"""

import torch
from torch.utils.data import Dataset


class DummyExpertDataset(Dataset):
    """Randomly generated dataset that mimics expert demonstrations."""

    def __init__(
        self,
        num_samples: int = 1000,
        img_channels: int = 3,
        img_height: int = 120,
        img_width: int = 160,
        state_dim: int = 4,   # e.g. speed, heading, lateral offset, yaw rate
        action_dim: int = 2,  # steering, throttle  (both in [-1, 1])
        seq_len: int = 1,     # >1 for sequential / recurrent training
    ):
        self.num_samples = num_samples
        self.img_shape = (img_channels, img_height, img_width)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        # Pre-generate all data once so DataLoader workers stay deterministic
        torch.manual_seed(42)
        if seq_len == 1:
            self.images = torch.rand(num_samples, *self.img_shape)
            self.states = torch.rand(num_samples, state_dim)
            self.prev_actions = torch.zeros(num_samples, action_dim)
            self.actions = torch.rand(num_samples, action_dim) * 2 - 1  # [-1, 1]
        else:
            # Sequential: shape (N, T, ...)
            self.images = torch.rand(num_samples, seq_len, *self.img_shape)
            self.states = torch.rand(num_samples, seq_len, state_dim)
            self.prev_actions = torch.zeros(num_samples, seq_len, action_dim)
            self.actions = torch.rand(num_samples, seq_len, action_dim) * 2 - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "state": self.states[idx],
            "prev_action": self.prev_actions[idx],
            "action": self.actions[idx],
        }
