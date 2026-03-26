"""
Policy networks.

Input at each timestep:
  z_t        — visual feature from PerceptionModule  (feature_dim,)
  a_{t-1}    — previous action                       (action_dim,)
  s_t        — vehicle state                         (state_dim,)

Output:
  action     — [steering, throttle] in [-1, 1]       (action_dim,)

Two variants:
  MLPPolicy  — stateless, maps concatenated input directly to action
  GRUPolicy  — stateful, uses a GRU to model temporal dependencies
"""

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = feature_dim + state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # constrain output to [-1, 1]
        )

    def forward(
        self,
        z: torch.Tensor,
        state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z           : (B, feature_dim)
            state       : (B, state_dim)
            prev_action : (B, action_dim)
        Returns:
            action      : (B, action_dim)
        """
        x = torch.cat([z, state, prev_action], dim=-1)
        return self.net(x)


class GRUPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        input_dim = feature_dim + state_dim + action_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z           : (B, T, feature_dim)
            state       : (B, T, state_dim)
            prev_action : (B, T, action_dim)
            hidden      : (num_layers, B, hidden_dim) or None
        Returns:
            action      : (B, T, action_dim)
            hidden      : (num_layers, B, hidden_dim)
        """
        x = torch.cat([z, state, prev_action], dim=-1)  # (B, T, input_dim)
        out, hidden = self.gru(x, hidden)                # (B, T, hidden_dim)
        action = self.head(out)
        return action, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
