"""
Inference script.

Loads a trained checkpoint and runs a forward pass on a single sample,
printing the predicted [steering, throttle] action.

Usage:
    python inference.py --checkpoint checkpoints/best_mlp.pt
    python inference.py --checkpoint checkpoints/best_gru.pt
"""

import argparse

import torch

from data.dataset import DummyExpertDataset
from models.perception import PerceptionModule
from models.policy import GRUPolicy, MLPPolicy


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index of the sample to run inference on")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["args"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")
    print(f"Config: {cfg}\n")

    policy_type = cfg["policy"]

    # Rebuild models
    perception = PerceptionModule(feature_dim=cfg["feature_dim"]).to(device)
    perception.load_state_dict(ckpt["perception"])
    perception.eval()

    if policy_type == "gru":
        policy = GRUPolicy(
            feature_dim=cfg["feature_dim"],
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            hidden_dim=cfg["hidden_dim"],
        ).to(device)
    else:
        policy = MLPPolicy(
            feature_dim=cfg["feature_dim"],
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            hidden_dim=cfg["hidden_dim"],
        ).to(device)

    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # Load one sample
    seq_len = cfg.get("seq_len", 1) if policy_type == "gru" else 1
    dataset = DummyExpertDataset(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        seq_len=seq_len,
    )
    sample = dataset[args.sample_idx]

    image       = sample["image"].unsqueeze(0).to(device)       # (1, ...)
    state       = sample["state"].unsqueeze(0).to(device)
    prev_action = sample["prev_action"].unsqueeze(0).to(device)
    action_gt   = sample["action"]

    # Forward pass
    with torch.no_grad():
        z = perception(image)
        if policy_type == "gru":
            pred, _ = policy(z, state, prev_action)
        else:
            pred = policy(z, state, prev_action)

    pred_np = pred.squeeze(0).cpu().numpy()
    gt_np   = action_gt.numpy()

    print(f"Sample index     : {args.sample_idx}")
    print(f"Predicted action : steering={pred_np[..., 0]:.4f}  "
          f"throttle={pred_np[..., 1]:.4f}")
    print(f"Ground truth     : steering={gt_np[..., 0]:.4f}  "
          f"throttle={gt_np[..., 1]:.4f}")


if __name__ == "__main__":
    main()
