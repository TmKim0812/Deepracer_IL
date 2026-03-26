"""
Training script for DeepRacer imitation learning.

Usage:
    python train.py                        # MLP, default settings
    python train.py --policy gru           # GRU policy
    python train.py --epochs 50 --lr 1e-3  # custom hyperparams
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import DummyExpertDataset
from models.perception import PerceptionModule
from models.policy import GRUPolicy, MLPPolicy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mlp", "gru"], default="mlp")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--state_dim", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=8,
                   help="Sequence length (only used for GRU policy)")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(perception, policy, loader, optimizer, criterion, device, policy_type, train=True):
    perception.train(train)
    policy.train(train)

    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            images      = batch["image"].to(device)
            states      = batch["state"].to(device)
            prev_actions = batch["prev_action"].to(device)
            actions_gt  = batch["action"].to(device)

            z = perception(images)

            if policy_type == "gru":
                pred_actions, _ = policy(z, states, prev_actions)
            else:
                pred_actions = policy(z, states, prev_actions)

            loss = criterion(pred_actions, actions_gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Policy type : {args.policy}")

    # Dataset
    seq_len = args.seq_len if args.policy == "gru" else 1
    dataset = DummyExpertDataset(
        num_samples=args.num_samples,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seq_len=seq_len,
    )

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Models
    perception = PerceptionModule(feature_dim=args.feature_dim).to(device)

    if args.policy == "gru":
        policy = GRUPolicy(
            feature_dim=args.feature_dim,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
    else:
        policy = MLPPolicy(
            feature_dim=args.feature_dim,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

    params = list(perception.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(perception, policy, train_loader, optimizer,
                               criterion, device, args.policy, train=True)
        val_loss   = run_epoch(perception, policy, val_loader, optimizer,
                               criterion, device, args.policy, train=False)

        print(f"Epoch [{epoch:>3}/{args.epochs}]  "
              f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, f"best_{args.policy}.pt")
            torch.save({
                "epoch": epoch,
                "perception": perception.state_dict(),
                "policy": policy.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
