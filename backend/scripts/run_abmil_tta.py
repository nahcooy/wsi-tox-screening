"""
ABMIL Test-Time Augmentation via feature-patch subsampling.

For each trial: randomly sample `sample_ratio` fraction of patches,
run ABMIL forward pass, collect softmax. Aggregates across n_trials.

Output (stdout, last line): JSON with TTA results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


CLASS_ORDER = ["normal", "abnormal"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_h5",      required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--mil_lab_root",    default="/home/nahcooy/MIL/MIL_260527/MIL-Lab")
    p.add_argument("--n_trials",        type=int,   default=10)
    p.add_argument("--sample_ratio",    type=float, default=0.8)
    p.add_argument("--device",          default="cuda:0")
    p.add_argument("--slide_id",        default="unknown")
    return p.parse_args()


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_features(h5_path: Path) -> np.ndarray:
    with h5py.File(str(h5_path), "r") as f:
        return f["features"][:]


def load_model(mil_lab_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(mil_lab_root))
    from src.builder import create_model
    model = create_model("abmil", num_classes=2).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_subset(model, features: np.ndarray, indices: np.ndarray, device: torch.device) -> np.ndarray:
    subset = features[indices]
    tensor = torch.from_numpy(subset).float().unsqueeze(0).to(device)
    with torch.no_grad():
        results_dict, _ = model(tensor, loss_fn=None, label=None, return_attention=True)
    logits = results_dict["logits"].detach().cpu().numpy().squeeze(0)
    return stable_softmax(logits)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    features = load_features(Path(args.feature_h5))
    n_patches = len(features)
    sample_k  = max(1, int(n_patches * args.sample_ratio))

    model = load_model(Path(args.mil_lab_root), Path(args.checkpoint_path), device)

    trial_softmax: list[np.ndarray] = []
    rng = np.random.default_rng(42)

    for _ in range(args.n_trials):
        idx = rng.choice(n_patches, size=sample_k, replace=False)
        sm  = infer_subset(model, features, idx, device)
        trial_softmax.append(sm)

    arr = np.stack(trial_softmax, axis=0)  # (n_trials, n_classes)
    mean_sm  = arr.mean(axis=0)
    std_sm   = arr.std(axis=0)
    ci95_lo  = np.percentile(arr, 2.5,  axis=0)
    ci95_hi  = np.percentile(arr, 97.5, axis=0)

    pred_votes = [CLASS_ORDER[int(np.argmax(s))] for s in trial_softmax]
    vote_counts = {cls: pred_votes.count(cls) for cls in CLASS_ORDER}
    majority    = max(vote_counts, key=vote_counts.get)

    result = {
        "slide_id":    args.slide_id,
        "n_trials":    args.n_trials,
        "sample_ratio": args.sample_ratio,
        "n_patches_total": n_patches,
        "n_patches_per_trial": sample_k,
        "mean_softmax":  {cls: float(mean_sm[i]) for i, cls in enumerate(CLASS_ORDER)},
        "std_softmax":   {cls: float(std_sm[i])  for i, cls in enumerate(CLASS_ORDER)},
        "ci95_lo":       {cls: float(ci95_lo[i]) for i, cls in enumerate(CLASS_ORDER)},
        "ci95_hi":       {cls: float(ci95_hi[i]) for i, cls in enumerate(CLASS_ORDER)},
        "vote_counts":   vote_counts,
        "majority_vote": majority,
        "trial_softmax": [
            {cls: float(s[i]) for i, cls in enumerate(CLASS_ORDER)}
            for s in trial_softmax
        ],
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
