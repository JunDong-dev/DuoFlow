"""Shared model, normalization, and SMPL-X I/O for the standalone Motion VAE."""

from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/motion_vae.yaml"
DEFAULT_STATS = REPO_ROOT / "assets/motion_vae_stats"


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    model = config["model"]
    if model["inputdim"] != 333 or model["down_t"] != 1 or model["stride_t"] != 3:
        raise ValueError("The standalone pipeline requires 333D motion and temporal stride three")
    return config


def load_model(config, checkpoint=None, device="cpu"):
    from models.tae3_parts import Causal_HumanTAE3parts

    model = Causal_HumanTAE3parts(**config["model"])
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("net", payload) if isinstance(payload, dict) else payload
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.to(device)


def load_stats(directory=DEFAULT_STATS):
    directory = Path(directory)
    mean = np.load(directory / "mean.npy").astype(np.float32)
    std = np.load(directory / "std.npy").astype(np.float32)
    if mean.shape != (333,) or std.shape != (333,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics must be finite 333D vectors")
    return mean, np.maximum(std, 1e-8)


def sample_ids(data_root, split="train", limit=None):
    root = Path(data_root)
    path = root / "split" / (split + ".txt")
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = list(dict.fromkeys(Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        ids = ids[:limit]
    if not ids:
        raise ValueError("The requested split is empty")
    for sample in ids:
        if not (root / "motion_smplx" / (sample + ".npz")).is_file():
            raise FileNotFoundError(root / "motion_smplx" / (sample + ".npz"))
    return ids


def read_motion(path):
    with np.load(path) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 165 or trans.shape != (len(poses), 3) or len(poses) == 0:
        raise ValueError(f"Expected poses[T,165] and trans[T,3]: {path}")
    matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 55, 3, 3)
    features = np.concatenate((trans, matrices[:, :, :2, :].reshape(-1, 330)), axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite motion features: {path}")
    return features


def save_motion(path, features):
    features = np.asarray(features, dtype=np.float32)
    np.save(str(path) + ".npy", features)
    d6 = torch.from_numpy(features[:, 3:].copy()).reshape(-1, 6)
    b1 = F.normalize(d6[:, :3], dim=-1)
    a2 = d6[:, 3:]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    matrices = torch.stack((b1, b2, b3), dim=-2).numpy()
    poses = Rotation.from_matrix(matrices).as_rotvec().reshape(-1, 165).astype(np.float32)
    np.savez_compressed(str(path) + ".npz", poses=poses, trans=features[:, :3], mocap_frame_rate=30)


def encode(model, normalized, device):
    frames = len(normalized)
    padding = (-frames) % 3
    padded = np.pad(normalized, ((0, padding), (0, 0)), mode="edge")
    latent, _, _ = model.encode(torch.from_numpy(padded).unsqueeze(0).to(device))
    return latent, frames


def decode(model, latent, mean, std, frames=None):
    reconstructed = model.forward_decoder(latent).squeeze(0).float().cpu().numpy()
    if frames is not None:
        reconstructed = reconstructed[:frames]
    reconstructed = reconstructed * std + mean
    if not np.isfinite(reconstructed).all():
        raise ValueError("The decoder produced non-finite motion features")
    return reconstructed


def vae_loss(predicted, target, mu, logvar, root_weight=7.0, kl_weight=1.0):
    """Original optimal-sigma Gaussian NLL, KL, and nine-channel root objective."""
    def nll(pred, truth):
        mse = (truth - pred).square().mean().clamp_min(torch.finfo(pred.dtype).tiny)
        log_sigma = -6.0 + F.softplus(0.5 * mse.log() + 6.0)
        return (0.5 * ((truth - pred) / log_sigma.exp()).square() + log_sigma + 0.5 * math.log(2 * math.pi)).sum()

    reconstruction = nll(predicted, target)
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=(1, 2))).mean()
    root = nll(predicted[..., :9], target[..., :9])
    total = reconstruction + kl_weight * kl + root_weight * root
    return total, {"reconstruction": reconstruction, "kl": kl, "root": root}
