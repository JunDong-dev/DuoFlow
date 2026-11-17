from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import smplx  # noqa: E402
from common.skeleton import Skeleton  # noqa: E402
from scripts import motion_process as process  # noqa: E402
from utils import paramUtil  # noqa: E402


BODY_HAND_INDICES = list(range(22)) + list(range(25, 55))
HAND_OFFSETS = np.array(
    [[1, 0, 0]] * 15 + [[-1, 0, 0]] * 15,
    dtype=np.float32,
)
RAW_OFFSETS = np.concatenate([paramUtil.t2m_raw_offsets, HAND_OFFSETS], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MotionSpeech SMPL-X clips to CLaM-H's 623-D Motion-X representation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model-root", type=Path, required=True, help="Parent directory containing smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--limit", type=int, default=0, help="0 uses the full split")
    parser.add_argument("--selection-seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def extract_caption(text: str, level: str) -> str:
    match = re.search(rf"<{level}>(.*?)</{level}>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"missing <{level}> caption")
    return " ".join(match.group(1).split())


def selected_ids(args: argparse.Namespace) -> list[str]:
    ids = [line.strip() for line in (args.data_root / "split" / f"{args.split}.txt").read_text().splitlines() if line.strip()]
    if args.limit and args.limit < len(ids):
        rng = np.random.default_rng(args.selection_seed)
        ids = [ids[i] for i in sorted(rng.choice(len(ids), size=args.limit, replace=False).tolist())]
    return ids


def load_batch(data_root: Path, ids: list[str], frame_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses, trans, betas = [], [], []
    for clip_id in ids:
        sample = np.load(data_root / "motion_smplx" / f"{clip_id}.npz")
        poses.append(sample["poses"][frame_index])
        trans.append(sample["trans"][frame_index])
        betas.append(sample["betas"][:10])
    return np.stack(poses), np.stack(trans), np.stack(betas)


def to_joints(model: torch.nn.Module, poses: np.ndarray, trans: np.ndarray, betas: np.ndarray, device: torch.device) -> np.ndarray:
    batch, frames = poses.shape[:2]
    pose = torch.from_numpy(poses.reshape(-1, 165)).to(device)
    transl = torch.from_numpy(trans.reshape(-1, 3)).to(device)
    shape = torch.from_numpy(np.repeat(betas[:, None, :], frames, axis=1).reshape(-1, 10)).to(device)
    expression = torch.zeros(len(pose), 10, device=device)
    with torch.inference_mode():
        output = model(
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:66],
            jaw_pose=pose[:, 66:69],
            leye_pose=pose[:, 69:72],
            reye_pose=pose[:, 72:75],
            left_hand_pose=pose[:, 75:120],
            right_hand_pose=pose[:, 120:165],
            transl=transl,
            betas=shape,
            expression=expression,
            return_verts=False,
        )
    return output.joints[:, BODY_HAND_INDICES].reshape(batch, frames, 52, 3).cpu().numpy()


def configure_motionx_process(target_joints: np.ndarray) -> None:
    process.l_idx1, process.l_idx2 = 5, 8
    process.fid_r, process.fid_l = [8, 11], [7, 10]
    process.face_joint_indx = [2, 1, 17, 16]
    process.r_hip, process.l_hip = 2, 1
    process.joints_num = 52
    process.n_raw_offsets = torch.from_numpy(RAW_OFFSETS)
    process.kinematic_chain = paramUtil.motionx_kinematic_chain
    skeleton = Skeleton(process.n_raw_offsets, process.kinematic_chain, "cpu")
    process.tgt_offsets = skeleton.get_offsets_joints(torch.from_numpy(target_joints[0]))


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    ids = selected_ids(args)
    if not ids:
        raise ValueError("empty split")
    args.output.mkdir(parents=True, exist_ok=True)
    frame_index = np.rint(np.linspace(0, 386, 197)).astype(np.int64)

    model = smplx.create(
        str(args.model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        num_betas=10,
        batch_size=args.batch_size * len(frame_index),
    ).to(device).eval()

    target_id = next(line.strip() for line in (args.data_root / "split" / "train.txt").read_text().splitlines() if line.strip())
    target_model = smplx.create(
        str(args.model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        num_betas=10,
        batch_size=1,
    ).to(device).eval()
    target_pose, target_trans, target_betas = load_batch(args.data_root, [target_id], frame_index[:1])
    target_joints = to_joints(target_model, target_pose, target_trans, target_betas, device)[0]
    del target_model
    configure_motionx_process(target_joints)

    output = np.lib.format.open_memmap(
        args.output / "motions.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(ids), 196, 623),
    )
    captions_l1: list[str] = []
    captions_l2: list[str] = []
    for start in tqdm(range(0, len(ids), args.batch_size), desc=f"prepare-{args.split}"):
        batch_ids = ids[start : start + args.batch_size]
        padded_ids = batch_ids + [batch_ids[-1]] * (args.batch_size - len(batch_ids))
        poses, trans, betas = load_batch(args.data_root, padded_ids, frame_index)
        joints = to_joints(model, poses, trans, betas, device)[: len(batch_ids)]
        for offset, (clip_id, joint_sequence) in enumerate(zip(batch_ids, joints)):
            features, _, _, _ = process.process_file(joint_sequence, 0.002)
            if features.shape != (196, 623) or not np.isfinite(features).all():
                raise ValueError(f"invalid features for {clip_id}: {features.shape}")
            output[start + offset] = features.astype(np.float16)
            caption_text = (args.data_root / "caption" / f"{clip_id}.txt").read_text(encoding="utf-8")
            captions_l1.append(extract_caption(caption_text, "L1"))
            captions_l2.append(extract_caption(caption_text, "L2"))
    output.flush()
    (args.output / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    (args.output / "captions_l1.json").write_text(json.dumps(captions_l1, indent=2), encoding="utf-8")
    (args.output / "captions_l2.json").write_text(json.dumps(captions_l2, indent=2), encoding="utf-8")
    metadata = {
        "source": str(args.data_root),
        "split": args.split,
        "count": len(ids),
        "selection_seed": args.selection_seed,
        "frame_selection": "197 uniformly spaced indices over 387 frames; process_file yields 196 frames",
        "representation": "Motion-X 623-D body-hand tomato representation",
        "target_skeleton_clip": target_id,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
