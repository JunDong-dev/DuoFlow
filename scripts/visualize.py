#!/usr/bin/env python3
"""Render a motion sequence as 1-frame-per-second snapshots on a unified ground plane.

Takes an EMAGE-style SMPL-X npz (poses (T,165) / betas / trans), samples one pose
every `--stride` frames, places the snapshots side by side on a checkered ground
plane and renders a single PNG with pyrender (run under xvfb-run for headless GL).
"""
import argparse
import os

import numpy as np
import torch
import trimesh
from PIL import Image
import pyrender
import smplx

DEFAULT_MODEL_FOLDER = os.environ.get("MOTIONSPEECH_SMPLX_MODELS")


def make_checker_texture(cell=64, n=8, c1=(243, 243, 243), c2=(214, 214, 214)):
    img = np.zeros((cell * n, cell * n, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = c1 if (i + j) % 2 else c2
    return Image.fromarray(img)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float64), (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=30)
    ap.add_argument('--max-poses', type=int, default=13)
    ap.add_argument('--spacing', type=float, default=0.78)
    ap.add_argument('--width', type=int, default=2000)
    ap.add_argument('--height', type=int, default=600)
    ap.add_argument('--yfov', type=float, default=0.42)
    ap.add_argument('--body-color', default='6fa8dc')
    ap.add_argument('--smplx-model-dir', default=DEFAULT_MODEL_FOLDER,
                    help='Directory containing the licensed SMPL-X model files.')
    ap.add_argument('--hands', default='relaxed', choices=['relaxed', 'flat', 'mocap'],
                    help='Hand pose source: model mean, flat, or the input motion capture.')
    args = ap.parse_args()
    if not args.smplx_model_dir:
        ap.error('--smplx-model-dir or MOTIONSPEECH_SMPLX_MODELS is required')
    model_folder = args.smplx_model_dir

    data = np.load(args.npz)
    poses = data['poses'].astype(np.float32)
    betas = data['betas'].astype(np.float32)
    T = poses.shape[0]
    idxs = list(range(0, T, args.stride))[:args.max_poses]
    N = len(idxs)
    print(f'frames={T} -> {N} snapshots (stride {args.stride})')

    model = smplx.create(model_folder, model_type='smplx', gender='NEUTRAL_2020',
                         num_betas=300, num_expression_coeffs=100, use_pca=False,
                         flat_hand_mean=True, ext='npz')
    p = torch.from_numpy(poses[idxs])
    if args.hands == 'mocap':
        hand_l, hand_r = p[:, 66:111], p[:, 111:156]
    elif args.hands == 'relaxed':
        hm = np.load(f'{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz')
        hand_l = torch.from_numpy(hm['hands_meanl'].astype(np.float32)).unsqueeze(0).repeat(N, 1)
        hand_r = torch.from_numpy(hm['hands_meanr'].astype(np.float32)).unsqueeze(0).repeat(N, 1)
    else:  # flat
        hand_l = torch.zeros(N, 45)
        hand_r = torch.zeros(N, 45)
    out = model(
        betas=torch.from_numpy(betas[:300]).float().unsqueeze(0).repeat(N, 1),
        expression=torch.zeros(N, 100),
        global_orient=p[:, :3], body_pose=p[:, 3:66],
        left_hand_pose=hand_l, right_hand_pose=hand_r,
        jaw_pose=torch.zeros(N, 3), leye_pose=torch.zeros(N, 3), reye_pose=torch.zeros(N, 3),
        return_verts=True)
    verts = out.vertices.detach().numpy()
    faces = model.faces

    # center each body on its own origin, feet on y=0, then lay out along X
    verts[:, :, 0] -= verts[:, :, 0].mean(axis=1, keepdims=True)
    verts[:, :, 2] -= verts[:, :, 2].mean(axis=1, keepdims=True)
    verts[:, :, 1] -= verts[:, :, 1].min()
    for i in range(N):
        verts[i, :, 0] += i * args.spacing

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.38, 0.38, 0.38])
    color = tuple(int(args.body_color[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)
    body_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.55)
    for i in range(N):
        tm = trimesh.Trimesh(verts[i], faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(tm, material=body_mat, smooth=True))

    # checkered ground
    span = (N - 1) * args.spacing
    plane_w, plane_d = span + 3.0, 7.0
    checker = make_checker_texture()
    gv = np.array([[-plane_w / 2, 0, -plane_d / 2], [plane_w / 2, 0, -plane_d / 2],
                   [plane_w / 2, 0, plane_d / 2], [-plane_w / 2, 0, plane_d / 2]], dtype=np.float32)
    gv[:, 0] += span / 2
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    gtm = trimesh.Trimesh(gv, [[0, 2, 1], [0, 3, 2]], process=False)  # normal +Y
    gtm.visual = trimesh.visual.TextureVisuals(uv=uv, image=checker)
    scene.add(pyrender.Mesh.from_trimesh(gtm, smooth=False))

    # lights
    for pos, inten in [([span / 2 - 2, 4.0, 3.0], 3.2), ([span / 2 + 2.5, 3.5, 2.0], 2.2)]:
        light = pyrender.DirectionalLight(intensity=inten)
        scene.add(light, pose=look_at(pos, [span / 2, 0.8, 0.0]))

    # camera: long lens, slightly above waist, straight-on (flat perspective like the reference)
    import math
    aspect = args.width / args.height
    hfov = 2 * math.atan(math.tan(args.yfov / 2) * aspect)
    dist = (span / 2 + 0.55) / math.tan(hfov / 2)
    cam = pyrender.PerspectiveCamera(yfov=args.yfov)
    scene.add(cam, pose=look_at([span / 2, 1.30, dist], [span / 2, 0.92, 0.0]))

    r = pyrender.OffscreenRenderer(args.width, args.height)
    img, _ = r.render(scene)
    Image.fromarray(img).save(args.out)
    print(f'saved {args.out} ({N} poses)')


if __name__ == '__main__':
    main()
