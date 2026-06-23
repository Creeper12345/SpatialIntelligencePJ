#!/usr/bin/env python3
import argparse
import ast
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2] / 'gaussian-splatting'
sys.path.insert(0, str(REPO))

from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = True


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def load_placements(info_path):
    text = Path(info_path).read_text(encoding='utf-8')
    for line in text.splitlines():
        if line.startswith('placements:'):
            return ast.literal_eval(line.split(':', 1)[1].strip())
    raise RuntimeError('placements not found')


def make_camera(eye, target, width, height, fovy_deg):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = normalize(target - eye)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(world_up, forward))
    up = normalize(np.cross(forward, right))
    c2w_rot = np.stack([right, up, forward], axis=1).astype(np.float32)

    # 3DGS Camera stores R so that R.T is world-to-camera rotation.
    R = c2w_rot
    T = -R.T @ eye
    fovy = math.radians(fovy_deg)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * width / height)
    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device='cuda').transpose(0, 1)
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, 0.01, 100.0, world_view, full)


def tensor_to_rgb_uint8(t):
    arr = t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', default='../output/garden_fused_table/point_cloud/iteration_30000/point_cloud.ply')
    ap.add_argument('--info', default='../output/garden_fused_table/fusion_info.txt')
    ap.add_argument('--out-dir', default='../output/garden_fused_table/official_3dgs_walkthrough')
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--fovy', type=float, default=38.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    gaussians = GaussianModel(3)
    gaussians.load_ply(args.ply)
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    pipe = Pipe()

    placements = load_placements(args.info)
    centers = np.array(list(placements.values()), dtype=np.float32)
    target = centers.mean(axis=0) + np.array([0.0, 0.42, 0.02], dtype=np.float32)
    radius = 4.0
    eye_y = target[1] + 1.05

    video_path = out_dir / 'task1_scene_fusion_table_3dgs.mp4'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError('failed to open video writer')

    preview_indices = {0, args.frames // 4, args.frames // 2, 3 * args.frames // 4}
    previews = []
    for i in range(args.frames):
        t = i / args.frames
        angle = 2.0 * math.pi * t
        eye = target + np.array([
            radius * math.cos(angle),
            0.35 * math.sin(2.0 * angle) + (eye_y - target[1]),
            radius * math.sin(angle),
        ], dtype=np.float32)
        cam = make_camera(eye, target, args.width, args.height, args.fovy)
        img = tensor_to_rgb_uint8(render(cam, gaussians, pipe, bg)['render'])
        Image.fromarray(img).save(frame_dir / f'{i:05d}.png')
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i in preview_indices:
            p = out_dir / f'preview_{i:03d}.png'
            Image.fromarray(img).save(p)
            previews.append(Image.fromarray(img))
        print(f'rendered {i + 1}/{args.frames}', flush=True)
    writer.release()

    if previews:
        sheet = Image.new('RGB', (args.width * 2, args.height * 2), (0, 0, 0))
        for i, im in enumerate(previews[:4]):
            sheet.paste(im, ((i % 2) * args.width, (i // 2) * args.height))
        sheet.save(out_dir / 'task1_scene_fusion_table_3dgs_preview.png')

    print(f'video: {video_path}')
    print(f'preview: {out_dir / "task1_scene_fusion_table_3dgs_preview.png"}')


if __name__ == '__main__':
    main()
