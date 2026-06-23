#!/usr/bin/env python3
import argparse
import ast
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SH_C0 = 0.28209479177387814


def read_gaussian_ply_xyz_rgb(path):
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError('bad ply header')
            text = line.decode('ascii')
            header.append(text)
            if text.strip() == 'end_header':
                break
        props = []
        n = None
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == 'element' and parts[1] == 'vertex':
                n = int(parts[2])
            elif len(parts) >= 3 and parts[0] == 'property':
                props.append(parts[-1])
        dtype = np.dtype([(p, '<f4') for p in props])
        arr = np.fromfile(f, dtype=dtype, count=n)
    xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
    rgb = np.column_stack([
        0.5 + SH_C0 * arr['f_dc_0'],
        0.5 + SH_C0 * arr['f_dc_1'],
        0.5 + SH_C0 * arr['f_dc_2'],
    ])
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return xyz, rgb


def load_placements(info_path):
    text = Path(info_path).read_text(encoding='utf-8')
    for line in text.splitlines():
        if line.startswith('placements:'):
            return ast.literal_eval(line.split(':', 1)[1].strip())
    raise RuntimeError('placements not found')


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def render_points(points, colors, eye, target, width, height, fov_deg, bg=(18, 20, 18)):
    forward = normalize(target - eye)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    rel = points - eye[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    near = 0.05
    mask = z > near
    if not np.any(mask):
        return np.full((height, width, 3), bg, dtype=np.uint8)
    x, y, z = x[mask], y[mask], z[mask]
    c = colors[mask]

    focal = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    u = (width * 0.5 + focal * x / z).astype(np.int32)
    v = (height * 0.5 - focal * y / z).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return np.full((height, width, 3), bg, dtype=np.uint8)
    u, v, z, c = u[inside], v[inside], z[inside], c[inside]

    order = np.argsort(z)[::-1]
    u, v, c = u[order], v[order], c[order]
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    pix = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    # A small 3x3 splat makes Gaussian/mesh samples readable in video.
    for du, dv in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
        uu = u + du
        vv = v + dv
        ok = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        img[vv[ok], uu[ok]] = pix[ok]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', default='output/garden_fused/point_cloud/iteration_30000/point_cloud.ply')
    ap.add_argument('--info', default='output/garden_fused/fusion_info.txt')
    ap.add_argument('--out-dir', default='output/garden_fused/walkthrough')
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--scene-samples', type=int, default=650000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    points, colors = read_gaussian_ply_xyz_rgb(args.ply)
    placements = load_placements(args.info)
    scene_n = 3623040
    rng = np.random.default_rng(20260601)

    scene_idx = rng.choice(scene_n, size=min(args.scene_samples, scene_n), replace=False)
    object_idx = np.arange(scene_n, len(points), dtype=np.int64)
    keep = np.concatenate([scene_idx, object_idx])
    points = points[keep]
    colors = colors[keep]

    centers = np.array(list(placements.values()), dtype=np.float32)
    target = centers.mean(axis=0) + np.array([0.0, 1.0, 0.0], dtype=np.float32)
    span_xz = max(float(centers[:, 0].max() - centers[:, 0].min()), float(centers[:, 2].max() - centers[:, 2].min()))
    radius = max(10.0, span_xz * 1.45)
    y_base = float(target[1] + 3.6)

    video_path = out_dir / 'task1_scene_fusion_walkthrough.mp4'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError('failed to open VideoWriter')

    for i in range(args.frames):
        t = i / args.frames
        angle = 2.0 * math.pi * t
        eye = target + np.array([
            radius * math.cos(angle),
            1.2 * math.sin(2.0 * angle) + (y_base - target[1]),
            radius * math.sin(angle),
        ], dtype=np.float32)
        frame = render_points(points, colors, eye, target, args.width, args.height, fov_deg=55.0)
        # Write unobtrusive labels as tiny colored markers in the first few frames only.
        cv2.imwrite(str(frame_dir / f'{i:05d}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i in (0, args.frames // 4, args.frames // 2, 3 * args.frames // 4):
            Image.fromarray(frame).save(out_dir / f'preview_{i:03d}.png')
    writer.release()

    # Build a contact sheet from representative views.
    previews = [Image.open(out_dir / f'preview_{i:03d}.png').convert('RGB') for i in (0, args.frames // 4, args.frames // 2, 3 * args.frames // 4)]
    sheet = Image.new('RGB', (args.width * 2, args.height * 2), (18, 20, 18))
    for i, im in enumerate(previews):
        sheet.paste(im, ((i % 2) * args.width, (i // 2) * args.height))
    sheet.save(out_dir / 'task1_scene_fusion_preview.png')

    print(f'points_rendered: {len(points)}')
    print(f'video: {video_path}')
    print(f'preview: {out_dir / "task1_scene_fusion_preview.png"}')


if __name__ == '__main__':
    main()
