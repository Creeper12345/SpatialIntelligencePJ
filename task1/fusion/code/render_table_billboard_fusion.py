#!/usr/bin/env python3
import ast
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 960, 540
FRAMES = 96
FPS = 24


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def load_placements(info_path):
    text = Path(info_path).read_text(encoding='utf-8')
    for line in text.splitlines():
        if line.startswith('placements:'):
            return ast.literal_eval(line.split(':', 1)[1].strip())
    raise RuntimeError('placements not found')


def camera_pose(i, centers):
    target = centers.mean(axis=0) + np.array([0.0, 0.42, 0.02], dtype=np.float32)
    radius = 4.0
    eye_y = target[1] + 1.05
    angle = 2.0 * math.pi * (i / FRAMES)
    eye = target + np.array([
        radius * math.cos(angle),
        0.35 * math.sin(2.0 * angle) + (eye_y - target[1]),
        radius * math.sin(angle),
    ], dtype=np.float32)
    return eye, target


def project(point, eye, target, fovy_deg=38.0):
    point = np.asarray(point, dtype=np.float32)
    fwd = normalize(target - eye)
    upw = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(fwd, upw))
    up = normalize(np.cross(right, fwd))
    rel = point - eye
    x, y, z = float(rel @ right), float(rel @ up), float(rel @ fwd)
    focal = 0.5 * H / math.tan(math.radians(fovy_deg) * 0.5)
    u = W * 0.5 + focal * x / z
    v = H * 0.5 - focal * y / z
    return u, v, z


def crop_alpha_bbox(im, pad=6):
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    a = np.asarray(im.getchannel('A'))
    ys, xs = np.where(a > 12)
    if len(xs) == 0:
        return im
    x0, x1 = max(0, xs.min() - pad), min(im.width, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(im.height, ys.max() + pad + 1)
    return im.crop((x0, y0, x1, y1))


def make_basketball_sprite(size=512):
    im = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = int(size * 0.07)
    bbox = (pad, pad, size - pad, size - pad)
    d.ellipse(bbox, fill=(235, 104, 22, 255), outline=(95, 38, 8, 255), width=max(5, size // 40))
    # Grooves drawn as arcs and curves.
    w = max(5, size // 38)
    d.arc((pad, size * 0.18, size - pad, size * 0.82), 90, 270, fill=(30, 22, 18, 255), width=w)
    d.arc((pad, size * 0.18, size - pad, size * 0.82), -90, 90, fill=(30, 22, 18, 255), width=w)
    d.line((size * 0.5, pad, size * 0.5, size - pad), fill=(30, 22, 18, 255), width=w)
    d.arc((size * 0.18, pad, size * 0.82, size - pad), 0, 180, fill=(30, 22, 18, 255), width=w)
    d.arc((size * 0.18, pad, size * 0.82, size - pad), 180, 360, fill=(30, 22, 18, 255), width=w)
    # Soft highlight.
    hi = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hi)
    hd.ellipse((size * 0.22, size * 0.15, size * 0.52, size * 0.34), fill=(255, 173, 82, 80))
    return Image.alpha_composite(im, hi.filter(ImageFilter.GaussianBlur(size * 0.025)))


def make_a_sprite(video_path='data/mov.mp4'):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('failed to read object A video')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # Use a central crop; the object-turntable video is centered.
    x0, x1 = int(w * 0.18), int(w * 0.82)
    y0, y1 = int(h * 0.08), int(h * 0.92)
    crop = rgb[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), np.uint8)
    rect = (int(cw * 0.08), int(ch * 0.08), int(cw * 0.84), int(ch * 0.84))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    rgba = np.dstack([crop, alpha])
    im = crop_alpha_bbox(Image.fromarray(rgba, 'RGBA'))
    return im


def make_c_sprite(path='data/fig.png'):
    im = Image.open(path).convert('RGBA')
    return crop_alpha_bbox(im)


def add_shadow(base, center, width, height):
    shadow = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    x, y = center
    d.ellipse((x - width / 2, y - height / 2, x + width / 2, y + height / 2), fill=(0, 0, 0, 75))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(6, int(width * 0.05))))
    base.alpha_composite(shadow)


def paste_sprite(base, sprite, bottom_center, target_height, shadow=True):
    sprite = crop_alpha_bbox(sprite)
    scale = target_height / sprite.height
    nw = max(1, int(sprite.width * scale))
    nh = max(1, int(sprite.height * scale))
    spr = sprite.resize((nw, nh), Image.Resampling.LANCZOS)
    x = int(bottom_center[0] - nw / 2)
    y = int(bottom_center[1] - nh)
    if shadow:
        add_shadow(base, (bottom_center[0], bottom_center[1] + nh * 0.035), nw * 0.70, nh * 0.16)
    base.alpha_composite(spr, (x, y))


def main():
    out = Path('output/garden_fused_table/billboard_fusion')
    frames_dir = out / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    bg_dir = Path('output/garden_table_bg_3dgs/frames')
    placements = load_placements('output/garden_fused_table/fusion_info.txt')
    centers = np.array(list(placements.values()), dtype=np.float32)

    sprites = {
        'A_video_3dgs': make_a_sprite(),
        'B_basketball': make_basketball_sprite(),
        'C_zero123': make_c_sprite(),
    }
    for k, im in sprites.items():
        im.save(out / f'{k}_sprite.png')

    video_path = out / 'task1_scene_fusion_table_billboard.mp4'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    preview_frames = []

    for i in range(FRAMES):
        bg = Image.open(bg_dir / f'{i:05d}.png').convert('RGBA')
        eye, target = camera_pose(i, centers)
        draw_items = []
        for name, pos in placements.items():
            p = np.array(pos, dtype=np.float32)
            p[1] += 0.04
            u, v, z = project(p, eye, target)
            if z <= 0:
                continue
            # Approximate perspective scaling. Clamp so objects stay readable and tabletop-sized.
            if name == 'B_basketball':
                hpx = np.clip(430.0 / z, 92, 150)
            else:
                hpx = np.clip(520.0 / z, 120, 190)
            draw_items.append((z, name, (u, v + hpx * 0.50), hpx))
        # Draw far to near.
        for _, name, bottom_center, hpx in sorted(draw_items, key=lambda x: -x[0]):
            paste_sprite(bg, sprites[name], bottom_center, hpx)
        out_img = bg.convert('RGB')
        out_img.save(frames_dir / f'{i:05d}.png')
        writer.write(cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR))
        if i in (0, 24, 48, 72):
            p = out / f'preview_{i:03d}.png'
            out_img.save(p)
            preview_frames.append(out_img)
    writer.release()

    sheet = Image.new('RGB', (W * 2, H * 2), (0, 0, 0))
    for idx, im in enumerate(preview_frames):
        sheet.paste(im, ((idx % 2) * W, (idx // 2) * H))
    sheet.save(out / 'task1_scene_fusion_table_billboard_preview.png')
    print(f'video: {video_path}')
    print(f'preview: {out / "task1_scene_fusion_table_billboard_preview.png"}')


if __name__ == '__main__':
    main()
