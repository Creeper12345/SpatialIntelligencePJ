#!/usr/bin/env python3
import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


def read_gaussian_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading header: {path}")
            text = line.decode("ascii")
            header.append(text)
            if text.strip() == "end_header":
                break

        props = []
        count = None
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                count = int(parts[2])
            elif len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
        if count is None:
            raise ValueError(f"No vertex count in {path}")

        dtype = np.dtype([(p, "<f4") for p in props])
        data = np.fromfile(f, dtype=dtype, count=count)
    return props, data


def write_gaussian_ply(path, props, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(data)}\n".encode("ascii"))
        for prop in props:
            f.write(f"property float {prop}\n".encode("ascii"))
        f.write(b"end_header\n")
        data.astype(data.dtype.newbyteorder("<"), copy=False).tofile(f)


def gaussian_rgb_to_dc(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb - 0.5) / SH_C0


def transform_gaussians(data, target_height, target_center, up_axis=1):
    out = data.copy()
    xyz = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    lo = np.quantile(xyz, 0.01, axis=0)
    hi = np.quantile(xyz, 0.99, axis=0)
    center = (lo + hi) * 0.5
    height = max(float(hi[up_axis] - lo[up_axis]), 1e-6)
    scale = target_height / height
    xyz = (xyz - center) * scale
    lo2 = np.quantile(xyz, 0.01, axis=0)
    xyz[:, up_axis] -= lo2[up_axis]
    xyz += np.asarray(target_center, dtype=np.float32)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for key in ("scale_0", "scale_1", "scale_2"):
        if key in out.dtype.names:
            out[key] += math.log(scale)
    return out


def parse_obj(path):
    verts = []
    colors = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vals = [float(x) for x in parts[1:]]
                verts.append(vals[:3])
                if len(vals) >= 6:
                    colors.append(vals[3:6])
                else:
                    colors.append([0.7, 0.7, 0.7])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    idx.append(int(tok.split("/")[0]) - 1)
                if len(idx) >= 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=np.float32), np.asarray(colors, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def sample_mesh_points(obj_path, count, seed=0):
    verts, colors, faces = parse_obj(obj_path)
    rng = np.random.default_rng(seed)
    if len(faces) == 0:
        idx = rng.choice(len(verts), size=min(count, len(verts)), replace=len(verts) < count)
        return verts[idx], colors[idx]

    tri = verts[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    area = np.maximum(area, 1e-12)
    prob = area / area.sum()
    tri_idx = rng.choice(len(faces), size=count, replace=True, p=prob)
    chosen = faces[tri_idx]
    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    pts = verts[chosen[:, 0]] * w0[:, None] + verts[chosen[:, 1]] * w1[:, None] + verts[chosen[:, 2]] * w2[:, None]
    cols = colors[chosen[:, 0]] * w0[:, None] + colors[chosen[:, 1]] * w1[:, None] + colors[chosen[:, 2]] * w2[:, None]

    # Keep original vertices too; they preserve sharp silhouettes and texture/color extrema.
    if len(verts) < count // 2:
        pts = np.concatenate([pts, verts], axis=0)
        cols = np.concatenate([cols, colors], axis=0)
    return pts.astype(np.float32), np.clip(cols.astype(np.float32), 0.0, 1.0)


def points_to_gaussians(points, colors, props, dtype, target_height, target_center, splat_radius, opacity=0.94, up_axis=1):
    pts = points.astype(np.float32)
    lo = np.quantile(pts, 0.01, axis=0)
    hi = np.quantile(pts, 0.99, axis=0)
    center = (lo + hi) * 0.5
    height = max(float(hi[up_axis] - lo[up_axis]), 1e-6)
    scale = target_height / height
    pts = (pts - center) * scale
    lo2 = np.quantile(pts, 0.01, axis=0)
    pts[:, up_axis] -= lo2[up_axis]
    pts += np.asarray(target_center, dtype=np.float32)

    arr = np.zeros(len(pts), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if "opacity" in props:
        arr["opacity"] = math.log(opacity / (1.0 - opacity))
    for key in ("scale_0", "scale_1", "scale_2"):
        if key in props:
            arr[key] = math.log(splat_radius)
    if "rot_0" in props:
        arr["rot_0"] = 1.0
    dc = gaussian_rgb_to_dc(colors)
    for i, key in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        if key in props:
            arr[key] = dc[:, i]
    return arr


def copy_scene_metadata(scene_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "exposure.json", "input.ply"):
        src = Path(scene_dir) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    cfg_args = out_dir / "cfg_args"
    if cfg_args.exists():
        text = cfg_args.read_text(encoding="utf-8")
        text = text.replace(str(Path(scene_dir).resolve()), str(out_dir.resolve()))
        cfg_args.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="output/garden")
    parser.add_argument("--out-dir", default="output/garden_fused")
    parser.add_argument("--scene-ply", default="output/garden/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--object-a-ply", default="task1/gs_output/point_cloud/iteration_3000/point_cloud.ply")
    parser.add_argument("--object-b-obj", default="task1/threestudio_outputs/task1_text_b/dreamfusion_sd_basketball_6000@20260531-151908/save/it6000-export/basketball_mesh.obj")
    parser.add_argument("--object-c-obj", default="task1/threestudio_outputs/task1_zero123/stable_lowmem@20260531-123503/save/it600-export/zero123_object_mesh.obj")
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    props, scene = read_gaussian_ply(args.scene_ply)
    _, object_a = read_gaussian_ply(args.object_a_ply)

    scene_xyz = np.column_stack([scene["x"], scene["y"], scene["z"]])
    q05 = np.quantile(scene_xyz, 0.05, axis=0)
    q50 = np.quantile(scene_xyz, 0.50, axis=0)
    q95 = np.quantile(scene_xyz, 0.95, axis=0)
    span = np.maximum(q95 - q05, 1e-4)

    ground_y = float(np.quantile(scene_xyz[:, 1], 0.08))
    base_z = float(q50[2] - 0.10 * span[2])
    placements = {
        "A_video_3dgs": [float(q50[0] - 0.18 * span[0]), ground_y, base_z],
        "B_basketball": [float(q50[0]), ground_y, float(base_z + 0.08 * span[2])],
        "C_zero123": [float(q50[0] + 0.18 * span[0]), ground_y, base_z],
    }

    # Scale objects to sit visibly in the garden while remaining plausible tabletop/garden props.
    obj_a = transform_gaussians(object_a, target_height=float(0.12 * span[1]), target_center=placements["A_video_3dgs"])

    pts_b, col_b = sample_mesh_points(args.object_b_obj, 140000, seed=11)
    pts_c, col_c = sample_mesh_points(args.object_c_obj, 90000, seed=17)
    obj_b = points_to_gaussians(pts_b, col_b, props, scene.dtype, float(0.10 * span[1]), placements["B_basketball"], splat_radius=float(0.0035 * max(span[0], span[2])))
    obj_c = points_to_gaussians(pts_c, col_c, props, scene.dtype, float(0.11 * span[1]), placements["C_zero123"], splat_radius=float(0.0035 * max(span[0], span[2])))

    fused = np.concatenate([scene, obj_a.astype(scene.dtype, copy=False), obj_b, obj_c])
    out_dir = Path(args.out_dir)
    copy_scene_metadata(args.scene_dir, out_dir)
    out_ply = out_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    write_gaussian_ply(out_ply, props, fused)

    info = out_dir / "fusion_info.txt"
    info.write_text(
        "\n".join(
            [
                "Task1 part 3 fused scene",
                f"scene_gaussians: {len(scene)}",
                f"object_A_gaussians: {len(obj_a)}",
                f"object_B_gaussians: {len(obj_b)}",
                f"object_C_gaussians: {len(obj_c)}",
                f"total_gaussians: {len(fused)}",
                f"scene_q05: {q05.tolist()}",
                f"scene_q50: {q50.tolist()}",
                f"scene_q95: {q95.tolist()}",
                f"placements: {placements}",
                f"output_ply: {out_ply}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(info.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
