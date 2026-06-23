"""
Scene fusion: merge Objects A (3DGS), B (mesh), C (mesh) into garden 3DGS scene.

Coordinate conventions:
  Garden (COLMAP / MipNeRF360): Y-DOWN  (positive Y = physically lower)
  Threestudio meshes (B, C):    Y-UP    (positive Y = physically higher)
  Object A (phone COLMAP):      Y-DOWN  (same as garden)

Placement strategy (fixes the "floating" problem):
  The garden tabletop is NOT a single flat Y plane in COLMAP space – it is
  tilted, so the surface height varies with (X, Z).  Instead of using one
  global TABLE_Y, we probe the actual garden Gaussians directly beneath each
  object's footprint and seat the object on that *local* surface height.
  Footprints are also constrained to lie on the real tabletop disk so nothing
  hangs off the edge.

Colour strategy (fixes the "pure orange" / muddy colours):
  - Object B (basketball) stores per-vertex colours in the OBJ `v x y z r g b`
    rows; we area-sample faces and barycentrically interpolate those colours.
  - Object C stores UVs + a texture image (texture_kd.jpg); we sample the
    texture at each point's interpolated UV so the real texture detail shows.
"""
import numpy as np
from plyfile import PlyData, PlyElement
import os, json, math, shutil

BASE       = "/remote-home/qyliu/class/final-pj"
GARDEN_PLY = f"{BASE}/output/garden/point_cloud/iteration_30000/point_cloud.ply"
OBJ_A_PLY  = f"{BASE}/task1/gs_output_masked/point_cloud/iteration_10000/point_cloud.ply"
OBJ_B_OBJ  = (f"{BASE}/task1/threestudio_outputs/task1_text_b/"
               "dreamfusion_sd_basketball_6000@20260531-151908/save/it6000-export/basketball_mesh.obj")
OBJ_C_OBJ  = (f"{BASE}/task1/threestudio_outputs/task1_zero123/"
               "stable_final@20260601-111142/save/it3701-export/model.obj")
OUT_DIR    = f"{BASE}/fused_scene"
os.makedirs(f"{OUT_DIR}/point_cloud/iteration_99999", exist_ok=True)

SH_C0 = 0.28209479177387814

PLY_PROPS = (['x','y','z','nx','ny','nz',
              'f_dc_0','f_dc_1','f_dc_2'] +
             [f'f_rest_{i}' for i in range(45)] +
             ['opacity','scale_0','scale_1','scale_2',
              'rot_0','rot_1','rot_2','rot_3'])

# ── PLY helpers ──────────────────────────────────────────────────────────────
def load_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    return {p.name: np.array(v[p.name], dtype=np.float32) for p in v.properties}

def save_ply(data, path):
    N = len(data['x'])
    arr = np.zeros(N, dtype=[(p, np.float32) for p in PLY_PROPS])
    for p in PLY_PROPS:
        if p in data:
            arr[p] = data[p]
    PlyData([PlyElement.describe(arr, 'vertex')]).write(path)
    print(f"  Saved {N:,} Gaussians -> {path}")

def rotation_y(deg):
    r = math.radians(deg)
    return np.array([[math.cos(r), 0, math.sin(r)],
                     [0,           1, 0           ],
                     [-math.sin(r),0, math.cos(r)]])

def rotation_x(deg):
    r = math.radians(deg)
    return np.array([[1, 0,            0           ],
                     [0, math.cos(r), -math.sin(r) ],
                     [0, math.sin(r),  math.cos(r) ]])

# Mesh -> garden up-axis correction.
# The threestudio meshes use Z as their canonical up axis (basketball's baked
# bottom shadow sits on -Z, the coffee-jar lid sits on +Z).  Rotating +90 deg
# about X maps mesh +Z -> garden -Y (up) and mesh -Z -> garden +Y (down/table),
# so the jar stands upright (lid up) and the basketball's dark side rests on
# the table.  Garden is Y-DOWN, so +Y is physically lower.
R_MESH_UP = rotation_x(90)

def merge(*dicts):
    out = {}
    for k in PLY_PROPS:
        arrays = [d[k] for d in dicts if k in d]
        if arrays:
            out[k] = np.concatenate(arrays).astype(np.float32)
    return out

# ── OBJ loading (vertex colours + UV/texture) ────────────────────────────────
def parse_obj(path):
    """Return verts(N,3), vcol(N,3) or None, uv(M,2) or None,
       faces_v(F,3), faces_uv(F,3) or None."""
    verts, vcol, uvs, fv, fuv = [], [], [], [], []
    has_vcol = False
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
                if len(p) >= 7:
                    vcol.append([float(p[4]), float(p[5]), float(p[6])])
                    has_vcol = True
                else:
                    vcol.append([0.6, 0.6, 0.6])
            elif line.startswith('vt '):
                p = line.split()
                uvs.append([float(p[1]), float(p[2])])
            elif line.startswith('f '):
                vi, ti = [], []
                for tok in line.split()[1:]:
                    s = tok.split('/')
                    vi.append(int(s[0]) - 1)
                    if len(s) >= 2 and s[1] != '':
                        ti.append(int(s[1]) - 1)
                for k in range(1, len(vi) - 1):
                    fv.append([vi[0], vi[k], vi[k + 1]])
                    if len(ti) == len(vi):
                        fuv.append([ti[0], ti[k], ti[k + 1]])
    verts = np.asarray(verts, np.float32)
    vcol = np.asarray(vcol, np.float32) if has_vcol else None
    uvs = np.asarray(uvs, np.float32) if uvs else None
    fv = np.asarray(fv, np.int64)
    fuv = np.asarray(fuv, np.int64) if fuv else None
    return verts, vcol, uvs, fv, fuv

def sample_surface(verts, fv, n, seed=0):
    rng = np.random.default_rng(seed)
    tri = verts[fv]                                    # F,3,3
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = np.maximum(area, 1e-12)
    fidx = rng.choice(len(fv), size=n, replace=True, p=area / area.sum())
    r1 = np.sqrt(rng.random(n, dtype=np.float32))
    r2 = rng.random(n, dtype=np.float32)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)   # n,3 barycentric
    pts = (verts[fv[fidx]] * w[:, :, None]).sum(axis=1)
    return pts.astype(np.float32), fidx, w

def mesh_to_gaussians(obj_path, n_samples=120000, gs_radius=0.0045, tex_path=None):
    print(f"  Loading: {os.path.basename(obj_path)}")
    verts, vcol, uvs, fv, fuv = parse_obj(obj_path)
    pts, fidx, w = sample_surface(verts, fv, n_samples)
    N = len(pts)

    if tex_path and uvs is not None and fuv is not None:
        from PIL import Image
        tex = np.asarray(Image.open(tex_path).convert('RGB'), np.float32) / 255.0
        H, W = tex.shape[:2]
        uv = (uvs[fuv[fidx]] * w[:, :, None]).sum(axis=1)      # N,2
        u = np.clip(uv[:, 0], 0, 1)
        v = np.clip(1.0 - uv[:, 1], 0, 1)                      # flip V for image
        px = np.clip((u * (W - 1)).astype(int), 0, W - 1)
        py = np.clip((v * (H - 1)).astype(int), 0, H - 1)
        colors = tex[py, px]
        print(f"    sampled texture {os.path.basename(tex_path)} ({W}x{H})")
    elif vcol is not None:
        colors = (vcol[fv[fidx]] * w[:, :, None]).sum(axis=1)
        print(f"    sampled per-vertex colours  mean={colors.mean(0).round(3)}")
    else:
        colors = np.full((N, 3), 0.6, np.float32)

    colors = np.clip(colors, 0.0, 1.0)
    f_dc = (colors - 0.5) / SH_C0
    data = {
        'x': pts[:, 0], 'y': pts[:, 1], 'z': pts[:, 2],
        'nx': np.zeros(N, np.float32), 'ny': np.zeros(N, np.float32),
        'nz': np.zeros(N, np.float32),
        'f_dc_0': f_dc[:, 0].astype(np.float32),
        'f_dc_1': f_dc[:, 1].astype(np.float32),
        'f_dc_2': f_dc[:, 2].astype(np.float32),
        'opacity': np.full(N, 6.0, np.float32),
        'rot_0': np.ones(N, np.float32),
        'rot_1': np.zeros(N, np.float32),
        'rot_2': np.zeros(N, np.float32),
        'rot_3': np.zeros(N, np.float32),
    }
    for i in range(45):
        data[f'f_rest_{i}'] = np.zeros(N, np.float32)
    log_r = math.log(gs_radius)
    for i in range(3):
        data[f'scale_{i}'] = np.full(N, log_r, np.float32)
    return data

# ── load background + table-surface probe ────────────────────────────────────
print("Loading garden scene ...")
garden = load_ply(GARDEN_PLY)
GX, GY, GZ = garden['x'], garden['y'], garden['z']
print(f"  {len(GX):,} Gaussians")

def table_surface_y(cx, cz, rad=0.13):
    """Local tabletop height (garden Y-down) directly under footprint (cx,cz).
    Returns (surface_y, support_count). Smaller Y == physically higher."""
    m = ((np.abs(GX - cx) < rad) & (np.abs(GZ - cz) < rad) &
         (GY > 1.3) & (GY < 2.7))
    cnt = int(m.sum())
    if cnt < 40:
        return None, cnt
    ys = GY[m]
    top = np.percentile(ys, 8)                 # near the highest (top) surface
    surf = ys[ys < top + 0.12]                 # points on the top surface layer
    return float(np.median(surf)), cnt

def place_on_table(data, scale, R, cx, cz, embed=0.02):
    """Center -> scale -> rotate by R (full mesh->garden rotation) -> seat the
    lowest point on the local table surface at (cx, cz). `embed` pushes the
    object slightly into the surface (+Y in Y-down) so it never floats."""
    out = {k: v.copy() for k, v in data.items()}
    xyz = np.stack([out['x'], out['y'], out['z']], axis=1)
    xyz -= xyz.mean(axis=0)
    xyz *= scale
    for i in range(3):
        out[f'scale_{i}'] = out[f'scale_{i}'] + math.log(scale)
    xyz = xyz @ R.T

    surf, cnt = table_surface_y(cx, cz)
    if surf is None:
        raise RuntimeError(f"No table surface under footprint ({cx},{cz}); cnt={cnt}")
    floor_y = xyz[:, 1].max()                  # physically lowest point
    xyz[:, 0] += cx
    xyz[:, 1] += (surf + embed) - floor_y
    xyz[:, 2] += cz
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    print(f"    footprint=({cx:+.2f},{cz:+.2f}) surface_Y={surf:.3f} support={cnt}")
    return out

# ── Object A: green bird (3DGS, COLMAP Y-down, no flip) ───────────────────────
print("\nObject A - green bird (3DGS) ...")
obj_a = load_ply(OBJ_A_PLY)
xyz_a = np.stack([obj_a['x'], obj_a['y'], obj_a['z']], axis=1)
a_height = xyz_a[:, 1].max() - xyz_a[:, 1].min()
TARGET_A = 0.52
scale_a = TARGET_A / a_height
# Bird is already a COLMAP Y-down 3DGS like the garden -> only azimuth spin.
obj_a_t = place_on_table(obj_a, scale_a, rotation_y(25), cx=-0.35, cz=0.12)
print(f"  scale={scale_a:.4f}")

# ── Object B: basketball (mesh; mesh-Z up, dark side -Z -> table) ─────────────
print("\nObject B - basketball (mesh) ...")
obj_b_raw = mesh_to_gaussians(OBJ_B_OBJ, n_samples=140000, gs_radius=0.0045)
xyz_b = np.stack([obj_b_raw['x'], obj_b_raw['y'], obj_b_raw['z']], axis=1)
xyz_b -= xyz_b.mean(axis=0)
b_diam = ((xyz_b.max(0) - xyz_b.min(0))).mean()
TARGET_B = 0.42
scale_b = TARGET_B / b_diam
R_b = rotation_y(0) @ R_MESH_UP          # dark -Z side rests on the table
obj_b_t = place_on_table(obj_b_raw, scale_b, R_b, cx=0.32, cz=0.10)
print(f"  scale={scale_b:.4f}")

# ── Object C: coffee jar (mesh + texture; mesh-Z up, lid +Z -> up) ────────────
print("\nObject C - coffee jar (mesh) ...")
C_TEX = os.path.join(os.path.dirname(OBJ_C_OBJ), "texture_kd.jpg")
obj_c_raw = mesh_to_gaussians(OBJ_C_OBJ, n_samples=140000, gs_radius=0.0045,
                              tex_path=C_TEX if os.path.exists(C_TEX) else None)
xyz_c = np.stack([obj_c_raw['x'], obj_c_raw['y'], obj_c_raw['z']], axis=1)
xyz_c -= xyz_c.mean(axis=0)
c_height = xyz_c[:, 2].max() - xyz_c[:, 2].min()   # jar height is along mesh-Z
TARGET_C = 0.42
scale_c = TARGET_C / c_height
R_c = rotation_y(140) @ R_MESH_UP        # stand upright, lid up; AGF label faces frame 78 view
obj_c_t = place_on_table(obj_c_raw, scale_c, R_c, cx=-0.02, cz=0.45)
print(f"  scale={scale_c:.4f}")

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\nMerging ...")
fused = merge(garden, obj_a_t, obj_b_t, obj_c_t)
print(f"  Total: {len(fused['x']):,} Gaussians")

out_ply = f"{OUT_DIR}/point_cloud/iteration_99999/point_cloud.ply"
save_ply(fused, out_ply)

cfg = ("Namespace(data_device='cpu', depths='', eval=False, images='images_4', "
       f"model_path='{OUT_DIR}', resolution=4, sh_degree=3, "
       f"source_path='{BASE}/data/nerf360/garden', "
       "train_test_exp=False, white_background=False)")
with open(f"{OUT_DIR}/cfg_args", 'w') as f:
    f.write(cfg)
shutil.copy(f"{BASE}/output/garden/cameras.json", f"{OUT_DIR}/cameras.json")
print(f"\nDone -> {OUT_DIR}")
