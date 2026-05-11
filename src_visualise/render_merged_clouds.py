"""
Headless renderer for merged point-cloud comparison figures.

Renders a panel:
  [image0 | image1 | GT alignment | Method1 | Method2 | ...]

Usage:
  python render_merged_clouds.py \
    --point0  path/to/image0_points.npy \
    --point1  path/to/image1_points.npy \
    --img0    path/to/image0.jpg \
    --img1    path/to/image1.jpg \
    --cam2w0  path/to/image0.txt \
    --cam2w1  path/to/image1.txt \
    --methods "Ours+RANSAC:result_procrustes_ransac.json,MADPose:result_madpose.json,MADPose+rect:result_madpose_rect.json" \
    --results-dir /path/to/results/pair_name \
    --output  /path/to/output.png \
    [--distance-threshold 50] \
    [--point-size 2] \
    [--width 1920] [--height 640] \
    [--camera-json path/to/camera.json]
    [--dataset kitti|adt]
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import compute_pose_error


# 90° CW rotation matrix (ADT coordinate convention)
R_CW90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cloud(path):
    """Load UniK3D point cloud (1,3,H,W) -> (N,3)."""
    P = np.load(path)
    if P.ndim == 4 and P.shape[0] == 1 and P.shape[1] == 3:
        P = P[0].transpose(0, 2, 1).reshape(3, -1).T   # (H*W, 3)
    elif P.ndim == 2 and P.shape[1] == 3:
        pass  # already (N,3)
    else:
        raise ValueError(f"Unexpected point cloud shape {P.shape}")
    return P


def load_colors(img_path, H, W):
    """Return (H*W, 3) float RGB matching cloud ordering (transposed to match cloud)."""
    img = np.array(Image.open(img_path).convert("RGB")) / 255.0  # (H,W,3)
    # transpose to match the cloud's (W,H) -> (H*W) ordering used in load_cloud
    return img.transpose(2, 1, 0).reshape(3, -1).T  # (H*W, 3)


def load_gt_pose(cam2w0, cam2w1):
    T0 = np.loadtxt(cam2w0)
    T1 = np.loadtxt(cam2w1)
    rel = np.linalg.inv(T1) @ T0
    return rel[:3, :3], rel[:3, 3]


def load_method(json_path):
    """Load R_est, t_est, scale from a result JSON. Returns (R, t, scale) or None."""
    p = Path(json_path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        R = d.get("R_est")
        t = d.get("t_est")
        if R is None or t is None:
            return None
        scale = d.get("scale_est") or d.get("scale") or 1.0
        return np.array(R), np.array(t), float(scale)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Point cloud building
# ---------------------------------------------------------------------------

def make_pcd(P, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return pcd


def build_merged(P0, P1, colors0, colors1, R, t, scale):
    """Transform P0 into P1 frame and merge."""
    P0_t = scale * (P0 @ R.T) + t
    pts = np.vstack([P0_t, P1])
    col = np.vstack([colors0, colors1])
    return make_pcd(pts, col)


def load_mask(mask_path, size, dataset, rotate_for_original_frame=True):
    """Load fisheye mask -> flat boolean array matching cloud ordering.

    rotate_for_original_frame: when True (standard pairs, unrotated images),
      applies a 90° CCW rotation to MaskADT_rot.png to recover MaskADT.png,
      which is the correct mask for clouds from unrotated images.
      Set to False for custom pairs whose images were physically rotated 90° CW
      before UniK3D: MaskADT_rot.png is already in the correct orientation.
    """
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize((size, size), Image.NEAREST)
    mask = np.array(mask) / 255.0   # (H, W, 3)
    if dataset == "adt" and rotate_for_original_frame:
        mask = np.rot90(mask, k=1)  # 90° CCW: MaskADT_rot → MaskADT (for unrotated clouds)
    mask = mask.transpose(2, 1, 0)  # (3, W, H)
    mask = mask.reshape(3, -1).T    # (H*W, 3)
    mask = mask[:, 2]               # any channel
    return (mask > 0.5)


def filter_by_distance(P, colors, threshold):
    d = np.linalg.norm(P, axis=1)
    mask = d < threshold
    return P[mask], colors[mask]


def resolve_render_size(width, height, camera_json=None):
    """Keep requested size, but never render below the saved camera resolution."""
    requested_width = int(width)
    requested_height = int(height)
    width = requested_width
    height = requested_height

    if camera_json and Path(camera_json).exists():
        try:
            cam_params = o3d.io.read_pinhole_camera_parameters(camera_json)
            cam_width = int(cam_params.intrinsic.width)
            cam_height = int(cam_params.intrinsic.height)
            width = max(width, cam_width)
            height = max(height, cam_height)
            if (width, height) != (requested_width, requested_height):
                print(
                    f"[INFO] Increasing render size from "
                    f"{requested_width}x{requested_height} to {width}x{height} "
                    f"to match camera parameters"
                )
        except Exception as exc:
            print(f"[WARN] Failed to read camera resolution from {camera_json}: {exc}")

    return width, height


def apply_view_control_camera(view_control, cam_params):
    """Load camera params while preserving arbitrary roll when Open3D supports it."""
    try:
        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    except TypeError:
        view_control.convert_from_pinhole_camera_parameters(cam_params)


# ---------------------------------------------------------------------------
# Open3D offscreen rendering
# ---------------------------------------------------------------------------

def render_pcd(pcd, width, height, camera_json=None, point_size=2.0,
               background=(1, 1, 1)):
    """Render a point cloud and return a (H,W,3) uint8 image.

    Prefer `OffscreenRenderer` for remote/headless Linux. On Windows and some
    local Open3D builds, EGL headless rendering is unavailable, so we fall back
    to a regular `Visualizer` screen capture.
    """
    try:
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size)

        render.scene.add_geometry("cloud", pcd, mat)
        render.scene.set_background(list(background) + [1.0])

        if camera_json and Path(camera_json).exists():
            cam_params = o3d.io.read_pinhole_camera_parameters(camera_json)
            render.setup_camera(cam_params.intrinsic, cam_params.extrinsic)
        else:
            bb = pcd.get_axis_aligned_bounding_box()
            render.scene.camera.look_at(bb.get_center(),
                                        bb.get_center() + np.array([0, 0, -1]),
                                        np.array([0, -1, 0]))

        img = render.render_to_image()
        render.scene.clear_geometry()
        return np.asarray(img)[..., :3]  # drop alpha if present

    except Exception as exc:
        print(f"[WARN] OffscreenRenderer unavailable ({exc}); falling back to Visualizer capture.")

        vis = o3d.visualization.Visualizer()
        try:
            ok = vis.create_window(window_name="Open3D Render",
                                   width=width,
                                   height=height,
                                   visible=False)
        except TypeError:
            ok = vis.create_window(window_name="Open3D Render",
                                   width=width,
                                   height=height)
        if not ok:
            raise RuntimeError("Failed to create Open3D Visualizer window for fallback rendering") from exc

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = np.array(background, dtype=float)

        ctr = vis.get_view_control()
        if camera_json and Path(camera_json).exists():
            cam_params = o3d.io.read_pinhole_camera_parameters(camera_json)
            apply_view_control_camera(ctr, cam_params)
        else:
            bb = pcd.get_axis_aligned_bounding_box()
            ctr.set_lookat(bb.get_center())
            ctr.set_front(np.array([0.0, 0.0, -1.0]))
            ctr.set_up(np.array([0.0, -1.0, 0.0]))
            ctr.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        vis.destroy_window()
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def stack_images(imgs, pad=4):
    """Horizontally stack (H,W,3) uint8 images with padding."""
    h = max(i.shape[0] for i in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] < h:
            p = np.full((h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.vstack([im, p])
        padded.append(im)
        if pad > 0:
            padded.append(np.full((h, pad, 3), 180, dtype=np.uint8))
    return np.hstack(padded[:-1] if pad > 0 else padded)


def resize_img(img_np, target_h):
    """Resize PIL-compatible (H,W,3) uint8 to target height, keeping aspect."""
    h, w = img_np.shape[:2]
    target_w = int(w * target_h / h)
    return np.array(Image.fromarray(img_np).resize((target_w, target_h),
                                                   Image.LANCZOS))


def orient_display_image(img_np, rotation_degrees=0):
    """Apply an optional clockwise quarter-turn rotation to display image panels."""
    rotation_degrees = int(rotation_degrees) % 360
    if rotation_degrees not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported image rotation: {rotation_degrees}. Use 0/90/180/270.")

    k_map = {
        0: 0,
        90: 3,   # 90° clockwise
        180: 2,
        270: 1,  # 90° counter-clockwise
    }
    if rotation_degrees == 0:
        return img_np
    return np.ascontiguousarray(np.rot90(img_np, k=k_map[rotation_degrees]))


def add_label(img, text, font_scale=1.6):
    """Burn a larger text label into the top-left of a (H,W,3) uint8 image using PIL."""
    from PIL import ImageDraw, ImageFont

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font_px = max(24, int(40 * font_scale))
    x0, y0 = 10, 8

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_px)
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((x0, y0), text, font=font)
        rect = [bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6]
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)
        rect = [x0 - 8, y0 - 6, x0 + text_w + 8, y0 + text_h + 6]

    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dist_thr = args.distance_threshold

    # ── Load point clouds ─────────────────────────────────────────────────
    P0_raw = load_cloud(args.point0)
    P1_raw = load_cloud(args.point1)

    # Determine image size from cloud shape for color loading
    # Load images for colours
    img0 = np.array(Image.open(args.img0).convert("RGB"))
    img1 = np.array(Image.open(args.img1).convert("RGB"))
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]

    # Transpose-based indexing (as used in load_cloud)
    colors0_raw = (img0 / 255.0).transpose(2, 1, 0).reshape(3, -1).T
    colors1_raw = (img1 / 255.0).transpose(2, 1, 0).reshape(3, -1).T

    P0, colors0 = filter_by_distance(P0_raw, colors0_raw, dist_thr)
    P1, colors1 = filter_by_distance(P1_raw, colors1_raw, dist_thr)

    # ── Fisheye mask ───────────────────────────────────────────────────────
    if args.mask:
        size = 1408 if args.dataset == "adt" else 1400
        # For custom pairs (--rotate-gt), images were rotated before UniK3D,
        # so MaskADT_rot.png already matches the cloud orientation.
        rotate_mask = not args.rotate_gt
        mask = load_mask(args.mask, size, args.dataset, rotate_for_original_frame=rotate_mask)
        # distance filter may have changed array length; apply mask only if sizes match
        if len(mask) == len(P0_raw):
            mask_filtered = mask[np.linalg.norm(P0_raw, axis=1) < dist_thr]
            if len(mask_filtered) == len(P0):
                P0, colors0 = P0[mask_filtered], colors0[mask_filtered]
            mask_filtered1 = mask[np.linalg.norm(P1_raw, axis=1) < dist_thr]
            if len(mask_filtered1) == len(P1):
                P1, colors1 = P1[mask_filtered1], colors1[mask_filtered1]
        else:
            print(f"[WARN] Mask size {len(mask)} != cloud size {len(P0_raw)}, skipping mask")

    # ── GT pose ────────────────────────────────────────────────────────────
    R_gt, t_gt = load_gt_pose(args.cam2w0, args.cam2w1)

    # For pairs whose images were physically rotated 90° CW before running
    # UniK3D (i.e. custom pairs prepared with prepare_render_pairs.py --rotate),
    # the point clouds are in the rotated camera frame, so the GT must be
    # converted to match: R_gt_rot = R_CW90 @ R_gt @ R_CW90.T, t_rot = R_CW90 @ t.
    # Pass --rotate-gt to enable this correction.
    if args.rotate_gt:
        R_gt = R_CW90 @ R_gt @ R_CW90.T
        t_gt = R_CW90 @ t_gt

    # Save GT pose in the same format as method result JSONs
    gt_json_path = Path(args.results_dir) / "result_gt.json"
    gt_json_path.parent.mkdir(parents=True, exist_ok=True)
    gt_json_path.write_text(json.dumps({
        "R_est": R_gt.tolist(),
        "t_est": t_gt.tolist(),
        "scale_est": 1.0,
    }, indent=2))
    print(f"[INFO] Saved GT pose -> {gt_json_path}")

    pcd_gt = build_merged(P0, P1, colors0, colors1, R_gt, t_gt, scale=1.0)

    # ── Method poses ──────────────────────────────────────────────────────
    method_pcds = []
    for name_json in args.methods.split(","):
        name_json = name_json.strip()
        if ":" not in name_json:
            continue
        name, fname = name_json.split(":", 1)
        json_path = Path(args.results_dir) / fname
        result = load_method(str(json_path))
        if result is None:
            print(f"[SKIP] {name}: {json_path} not available or null")
            continue
        R, t, scale = result
        d_t, d_R = compute_pose_error(R_gt, t_gt, R, t)
        print(f"[METRIC] {name}: d_R = {d_R:.3f} deg, d_t = {d_t:.3f}, scale = {scale:.4f}")
        pcd = build_merged(P0, P1, colors0, colors1, R, t, scale)
        method_pcds.append((name, pcd, d_t, d_R))

    if not method_pcds:
        print("[WARN] No method results loaded — only GT will be rendered")

    # ── Render ────────────────────────────────────────────────────────────
    cam_json = args.camera_json
    W, H = resolve_render_size(args.width, args.height, cam_json)

    panels = []

    # Input images
    img0_display = orient_display_image(img0, args.image_rotation)
    img1_display = orient_display_image(img1, args.image_rotation)
    img0_panel = resize_img(img0_display, H)
    img1_panel = resize_img(img1_display, H)
    img0_panel = add_label(img0_panel, "Image 0", args.label_font_scale)
    img1_panel = add_label(img1_panel, "Image 1", args.label_font_scale)
    panels += [img0_panel, img1_panel]

    # GT
    gt_render = render_pcd(pcd_gt, W, H, cam_json, args.point_size)
    gt_render = add_label(gt_render, "GT", args.label_font_scale)
    panels.append(gt_render)

    # Methods
    for name, pcd, d_t, d_R in method_pcds:
        r = render_pcd(pcd, W, H, cam_json, args.point_size)
        r = add_label(r, name, args.label_font_scale)
        panels.append(r)

    panel = stack_images(panels)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(str(out))
    print(f"[DONE] Saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--point0",   required=True, help="UniK3D .npy cloud for image 0")
    ap.add_argument("--point1",   required=True, help="UniK3D .npy cloud for image 1")
    ap.add_argument("--img0",     required=True, help="Image 0 (for colours)")
    ap.add_argument("--img1",     required=True, help="Image 1 (for colours)")
    ap.add_argument("--cam2w0",   required=True, help="cam-to-world TXT for image 0")
    ap.add_argument("--cam2w1",   required=True, help="cam-to-world TXT for image 1")
    ap.add_argument("--methods",  required=True,
                    help='Comma-separated "Label:result_filename.json" pairs')
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing per-method result JSON files")
    ap.add_argument("--output",   required=True, help="Output PNG path")
    ap.add_argument("--mask",         default=None,
                    help="Fisheye mask image (PNG). Applied after distance filter.")
    ap.add_argument("--dataset",  default="adt", choices=["adt", "kitti"],
                    help="Dataset type (affects GT coordinate convention)")
    ap.add_argument("--distance-threshold", type=float, default=50.0,
                    help="Max point distance from camera (metres)")
    ap.add_argument("--point-size",  type=float, default=2.0)
    ap.add_argument("--width",       type=int,   default=1200,
                    help="Width of each cloud rendering panel")
    ap.add_argument("--height",      type=int,   default=900,
                    help="Height of each cloud rendering panel (also scales input images)")
    ap.add_argument("--camera-json", default=None,
                    help="Open3D pinhole camera params JSON for a fixed view")
    ap.add_argument("--image-rotation", type=int, default=0,
                    help="Clockwise rotation (degrees) applied to the input image panels: 0/90/180/270")
    ap.add_argument("--label-font-scale", type=float, default=1.6,
                    help="Scale factor for panel label font size")
    ap.add_argument("--rotate-gt", action="store_true", default=False,
                    help="Apply 90° CW frame correction to GT pose (use for custom ADT pairs "
                         "whose images were physically rotated before UniK3D inference)")
    main(ap.parse_args())
