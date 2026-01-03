"""Inference helper used by `slurm_inference_triton.sh`.

Usage (from SLURM wrapper):
  srun python scripts/triton_inference.py --config-file configs/train/vitb.json \
      --data-source-dir /path/to/images --output-dir /path/to/out --images-output-dir /path/to/out/images
"""
import argparse
import glob
from html import parser
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps as mpl_colormaps

from unik3d.models import UniK3D

import matplotlib.pyplot as plt
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)

import huggingface_hub

from safetensors.torch import load_file as _load_safetensors

def load_model_from_config(cfg):
         
    # Build model from config and load pretrained if provided
    model = UniK3D(cfg)
    pretrained = cfg.get("training", {}).get("pretrained", None)
    if pretrained:
        # prefer safetensors file; assume config points to a file
        try:
            # try safetensors first
            if str(pretrained).endswith(".safetensors"):                

                state = _load_safetensors(pretrained)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                model.load_state_dict(state, strict=False)
            else:
                # fallback to torch.load for .bin/.pth
                st = torch.load(pretrained, map_location="cpu")
                if isinstance(st, dict) and "model" in st:
                    st = st["model"]
                model.load_state_dict(st, strict=False)
        except Exception as e:
            print("Warning: failed to load pretrained model:", e)
    return model

def load_model_from_pretrained_1(cfg):
         
        model = UniK3D(cfg)
        backbone = cfg.get("model", {}).get("backbone", "vitb")
        path = huggingface_hub.hf_hub_download(repo_id=f"lpiccinelli/unik3d-{backbone}", filename=f"pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print(f"UniK3D-{backbone} is loaded with:")
        print(f"\t missing keys: {info.missing_keys}")
        print(f"\t additional keys: {info.unexpected_keys}")

        return model

def find_images(src_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(src_dir, e))))
    return files


def colorize_depth(depth_np, cmap_name="inferno"):
    d = depth_np.copy()
    # mask NaNs
    nan_mask = np.isnan(d)
    if nan_mask.all():
        return Image.new("RGB", (d.shape[1], d.shape[0]), color=(0, 0, 0))
    d[nan_mask] = 0
    d = d - d.min()
    mx = d.max()
    if mx > 0:
        d = d / mx
    cmap = mpl_colormaps.get_cmap(cmap_name)
    rgba = cmap(d)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)

def instantiate_camera(camera_name, params, device):
    if camera_name == "Predicted":
        return None
    fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2, hfov, H, W = params
    if camera_name == "Pinhole":
        params = [fx, fy, cx, cy]
    elif camera_name == "Fisheye624":
        params = [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2]
    elif camera_name == "OPENCV":
        params = [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2]
    elif camera_name == "Equirectangular":
        # dummy intrinsics for spherical camera, assume hfov -> vfov based on input shapes
        hfov2 = hfov * torch.pi / 180.0 / 2
        params = [fx, fy, cx, cy, W, H, hfov2, H / W * hfov2]
        camera_name = "Spherical"

    return eval(camera_name)(params=torch.tensor(params).float()).to(device)


def instantiate_camera_1(args):
    
    camera = None
    camera_path = args.camera_path
    
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)

        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)
    return camera

def main():
    
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/train/vitb.json")
    p.add_argument("--data-source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--images-output-dir", required=True)
    p.add_argument("--master-port", type=str)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--camera-path", type=str)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params=[]
    camera = instantiate_camera_1(args)

    cfg = json.load(open(args.config_file, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl") # vitl for ViT-L backbone

    #model = load_model_from_pretrained_1(cfg)
    model = model.to(device)
    model.eval()
    model.resolution_level=1

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.images_output_dir, exist_ok=True)

    img_paths = find_images(args.data_source_dir)
    
    if not img_paths:
        print("No images found in", args.data_source_dir)
        return

    for img_path in img_paths:
        name = Path(img_path).stem
        
        print("Processing", img_path)
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)

        #with torch.no_grad():
        #preds = model.infer(tensor, camera=camera, normalize=True, rays=None)
        preds = model.infer(tensor, camera=None, normalize=True, rays=None)
        

        print("Predictions keys:", preds.keys())

        depth = preds.get("depth", None)
        if depth is None:
            print("No depth returned for", img_path)
            continue

        if isinstance(depth, torch.Tensor):
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = np.array(depth)

        # Save colorized depth and raw arrays
        depth_img = colorize_depth(depth_np)
        depth_img.save(os.path.join(args.images_output_dir, f"{name}_depth.png"))
        np.save(os.path.join(args.output_dir, f"{name}_depth.npy"), depth_np)

        if "points" in preds:
            pts = preds["points"]
            if isinstance(pts, torch.Tensor):
                pts = pts.cpu().numpy()
            np.save(os.path.join(args.output_dir, f"{name}_points.npy"), pts)

        print("Saved outputs for", name)

        if "rays" in preds:
            rays = preds["rays"]
            if isinstance(rays, torch.Tensor):
                rays = rays.cpu().numpy()
            np.save(os.path.join(args.output_dir, f"{name}_rays.npy"), rays)


if __name__ == "__main__":
    main()

