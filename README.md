# 🎯 Relative Pose Estimation and 3D Reconstruction for Fisheye Cameras through UniK3D Monocular 3D Geometry Priors

This repository implements a **Unik3D → Multiview geometry** pipeline, providing the alignment of point clouds and camera poses from two monocular point cloud estimations. 

The pipeline uses inference scripts from the following repositories:

- [Unik3D](https://github.com/lpiccinelli-eth/UniK3D)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

based on the papers:

- [UniK3D: Universal Camera Monocular 3D Estimation](https://arxiv.org/abs/2503.16591)
- [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763)

Experiments were done with fisheye image sequences from:

- [Aria Digital Twin dataset](https://explorer.projectaria.com/adt)
- [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)


Below is an example using the [Aria Digital Twin dataset](https://explorer.projectaria.com/adt). More examples can be found on the [Demo Page](https://anonymous.4open.science/r/Fisheye-MVS-demo-4F47/). 

## 📊 Examples

- [Apartment_release_clean_seq133_M1292](https://explorer.projectaria.com/adt/Apartment_release_clean_seq133_M1292?st=%220%22)

---

| Source images | Combined 3D point cloud, Estimated camera pose  |
|:------:|:----------:|
| <img src="assets/Apartment_release_clean_seq133_M1292/image0.jpg" width="170"/><br><img src="assets/Apartment_release_clean_seq133_M1292/image1.jpg" width="170"/> | <img src="assets/Apartment_release_clean_seq133_M1292/rotation_est.gif" width="600"/> |

---

## ⚙️ Setup, Installation & Running

### 🖥️ 1. Hardware and System Environment

This project was developed, tested, and run in the following hardware/system environment:

```
Hardware Environment

    CPU(s)        13th Gen Intel® Core™ i7-13700K x 24 
    GPU           NVIDIA GeForce RTX 4090 — 24 GB VRAM
    RAM           125 GB total
    Disk(s)       3.6 TB HDD, 1.8 TB NVMe SSD 

System Environment:
    OS            Ubuntu 22.04.5 LTS
    NVIDIA Driver 550.144.03
    Conda         version: conda 25.3.1
```

The presented configuration relies on the SSH connection to the remote machine. 
Put corresponding REMOTE_USER and REMOTE_HOST in the .env file (see example in .env_example)


### 📦 2. Environment Setup 

1. On the local machine: clone the repository and install dependencies for pose estimattion and visualisation

```bash
cd $LOCAL_ROOT
git clone https://anonymous.4open.science/r/Fisheye-MVS-86EB/
cd Fisheye-MVS
conda create -n 3d_pose_env python=3.8
conda activate 3d_pose_env
pip install -r requirements.txt
```

On the remote machine:

1. Create base environment for pose estimation:

```bash
conda create -n 3d_pose_env python=3.8
conda activate 3d_pose_env
pip install numpy>=1.24 scipy>=1.11 pillow>=9.5 imageio>=2.31
```

2. Clone Unik3D repository and install dependencies:

```bash
cd $REMOTE_DIR_ROOT
git clone git@github.com:lpiccinelli-eth/UniK3D.git
cd UniK3D
conda create -n mvf-unik3d python=3.11
conda activate mvf-unik3d
pip install -r requirements.txt
pip install psutil
```

3. Clone SuperGlue repository and install dependencies:

```bash
cd $REMOTE_DIR_ROOT
git clone git@github.com:magicleap/SuperGluePretrainedNetwork.git
cd SuperGluePretrainedNetwork
conda create -n superglue38 python=3.8
conda activate superglue38
pip install -r requirements.txt
```


**Summary of virtual environments:**

On the remote machine:
- `mvf-unik3d`: UniK3D requirements
- `superglue38`: SuperGlue requirements
- `3d_pose_env`: Pose estimation dependencies

On the local machine:
- `3d_pose_env`: Pose estimation and visualization dependencies

4. Define paths in `path_config.sh` (see example in `path_config_example.sh`)

### 📦 3. Prepare images 

Put pair of images, e.g. image0.jpg, image1.jpg  and text file image_pair.txt with their names (in one string, separated by space image0.jpg image1.jpg ) in separate folder. An example of local folder structure is

``` 
$LOCAL_ROOT/
├── Fisheye-MVS/
└── experiments/
    ├── KITTI-360
    ├    ├──IMAGES_DIR_experiment_1
         |           ├──image0.jpg image1.jpg image_pair.txt
         results   
```

Define the following in the config file:
- Directory with images
- Image extension and size in pixels
- Distance threshold (to filter out sky points)
- Path to image mask

Example config files:
- `experiment_config_adt.sh` for ADT images
- `experiment_config_kitti.sh` for KITTI-360 images

Source the configuration file in `run_pipeline_local.sh` and `run_visualise_local.sh`


### 🚀 Running Pipelines 

**Run the pipeline to solve for the relative pose:**
```bash
cd $LOCAL_ROOT
./run_pipeline_local.sh
```

**Fetch the results and visualise the combined 3D point cloud for the current local pair:**
```bash
./run_visualise_local.sh
```

---

## 🖼️ Rendering evaluation pairs and exact custom frame pairs

### What the render jobs do

`run_visualise_frames_adt.sh` and `run_visualise_frames_kitti.sh` are local launcher scripts that:

1. **Resolve pairs** — verify which named pairs exist on the remote, and pass any `frames:<f0>|<f1>` specs through unchanged for on-demand creation.
2. **Upload scripts** — sync `render_merged_clouds.py`, `prepare_render_pairs.py`, `sbatch_render_examples.sh`, the fisheye mask, and the fixed-viewpoint camera JSON to the remote.
3. **Write a pairs list** — create a text file on the remote (`render_pairs_adt.txt` / `render_pairs_kitti.txt`) with one pair spec per line.
4. **Submit a SLURM GPU job** — call `sbatch` with `sbatch_render_examples.sh`, which for each pair:
   - Resolves `frames:` specs into real pair directories via `prepare_render_pairs.py` (creating a `pair_custom_*` folder from dataset images/poses if needed).
   - Runs **UniK3D** to produce per-image point clouds.
   - Copies all available result JSONs from `eval_full/results/<pair>/` into the pair output folder (these include `result_procrustes_ransac.json`, `result_madpose.json`, `result_madpose_rect.json`, `result_dust3r.json`, `result_pinhole.json` etc.).
   - Runs `render_merged_clouds.py` to produce a side-by-side comparison PNG (`GT | Ours+RANSAC | MADPose | MADPose+rect`).
5. **Print the job ID and fetch command** for use after the job completes.

The render job runs on the `gpu-v100-32g` partition (time limit 2 h, 1 GPU).

Output layout on the remote:

```text
experiments/<EXP>/eval_full/rendered_example/
├── <pair_name>.png              ← side-by-side comparison panel
└── <pair_name>/
    ├── image0_points.npy
    ├── image1_points.npy
    ├── image0.txt  /  image1.txt   ← GT poses
    └── result_*.json               ← copied from eval results
```

### How to submit

Submit for all default evaluation pairs plus the hardcoded custom `frames:` pairs:

```bash
bash ./src_visualise/run_visualise_frames_adt.sh
bash ./src_visualise/run_visualise_frames_kitti.sh
```

Submit for **only specific custom frame pairs** via the `PAIRS` override (space-separated, each spec quoted together):

```bash
PAIRS='frames:Apartment_release_clean_seq136_M1292_frame000900|Apartment_release_clean_seq136_M1292_frame000840 frames:Apartment_release_clean_seq136_M1292_frame002563|Apartment_release_clean_seq136_M1292_frame002537 frames:Apartment_release_decoration_seq136_M1292_frame002000|Apartment_release_decoration_seq136_M1292_frame001965 frames:Apartment_release_multiuser_cook_seq141_M1292_frame002033|Apartment_release_multiuser_cook_seq141_M1292_frame002090' \
  bash ./src_visualise/run_visualise_frames_adt.sh

PAIRS='frames:0000006360|0000006367 frames:0000008603|0000008587' \
  bash ./src_visualise/run_visualise_frames_kitti.sh
```

The scripts print the submitted SLURM job ID and the exact `fetch_rendered_examples.sh` command to run once the job finishes.

Monitor progress:
```bash
ssh <remote-user>@<remote-host> 'squeue -u <remote-user>'
```

---

The repository also supports rendering the merged-cloud comparison panels used in the evaluation folders:

```text
experiments/<EXP>/rendered_example/
├── pair_rot_...png
├── pair_custom_...png
└── pair_custom_.../
    ├── image0_points.npy
    ├── image1_points.npy
    ├── image0.txt
    ├── image1.txt
    └── result_*.json
```

### 1. Render the default evaluation examples on the remote machine

```bash
bash ./src_visualise/run_visualise_frames_adt.sh
bash ./src_visualise/run_visualise_frames_kitti.sh
```

### 2. Render exact custom frame pairs

You can pass explicit frame IDs through the `PAIRS` environment variable using the form:

```bash
frames:<frame0>|<frame1>
```

Examples:

```bash
PAIRS='frames:Apartment_release_clean_seq136_M1292_frame000900|Apartment_release_clean_seq136_M1292_frame000840' \
  bash ./src_visualise/run_visualise_frames_adt.sh

PAIRS='frames:0000006360|0000006367 frames:0000008603|0000008587' \
  bash ./src_visualise/run_visualise_frames_kitti.sh
```

If the exact pair does not already exist in `eval_full/test_pairs`, the helper `src_visualise/prepare_render_pairs.py` will automatically build a `pair_custom_*` directory from the dataset images and poses.

### 3. Fetch the rendered results locally

If you pass the SLURM job ID, the fetch script waits for the render job to finish and then downloads both the top-level PNGs and the per-pair directories:

```bash
bash ./src_visualise/fetch_rendered_examples.sh adt <JOB_ID>
bash ./src_visualise/fetch_rendered_examples.sh kitti <JOB_ID>
```

The files are saved to:

```bash
$LOCAL_ROOT/experiments/ADT_seq133/rendered_example/
$LOCAL_ROOT/experiments/KITTI-360/rendered_example/
```

---

## 💻 Produce missing PNGs locally from downloaded pair folders

Sometimes the per-pair folders are already available locally, but the top-level `pair_custom_*.png` summaries are still missing. In that case render them locally with:

`src_visualise_local/local_render_examples.sh`

### Requirements

Make sure your local visualisation environment has the dependencies (notably `open3d`).

### Render all pairs in a dataset folder

```bash
bash ./src_visualise_local/local_render_examples.sh adt
bash ./src_visualise_local/local_render_examples.sh kitti
```

### Render one particular pair

```bash
bash ./src_visualise_local/local_render_examples.sh adt \
  pair_custom_Apartment_release_clean_seq136_M1292_frame000900__Apartment_release_clean_seq136_M1292_frame000840

bash ./src_visualise_local/local_render_examples.sh kitti \
  pair_custom_0000006360__0000006367
```

### Render several specific pairs with `PAIRS`

```bash
PAIRS="pair_custom_Apartment_release_clean_seq136_M1292_frame000900__Apartment_release_clean_seq136_M1292_frame000840 \
pair_custom_Apartment_release_clean_seq136_M1292_frame002563__Apartment_release_clean_seq136_M1292_frame002537" \
  bash ./src_visualise_local/local_render_examples.sh adt

PAIRS="pair_custom_0000006360__0000006367 pair_custom_0000008603__0000008587" \
  bash ./src_visualise_local/local_render_examples.sh kitti
```

### Useful options via environment variables

- `PYTHON=/path/to/python` (default: `python`)
- `RENDER_WIDTH=1920` / `RENDER_HEIGHT=1080`
- `IMAGE_ROTATION_DEG=0|90|180|270`
- `LABEL_FONT_SCALE=1.6`

The script loads camera poses from each pair folder (`camera_adt.json` / `camera_kitti.json`) when available, and falls back to `src_visualise_local/camera_*.json`.

> On Windows, `open3d` may print `EGL Headless is not supported on this platform`. This is expected; rendering falls back to standard Open3D visualizer capture and still saves the PNG.

---

## 📏 Multiview consistency metrics for merged clouds

Use `src_metrics/compute_cloud_consistency.py` to score how well the merged point clouds agree for a rendered pair.

For each method, cloud `P0` (image 0) is transformed by the estimated similarity `scale * P0 @ R_est.T + t_est` and then compared against the untransformed cloud `P1` (image 1) using symmetric nearest-neighbour (NN) distances.

Let `d_ab` = per-point NN distances from transformed `P0` to `P1`, and `d_ba` = per-point NN distances from `P1` to transformed `P0`.

The script reports:

- **pose consistency**
  - `dR (deg)`: rotation error between estimated and GT relative rotation (geodesic angle)
  - `dt (m)`: Euclidean distance between estimated and GT translation vectors

- **geometry consistency** (lower = better alignment)
  - `ChamferL1`: symmetric Chamfer L1 distance = `mean(d_ab) + mean(d_ba)`
  - `ChamferL2`: symmetric Chamfer L2 distance = `mean(d_ab²) + mean(d_ba²)`, more sensitive to outliers
  - `RMSE`: symmetric root-mean-squared NN distance = `sqrt(mean([d_ab, d_ba]²))`, i.e. RMS over all pairwise distances combined

- **overlap consistency** (higher = more overlap; threshold set by `--overlap-threshold`)
  - `inlier_ratio`: fraction of points whose NN distance is below the overlap threshold (averaged over both directions)
  - `overlap_ratio`: same as mean inlier ratio = `0.5 * (mean(d_ab ≤ τ) + mean(d_ba ≤ τ))`

### Example: ADT

```bash
python src_metrics/compute_cloud_consistency.py \
  --pair-dir "$LOCAL_ROOT/experiments/ADT_seq133/rendered_example/pair_custom_Apartment_release_clean_seq136_M1292_frame000900__Apartment_release_clean_seq136_M1292_frame000840" \
  --dataset adt \
  --mask assets/fisheye_masks/MaskADT_rot.png \
  --distance-threshold 1000 \
  --overlap-threshold 0.05 \
  --max-points 200000 \
  --output cloud_consistency_sampled.json
```

### Example: KITTI-360

```bash
python src_metrics/compute_cloud_consistency.py \
  --pair-dir "$LOCAL_ROOT/experiments/KITTI-360/rendered_example/pair_custom_0000006360__0000006367" \
  --dataset kitti \
  --mask assets/fisheye_masks/MaskKitti360.png \
  --distance-threshold 20 \
  --overlap-threshold 0.05 \
  --max-points 200000 \
  --output cloud_consistency_sampled.json
```

Notes:

- `--max-points` speeds up scoring on dense clouds by random subsampling.
- The output JSON is written into the pair folder when `--output` is provided.
- The script can evaluate `GT`, `Ours`, `Ours+RANSAC`, `MADPose`, `MADPose+rect`, `Pinhole`, `DUSt3R`, and `DUSt3R+rect` when the corresponding result JSONs are present.

