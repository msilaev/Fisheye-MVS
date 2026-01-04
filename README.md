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


Below is an example using the [Aria Digital Twin dataset](https://explorer.projectaria.com/adt). More examples can be found on the [Demo Page](https://github.com/msilaev/Fisheye-MVS-demo). 

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
git clone https://github.com/msilaev/Fisheye-MVS
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

**Run the pipeline to get solve for the relative pose:**
```bash
cd $LOCAL_ROOT
./run_pipeline_local.sh
```

**Run the pipeline to fetch the results and visualise combined 3D point cloud:**
```bash
./run_visualise_local.sh
```

