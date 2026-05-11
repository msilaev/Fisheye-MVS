#!/bin/bash

train_list=("2013_05_28_drive_0000_sync"
           )
cam_list=("02")

LOCAL_DIR_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360"

root_dir=$LOCAL_DIR_ROOT #KITTI-360
data_2d_dir=data_2d_raw

mkdir -p $root_dir
mkdir -p $root_dir/$data_2d_dir

cd $root_dir 

# fisheye images
for sequence in ${train_list[@]}; do
    for camera in ${cam_list[@]}; do 
	zip_file=${sequence}_image_${camera}.zip
        wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}
	unzip -d ${data_2d_dir} ${zip_file} 
	rm ${zip_file}
    done
done

# timestamps
zip_file=data_timestamps_fisheye.zip
wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}
unzip -d ${data_2d_dir} ${zip_file}
rm $zip_file
