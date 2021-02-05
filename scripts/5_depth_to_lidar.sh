# Convert depth maps to Pseudo-Lidar Point Clouds
# trainval
python ./src/preprocess/generate_lidar_from_depth.py --calib_dir  ./data/kitti/training/calib --depth_dir ./results/sdn_kitti_train_set/depth_maps/trainval/  --save_dir  ./results/sdn_kitti_train_set/pseudo_lidar_trainval/
#test
python ./src/preprocess/generate_lidar_from_depth.py --calib_dir  ./data/kitti/testing/calib --depth_dir ./results/sdn_kitti_train_set/depth_maps/test/  --save_dir  ./results/sdn_kitti_train_set/pseudo_lidar_test/
