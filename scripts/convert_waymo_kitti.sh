# prefix 0: train, 1: valid, 2: test
cd waymo_kitti_converter

python converter.py /data/waymo/tfs ./data/waymo/training --prefix 0 --num_proc $1