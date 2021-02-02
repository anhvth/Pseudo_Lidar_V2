python ./src/main.py -c src/configs/sdn_kitti_train.config \
    --resume pretrained/sdn_kitti_object.tar --datapath  ./data/kitti/training/ \
    --data_list ./split/trainval.txt --generate_depth_map --data_tag trainval