#trainval
python ./src/main.py -c src/configs/sdn_kitti_train.config \
    --resume pretrained/sdn_kitti_object.tar --datapath  ./data/kitti/training/ \
    --data_list ./split/trainval.txt --generate_depth_map --data_tag trainval


#test
python ./src/main.py -c src/configs/sdn_kitti_train.config \
    --resume pretrained/sdn_kitti_object.tar --datapath  ./data/kitti/testing/ \
    --data_list ./split/test.txt --generate_depth_map --data_tag test