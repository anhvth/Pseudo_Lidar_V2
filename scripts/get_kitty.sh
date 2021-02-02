!wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

mkdir -p data/kitti
rsync -avzhe ssh --progress /mnt/vinai-public-dataset/3dod/training/ ./data/kitti/training
rsync -avzhe ssh --progress /mnt/vinai-public-dataset/3dod/testing/ ./data/kitti/testing