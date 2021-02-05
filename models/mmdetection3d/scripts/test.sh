CONFIG='configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
CKPT=work_dirs/hv_second_secfpn_6x8_80e_kitti-3d-3class/epoch_40.pth
./tools/dist_test.sh $CONFIG $CKPT 4 --out $CKPT".pkl"