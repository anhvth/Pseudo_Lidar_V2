from glob import glob
import mmcv
import time
timer = mmcv.utils.Timer()
file_dir = '/Pseudo_Lidar_V2/waymo_kitti_converter/data/waymo/training/calib'

get_num = lambda : len(glob(file_dir+'/*'))

previous = get_num()
start = timer.start()
while True:
    now = get_num()
    speed = (now-previous)/timer.since_start()
    print('Generating speed: {:0.2f} files/s'.format(speed))
    # previous = now
    time.sleep(2)
        
