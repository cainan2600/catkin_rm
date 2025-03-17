import sys
import os
import numpy as np
import torch


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # 关键改动：'../..' 而不是 '../../..'
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from RPSN_4.run_start import start
    
    
def all_object_position_to_7(all_object_position):
    for step_all_obj_position in all_object_position:
        # for step_obj_position in step_all_obj_position:
            current_len = len(step_all_obj_position)
            i = 0
            while current_len < 7:
                
                element = step_all_obj_position[i]
                step_all_obj_position.append(element)
                i += 1
                current_len += 1

    return np.array(all_object_position)
    
    

def main_RPSN(all_object_position):
    '''
    Location_of_objects_list = [
    [
        [1.0795431990611055, -0.651733994125367, 2.3202460294687204, 3.359652528665011, 0.14049140175077102, 0.043154842569564665],
        [2.414453562280456, 0.3016376404923616, -0.9324502881535464, 3.8533742591570808, 0.004817449442475252, 0.05296482461208256],
        [-2.5870679261087712, 0.2627746184031503, 1.6741349722522685, 2.740858388572259, 0.210222602052977, 0.03833761843056219],
        [-1.5381527451771089, 0.3492733294929023, 3.127002040915903, 3.731611643324439, 0.14583353860873177, 0.032363500315085815],

    ],
    [
        [1.0795431990611055, -0.651733994125367, 2.3202460294687204, 3.359652528665011, 0.14049140175077102, 0.043154842569564665],
        [2.414453562280456, 0.3016376404923616, -0.9324502881535464, 3.8533742591570808, 0.004817449442475252, 0.05296482461208256],

    ]
    
    ]
    # 1.用有效数字填充满7个！！！！！！！！！！！！！！！！！！！并且转换为tensor类型

    # 2.输入RPSN得到底盘位置
    
    # 3.得到底盘位置按顺序排列
    # all_tar_chasis_position = [[x, y, z, w], [x, y, z, w], [x, y, z, w]]
    # all_object_position = [[[[x, y, z, w], [x, y, z, w]], [[x, y, z, w], [x, y, z, w]]], [[[x, y, z, w], [None]]], [[[None], [x, y, z, w]]]]
    '''

    all_object_position_to7 = all_object_position_to_7(all_object_position)
    # print(all_object_position_to7)

    all_object_position_to7_totensor = torch.FloatTensor(all_object_position_to7)

    all_tar_chasis_position = start(all_object_position_to7_totensor)

    return all_tar_chasis_position
    
    



if __name__ == "__main__":
    # a = run_start.main()
    # a.train()
    all_object_position = [
        [[1.0795431990611055, -0.651733994125367, 2.3202460294687204, 3.359652528665011, 0.14049140175077102, 0.043154842569564665],
        [2.414453562280456, 0.3016376404923616, -0.9324502881535464, 3.8533742591570808, 0.004817449442475252, 0.05296482461208256]],

        [[1.0795431990611055, -0.651733994125367, 2.3202460294687204, 3.359652528665011, 0.14049140175077102, 0.043154842569564665],
        [2.414453562280456, 0.3016376404923616, -0.9324502881535464, 3.8533742591570808, 0.004817449442475252, 0.05296482461208256],
        [-2.5870679261087712, 0.2627746184031503, 1.6741349722522685, 2.740858388572259, 0.210222602052977, 0.03833761843056219],
        [-1.5381527451771089, 0.3492733294929023, 3.127002040915903, 3.731611643324439, 0.14583353860873177, 0.032363500315085815]]
        ]
    main_RPSN(all_object_position)