import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models import MLP_3, MLP_6, MLP_9
from lib.trans_all import *
from lib import IK, IK_loss, planner_loss
import torch
import torch.nn as nn
import math
import os
import sys
from lib.save import checkpoints
from lib.plot import *
from data.data_generate_fk_ik import save_data, save_MLP_output

from torch.optim.lr_scheduler import ReduceLROnPlateau



def start(all_object_position):

    data = TensorDataset(all_object_position)
    data_loader_test = DataLoader(data, batch_size=1, shuffle=False)

    link_length = torch.tensor([0, 0, 0.256, 0, 0, 0])
    link_offset = torch.tensor([0.2405, 0, 0, 0.210, 0, 0.274])
    link_twist = torch.FloatTensor([0, math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2])

    num_i = 6
    num_h = 32
    num_o = 3
    model_MLP = MLP_9
    model = model_MLP.MLP_self(num_i , num_h, num_o, num_heads=1)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.000)  # 定义优化器

    model_path = r'/home/cn/catkin_rm/src/RPSN_4/work_dir/batch_back/checkpoint-epoch100.pt'
    if os.path.exists(model_path):          
        checkpoint = torch.load(model_path)  
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        print('-' * 100 + '\n' + f"checkpoint is loaded" + '\n' + '-' * 100)
    else:
        print('-' * 100 + '\n' + "NO checkpoint" + '\n' + '-' * 100)


    all_tar_chasis_position = []    
    for data_test in data_loader_test:
        # print(len(data_loader_test))
        num_correct_but_dipan_in_tabel = 0
        NUM_all_have_solution = 0
        # print("data_test", data_test)
        for data_step in data_test:
            # print(len(data_step))
            for data_step_step in data_step:
                
                # print(len(data_step_step))
                data_step_step_random = data_step_step[torch.randperm(data_step_step.size(0))]
                # print(data_step_step_random)
                intermediate_outputs = model(data_step_step_random)

                # yaw,x,y
                intermediate_outputs_list = intermediate_outputs.detach().numpy()
                output_list = [0,0,0,0]
                output_list[0] = intermediate_outputs_list[1] # +xy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                output_list[1] = intermediate_outputs_list[2]

                input_tar = shaping(data_step_step)

                # print(intermediate_outputs)  
                outputs = torch.empty((0, 6)) # 创建空张量
                # for each_result in intermediate_outputs: # 取出每个batch_size中的每个数据经过网络后的结果1x3
                pinjie1 = torch.cat([intermediate_outputs, torch.zeros(1).detach()])
                pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                outputs = torch.cat([outputs, pinjie2.unsqueeze(0)], dim=0)
                MLP_output_base = shaping(outputs)

                num_all_have_solution = 0
                num_not_all_0 = 0

                for i in range(len(input_tar)):
                    if torch.all(data_step_step[i].ne(0)):
                        num_not_all_0 += 1
                        num_all_have_solution += 1
                        angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan = IK.calculate_IK(
                            input_tar[i], 
                            MLP_output_base, 
                            link_length, 
                            link_offset, 
                            link_twist)
                        IK_loss1, num_NOError1, num_NOError2 = IK_loss.calculate_IK_loss(angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan)
                        num_all_have_solution = num_all_have_solution - num_NOError1

                if num_all_have_solution == num_not_all_0:
                    NUM_all_have_solution += 1
                    
                    if -1<intermediate_outputs_list[1]<1:
                        if -0.425<intermediate_outputs_list[2]<0.425:
                            num_correct_but_dipan_in_tabel += 1


                            # 转换为四元数形式xyzw
                            rot = euler_to_rotMat(intermediate_outputs_list[0], 0, 0)
                            q_target = rotation_matrix_to_quaternion(rot)
                            output_list[2] = q_target[2]
                            output_list[3] = q_target[3]
                            all_tar_chasis_position.append(output_list)
                            

                

    if NUM_all_have_solution == len(data_loader_test):
        print("have solution")
        if num_correct_but_dipan_in_tabel == len(data_loader_test):
            print("solution not in table")

        else:
            # print("NO solution")
            sys.exit("Exiting the program due to solution in table.")
    else:

        sys.exit("Exiting the program due to NO solution.")

    # except NUM_all_have
    # 再写一个LLM报错的！！！！！！！！！！！！！！！！！！！！！！！！！！！！！





    return all_tar_chasis_position


if __name__ == "__main__":
    a = start(
        torch.FloatTensor([
        [[1.0795431990611055, -0.651733994125367, 2.3202460294687204, 3.359652528665011, 0.14049140175077102, 0.043154842569564665],
        [2.414453562280456, 0.3016376404923616, -0.9324502881535464, 3.8533742591570808, 0.004817449442475252, 0.05296482461208256],
        [-2.5870679261087712, 0.2627746184031503, 1.6741349722522685, 2.740858388572259, 0.210222602052977, 0.03833761843056219],
        [-1.5381527451771089, 0.3492733294929023, 3.127002040915903, 3.731611643324439, 0.14583353860873177, 0.032363500315085815],
        [2.079981464438941, 1.5406901242609825, 0.4278294944284019,3.0330625281428505, 0.04235119928988959, 0.05668868563346661],
        [1.816117760202489, 1.4113224159181246, -2.009506160620853, 3.6895157232542517, 0.04988997499705217, 0.027776114858980322],
        [1.3528615748031758, 0.4567567315212199, 1.8470664086382256, 3.3490107619168166, 0.7646220864967979, 0.020519138612814267]],

        [[1.0795431990611055, -0.651733994125367, 2.3202460294687204, 3.359652528665011, 0.14049140175077102, 0.043154842569564665],
        [2.414453562280456, 0.3016376404923616, -0.9324502881535464, 3.8533742591570808, 0.004817449442475252, 0.05296482461208256],
        [-2.5870679261087712, 0.2627746184031503, 1.6741349722522685, 2.740858388572259, 0.210222602052977, 0.03833761843056219],
        [-1.5381527451771089, 0.3492733294929023, 3.127002040915903, 3.731611643324439, 0.14583353860873177, 0.032363500315085815],
        [2.079981464438941, 1.5406901242609825, 0.4278294944284019,3.0330625281428505, 0.04235119928988959, 0.05668868563346661],
        [1.816117760202489, 1.4113224159181246, -2.009506160620853, 3.6895157232542517, 0.04988997499705217, 0.027776114858980322],
        [1.3528615748031758, 0.4567567315212199, 1.8470664086382256, 3.3490107619168166, 0.7646220864967979, 0.020519138612814267]]
        ])
    )
