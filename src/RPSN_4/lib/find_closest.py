import torch
import math
import numpy as np
from torchviz import make_dot

# def find_closest(angle_solution, where_is_the_illegal):

#     the_NANLOSS_of_illegal_solution_with_num_and_Nan = 0

#     min_distance = 100  # 记录非法数据中，距离3.14最近的数的绝对值距离，初始化为一个足够大的值
#     min_index = []      # 记录比较后距离3.14最近的值的索引
#     single_ik_loss = torch.tensor(0.0, requires_grad=True)

#     for index in where_is_the_illegal:
#         there_exist_nan = 0
#         i, j = index
#         if math.isnan(angle_solution[i][j]):
#             pass
#         else:
#             for angle in range(6):
#                 if math.isnan(angle_solution[i][angle]):
#                     there_exist_nan +=1
#             if there_exist_nan == 0:
#                 num = angle_solution[i][j]
#                 distance = abs(num) - (torch.pi)          # 计算拿出来的值距离(pi)的距离
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_index = index
#             else:
#                 pass
#         single_ik_loss = single_ik_loss + min_distance
#     return the_NANLOSS_of_illegal_solution_with_num_and_Nan

def find_closest(angle_solution, where_is_the_illegal):
    
    

    single_ik_loss = torch.tensor([0.0], requires_grad=True)

    fanwei1 = [math.pi * 178/180, math.pi * 130/180, math.pi * 135/180, math.pi * 178/180, math.pi * 128/180, math.pi]
    # jiaodu3_limited = torch.FloatTensor([math.pi * 130/180])
    



    for index in where_is_the_illegal:
        there_exist_nan = 0
        i, j = index
        if torch.isnan(angle_solution[i][j]):
            pass

        else:
            for angle in range(6):
                if torch.isnan(angle_solution[i][angle]):
                    there_exist_nan +=1
            if there_exist_nan == 0:
                diff_mini = 1000
                for angle_1 in range(6):
                    num = angle_solution[i][angle_1]
                    tar_num = fanwei1[angle_1]
                    if abs(num) > abs(tar_num):
                        diff = abs(num) - abs(tar_num)
                        if diff < diff_mini:
                            diff_mini = diff
                single_ik_loss = single_ik_loss + diff_mini * 1000
                # single_ik_loss = single_ik_loss + 100
                # diff_mini.register_hook(save_grad('diff_mini'))
                # print("[grads]diff_mini:", grads)
                print(single_ik_loss)
                        # print(single_ik_loss, the_NANLOSS_of_illegal_solution_with_num_and_Nan)
            else:
                pass
            # num_diff = (abs(angle_solution[i][j]) - abs(fanwei1[j])) * 100
            # print(num_diff)
            # single_ik_loss = single_ik_loss + num_diff


    # print(single_ik_loss, "\n", where_is_the_illegal, "\n", angle_solution)
    # make_dot(single_ik_loss).view()
    return single_ik_loss
    # return the_NANLOSS_of_illegal_solution_with_num_and_Nan

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
