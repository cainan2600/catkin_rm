import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models import MLP_for_fk, MLP_9_for_fk, MLP_for_0_1
from lib.trans_all import *
from lib import FK_diff, FK_loss
import torch
import torch.nn as nn
import math
import os
from lib.save import checkpoints
from lib.plot import *
from data.data_generate_fk_ik import save_data, save_MLP_output

from torch.optim.lr_scheduler import ReduceLROnPlateau


class main():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training MLP")
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 1)')
        self.parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=200, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=1000)
        self.parser.add_argument('--num_test', type=int, default=400)
        self.args = self.parser.parse_args()

        # 使用cuda!!!!!!!!!!!!!!!未补齐
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练集数据导入
        self.load_train_data = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/train-{}-1/train_dataset_{}.pt'.format(self.args.num_train, self.args.num_train))
        self.data_loader_train_dipan = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/train-{}-1/train_dataset_dipan_panduan_{}.pt'.format(self.args.num_train, self.args.num_train))
        # self.load_train_data = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-ik-all-random-with-dipan/train/train_dataset_5000.pt')
        # self.data_loader_train_dipan = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-ik-all-random-with-dipan/train/train_dataset_dipan_5000.pt')


        self.data_train = TensorDataset(self.load_train_data[:self.args.num_train], self.data_loader_train_dipan[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=False)

        # 测试集数据导入
        self.load_test_data = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/test-400-1/test_dataset_400.pt')
        self.data_test = TensorDataset(self.load_test_data[:self.args.num_test])
        self.data_loader_test = DataLoader(self.data_test, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/catkin_rm/src/RPSN_4/work_dir/test02-1-output5-1111111obj-11111111111111111111'
        # 多少伦保存一次
        self.num_epoch_save = 100

        # 选择模型及参数
        self.num_i = 6
        self.num_h = 128
        self.num_o = 5
        self.num_o_0_1 = 2
        self.model = MLP_for_fk
        self.model_0_1 = MLP_for_0_1
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''
        self.model_0_1_path = r'/home/cn/catkin_rm/src/RPSN_4/work_dir/test000000000001-1/checkpoint-epoch200.pt'

        # 定义DH参数
        # self.link_length = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
        # self.link_offset = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
        # self.link_twist = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])
        self.link_length = torch.tensor([0, 0, 0.256, 0, 0, 0])
        self.link_offset = torch.tensor([0.2405, 0, 0, 0.210, 0, 0.274])
        self.link_twist = torch.FloatTensor([0, math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2])

    def train(self):
        num_i = self.num_i
        num_h = self.num_h
        num_o = self.num_o
        num_heads = 4

        NUMError1 = []
        NUMError2 = []
        NUM_incorrect = []
        NUM_correct = []
        NUM_correct_test = []
        NUM_incorrect_test = []
        echo_loss = []
        echo_loss_test = []
        erro_inputs = []
        no_erro_inputs = []
        # NUM_dipan_in_tabel = []
        NET_output = []

        NUM_all_correct_but_dipan_in_tabel = []
        NUM_all_correct_and_not_in_tabel = []
        NUM_not_all_correct = []
        NUM_some_incorrect_of_correct = []
        
        NUM_ALL_HAVE_SOLUTION = []
        NUM_ALL_HAVE_SOLUTION_test = []

        CORRECT_obj = []
        CORRECT_chasis = []
        INCORRECT_obj = []
        INCORRECT_chasis = []
        save_data_correct_incorrect = []
        SAVE_DATA_CORRECT_INCORRECT = []

        epochs = self.args.epochs
        data_loader_train = self.data_loader_train
        learning_rate = self.args.learning_rate
        model = self.model.MLP_self(num_i , num_h, num_o, num_heads)
        model_0_1 = self.model_0_1.MLP_self(num_i, 2*num_h, self.num_o_0_1, num_heads)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.000)  # 定义优化器
        optimizer_0_1 = torch.optim.Adagrad(model_0_1.parameters(), lr=0.003, weight_decay=0.000)  # 定义优化器
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=6, min_lr=0.03)
        model_path = self.model_path
        model_0_1_path = self.model_0_1_path

        if os.path.exists(model_path):          
            checkpoint = torch.load(model_path)  
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            print('-' * 100 + '\n' + f"The loading model is complete, let's start this training from the {start_epoch} epoch, the current loss is : {loss}" + '\n' + '-' * 100)
        else:
            print('-' * 120 + '\n' + "There is no pre-trained model under the path, and the following training starts from [epoch1] after random initialization" + '\n' + '-' * 120)
            start_epoch = 1

        if os.path.exists(model_0_1_path):          
            checkpoint_0_1 = torch.load(model_0_1_path)  
            model_0_1.load_state_dict(checkpoint_0_1['state_dict'])
            optimizer_0_1.load_state_dict(checkpoint_0_1['optimizer'])

        # 开始训练
        for epoch in range(start_epoch , start_epoch + epochs):
  
            sum_loss = 0.0
            sum_loss_test = 0.0
            numError1 = 0
            numError2 = 0
            num_incorrect = 0
            num_correct = 0
            NUM_all_have_solution = 0
            num_dipan_in_tabel = 0
            num_all_correct_but_dipan_in_tabel = 0

            for data in data_loader_train:  # 读入数据开始训练
                data, lables = data
                inputs_bxxx6 = data
                # print("11111",data)
                # 将batch_size中的每一组数据输入网络
                num_zu_in_epoch = 0

                # 计算 FK_loss_batch
                FK_loss_batch = torch.tensor(0.0, requires_grad=True)
                IK_loss2 = torch.tensor(0.0, requires_grad=True)
                # IK_loss3 = torch.tensor(0.0, requires_grad=True)
                loss_fn = torch.nn.MSELoss()

                for inputs_xx6 in inputs_bxxx6:
                    # print(inputs_xx6)
                    num_zu_in_epoch += 1
                    # inputs = inputs_xx6
                    # 将7x6打乱并转换为1x42
                    inputs_xx6_no_random = inputs_xx6
                    # inputs_xx6 = inputs_xx6[torch.randperm(inputs_xx6.size(0))]
                    # print(inputs_xx6_no_random, inputs_xx6_no_random.size())
                    # inputs = shaping_inputs_xx6_to_1xx(inputs_xx6)
                    inputs = inputs_xx6
                    # xxxxxx = model_0_1(inputs)
                    xxxxxx = lables[num_zu_in_epoch - 1]
                    # print(xxxxxx)

                    # print(xxxxxx, inputs)
                    intermediate_outputs_chasis, intermediate_outputs_angel = model(inputs, xxxxxx)
                    # print(intermediate_outputs_chasis.size(), intermediate_outputs_chasis)

                    # # 推出归一化
                    # intermediate_outputs_chasis = (intermediate_outputs_chasis + 1) / 2
                    # intermediate_outputs_chasis_x = intermediate_outputs_chasis[1] * 3.48 + (-0.99)
                    # intermediate_outputs_chasis_y = intermediate_outputs_chasis[2] * 2.33 + 0.063
                    # intermediate_outputs_chasis_w = intermediate_outputs_chasis[0] * torch.pi
                    # intermediate_outputs_chasis = torch.stack([intermediate_outputs_chasis_w, intermediate_outputs_chasis_x, intermediate_outputs_chasis_y], dim=-1)

                    # intermediate_outputs_angel = intermediate_outputs_angel * torch.pi

                    intermediate_outputs_list = intermediate_outputs_chasis.detach().numpy()
                    # print(intermediate_outputs, intermediate_outputs_list)
                    if epoch == (start_epoch + epochs - 1):
                        NET_output.append(intermediate_outputs_list)
                    # print(intermediate_outputs_list)

                    # 得到每个1x6的旋转矩阵(7x6)
                    # print(inputs_xx6.size())
                    input_tar = shaping(inputs_xx6)
                    # last_reverse = [
                    #     [1,0,0,0],
                    #     [0,1,0,0],
                    #     [0,0,1,-0.149],
                    #     [0,0,0,1]
                    # ]
                    # input_tar = input_tar * last_reverse
                
                    # 将网络输出1x21转换为7x3
                    # intermediate_outputs = shaping_outputs_1xx_to_xx3(intermediate_outputs, num_i)

                    outputs = torch.empty((0, 6)) # 创建空张量
                    # for each_result in intermediate_outputs: # 取出每个batch_size中的每个数据经过网络后的结果1x3
                    pinjie1 = torch.cat([intermediate_outputs_chasis, torch.zeros(1).detach()])
                    pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                    outputs = torch.cat([outputs, pinjie2.unsqueeze(0)], dim=0)

                    outputs_tensor = outputs[0]
                    # print(outputs.size(), outputs_tensor.size())

                    intermediate_outputs_chasis.retain_grad()
                    outputs.retain_grad()

                    MLP_output_base = shaping(outputs)  # 对输出做shaping运算-1X6变为4X4
                    MLP_output_base.retain_grad()

                    NUM_obj = 0
                    NUM_correct_obj = 0
                    save_end_eff_calcu_by_FK = []
                    for i in range(len(input_tar)):
                        if torch.all(inputs_xx6[i].ne(0)):

                            NUM_obj += 1

                            end_eff_calcu_by_FK = FK_diff.FK(
                                intermediate_outputs_angel[i], 
                                MLP_output_base[0], 
                                self.link_length, 
                                self.link_offset, 
                                self.link_twist)

                            # print(end_eff_calcu_by_FK, end_eff_calcu_by_FK[:3, 3])
                            save_end_eff_calcu_by_FK.append(end_eff_calcu_by_FK[:3, 3])

                            # 计算单IK_loss
                            FK_loss_batch, num_Error1, num_Error2, num_NOError1, num_NOError2= FK_loss.calculate_FK_loss(
                                intermediate_outputs_angel[i], 
                                end_eff_calcu_by_FK, 
                                inputs[i], 
                                intermediate_outputs_chasis[1:3])
                            # make_dot(IK_loss1).view()

                            # 总loss
                            FK_loss_batch = FK_loss_batch + FK_loss_batch

                            numError1 = numError1 + num_Error1
                            numError2 = numError2 + num_Error2
                            num_incorrect = num_incorrect + num_NOError1
                            num_correct = num_correct + num_NOError2

                            if not num_NOError2==0:
                                NUM_correct_obj += 1

                    # 打印
                    list_0 = torch.tensor([0, 0, 0])
                    num_data = len(save_end_eff_calcu_by_FK)
                    while num_data < 7:
                        # 用0填充
                        element = list_0
                        save_end_eff_calcu_by_FK.append(element)
                        num_data += 1

                    if NUM_correct_obj == NUM_obj:
                        NUM_all_have_solution += 1
                        if -0.55<intermediate_outputs_list[1]<2.05:
                            if 0.503<intermediate_outputs_list[2]<1.953:
                                num_all_correct_but_dipan_in_tabel += 1
                    
                    else:
                        if epoch == start_epoch + epochs - 1:
                            for save_dddd in inputs_xx6_no_random.detach().numpy():
                                CORRECT_obj.append(save_dddd)
                            CORRECT_chasis.append(lables[num_zu_in_epoch - 1].detach().numpy())
                            
                            INCORRECT_chasis.append(outputs_tensor.detach().numpy())
                            for save_data_incorrect in save_end_eff_calcu_by_FK:
                                INCORRECT_obj.append(save_data_incorrect.detach().numpy())

                    # FK_loss_batch = FK_loss_batch + \
                    # min(max(0, outputs_tensor[3] - (-0.55)), max(0, 2.05 - outputs_tensor[3])) + \
                    # min(max(0, outputs_tensor[4] - 0.503), max(0, 1.953 - outputs_tensor[4]))
                    

                    # FK_loss_batch = FK_loss_batch + min(
                    #     min(max(0, outputs_tensor[3] - (-0.55)), max(0, 2.05 - outputs_tensor[3])),
                    #     min(max(0, outputs_tensor[4] - 0.503), max(0, 1.953 - outputs_tensor[4]))
                    #     )

                    # FK_loss_batch = FK_loss_batch + \
                    #     min(torch.relu(outputs_tensor[3] - (-0.55)), torch.relu(2.05 - outputs_tensor[3])) * \
                    #     min(torch.relu(outputs_tensor[4] - 0.503), torch.relu(1.953 - outputs_tensor[4]))

                    # x_low_penalty = torch.relu(outputs_tensor[3] - (-0.55))  # x < -0.55时为正
                    # x_high_penalty = torch.relu(2.05 - outputs_tensor[3])  # x >2.05时为正
                    # y_low_penalty = torch.relu(outputs_tensor[4] - 0.503)
                    # y_high_penalty = torch.relu(1.953 - outputs_tensor[4])
                    # FK_loss_batch = FK_loss_batch + (x_low_penalty + x_high_penalty + y_low_penalty + y_high_penalty).mean()

                    # FK_loss_batch = FK_loss_batch + loss_fn(outputs_tensor, lables[num_zu_in_epoch - 1])


                FK_loss_batch.retain_grad()
                # make_dot(FK_loss_batch).view()

                optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

                # 定义总loss函数
                # loss = FK_loss_batch / len(inputs_bxxx6)
                loss = FK_loss_batch
                loss.retain_grad()

                # assert torch.isnan(loss).sum() == 0

                # 绘制计算图
                # make_dot(loss).view()

                loss.backward()  # 反向传播求梯度

                # for name, weight in model.named_parameters():
                #     # print("weight:", weight) # 打印权重，看是否在变化
                #     if weight.requires_grad:
                #         # print("weight:", weight.grad) # 打印梯度，看是否丢失
                #         # 直接打印梯度会出现太多输出，可以选择打印梯度的均值、极值，但如果梯度为None会报错
                #         print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())


                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)  # 进行梯度裁剪
                optimizer.step()  # 更新所有梯度
                sum_loss = sum_loss + loss.data

            # 记录x轮以后网络模型checkpoint，用来查看数据流
            if epoch % self.num_epoch_save == 0:
                # print("第{}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等".format(epoch))
                checkpoints(model, epoch, optimizer, loss, self.checkpoint_dir)

            accuracy = NUM_all_have_solution / self.args.num_train
            scheduler.step(accuracy)
            sum_loss_array = np.array(sum_loss)
            # echo_loss.append(sum_loss_array / len(data_loader_train))
            echo_loss.append(sum_loss_array)
            
            NUMError1.append(numError1)
            NUMError2.append(numError2)
            NUM_incorrect.append(num_incorrect)
            NUM_correct.append(num_correct)
            NUM_ALL_HAVE_SOLUTION.append(NUM_all_have_solution / self.args.num_train)
            NUM_all_correct_but_dipan_in_tabel.append(num_all_correct_but_dipan_in_tabel)

            print("numError1", numError1)
            print("numError2", numError2)
            print("num_correct", num_correct)
            print("num_incorrect", num_incorrect)
            print("NUM_all_have_solution", NUM_all_have_solution)
            print("num_all_correct_but_dipan_in_tabel", num_all_correct_but_dipan_in_tabel)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")


            model.eval()

            data_loader_test = self.data_loader_test
            num_incorrect_test = 0
            num_correct_test = 0
            NUM_all_have_solution_test = 0

            for data_test in data_loader_test:
                with torch.no_grad():
                    inputs_bxxx6_test = data_test[0]
                    # print("data_test", data_test, "data_test[0]", inputs_bxxx6_test)
                    for inputs_xx6_test in inputs_bxxx6_test:
                        # inputs_xx6_test = inputs_xx6_test[torch.randperm(inputs_xx6_test.size(0))]
                        inputs_test = inputs_xx6_test
                        xxxxxx_test = model_0_1(inputs_test)
                        intermediate_outputs_chasis_test, intermediate_outputs_angel_tese = model(inputs_test, xxxxxx_test)
                        # print(inputs_xx6_test.size())
                        input_tar_test = shaping(inputs_xx6_test)
                        outputs_test = torch.empty((0, 6))
                        pinjie1 = torch.cat([intermediate_outputs_chasis_test, torch.zeros(1).detach()])
                        pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                        outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

                        MLP_output_base_test = shaping(outputs_test)

                        # 计算 FK_loss_batch
                        FK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
                        NUM_obj_test = 0
                        NUM_correct_obj_test = 0                      
                        for i in range(len(input_tar_test)):
                            if torch.all(inputs_xx6_test[i].ne(0)):                          
                                NUM_obj_test += 1
                                end_eff_calcu_by_FK_test = FK_diff.FK(
                                    intermediate_outputs_angel_tese[i], 
                                    MLP_output_base_test[0], 
                                    self.link_length, 
                                    self.link_offset, 
                                    self.link_twist)

                                # 计算单IK_loss
                                FK_loss_batch, IK_loss_test_incorrect, IK_loss_test_correct= FK_loss.calculate_FK_loss_test(
                                    intermediate_outputs_angel_tese[i], 
                                    end_eff_calcu_by_FK_test, 
                                    inputs_test[i], 
                                    intermediate_outputs_chasis_test[1:3])
                                # 计算IK_loss时存在的错误与正确的打印

                                num_incorrect_test = num_incorrect_test + IK_loss_test_incorrect
                                num_correct_test = num_correct_test + IK_loss_test_correct
                                if not IK_loss_test_correct==0:
                                    NUM_correct_obj_test += 1
                        if NUM_correct_obj_test== NUM_obj_test:
                            NUM_all_have_solution_test += 1
              

            print("num_correct_test", num_correct_test)
            print("num_incorrect_test", num_incorrect_test)
            print("NUM_all_have_solution_test", NUM_all_have_solution_test)

            NUM_incorrect_test.append(num_incorrect_test)
            NUM_correct_test.append(num_correct_test)
            NUM_ALL_HAVE_SOLUTION_test.append(NUM_all_have_solution_test / self.args.num_test)


            print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss), "-" * 100)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # 分开保存最后一轮中错误和正确的数据
        # save_data(no_erro_inputs, self.checkpoint_dir, "save_no_erro_data.txt")
        # save_data(erro_inputs, self.checkpoint_dir, "save_erro_data.txt")
        save_MLP_output(NET_output, self.checkpoint_dir, "NET_output.txt")

        save_MLP_output(INCORRECT_obj, self.checkpoint_dir, "INCORRECT_obj.txt")
        save_MLP_output(INCORRECT_chasis, self.checkpoint_dir, "INCORRECT_chasis.txt")
        save_MLP_output(CORRECT_obj, self.checkpoint_dir, "CORRECT_obj.txt")
        save_MLP_output(CORRECT_chasis, self.checkpoint_dir, "CORRECT_chasis.txt")
        

        # 画图
        plot_train_loss(self.checkpoint_dir, start_epoch, epochs, echo_loss)
        plot_train_fk(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUMError1, NUMError2, NUM_incorrect, NUM_correct)
        plot_test_fk(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUM_incorrect_test, NUM_correct_test)
        plot_no_not_have_solution(self.checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION)
        plot_no_not_have_solution_test(self.checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION_test)
        plot_correct_but_dipan_in_tabel(self.checkpoint_dir, start_epoch, epochs, NUM_all_correct_but_dipan_in_tabel)

if __name__ == "__main__":
    a = main()
    a.train()
