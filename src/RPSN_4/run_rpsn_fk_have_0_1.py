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
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
        self.parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=200, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=1000)
        self.parser.add_argument('--num_test', type=int, default=400)
        self.args = self.parser.parse_args()

        # 使用cuda!!!!!!!!!!!!!!!未补齐
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练集数据导入
        self.load_train_data = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/train-{}/train_dataset_{}.pt'.format(self.args.num_train, self.args.num_train))
        self.data_loader_train_dipan = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/train-{}/train_dataset_dipan_panduan_{}.pt'.format(self.args.num_train, self.args.num_train))
        # self.load_train_data = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-ik-all-random-with-dipan/train/train_dataset_5000.pt')
        # self.data_loader_train_dipan = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-ik-all-random-with-dipan/train/train_dataset_dipan_5000.pt')


        self.data_train = TensorDataset(self.load_train_data[:self.args.num_train], self.data_loader_train_dipan[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=True)

        # 测试集数据导入
        self.load_test_data = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/test-400/test_dataset_400.pt')
        self.load_test_dipan_data = torch.load('/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-have-0-1/test-400/test_dataset_dipan_panduan_1000.pt')
        self.data_test = TensorDataset(self.load_test_data[:self.args.num_test], self.load_test_dipan_data[:self.args.num_test])
        self.data_loader_test = DataLoader(self.data_test, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/catkin_rm/src/RPSN_4/work_dir/test000000000001-2'
        # 多少伦保存一次
        self.num_epoch_save = 100

        # 选择模型及参数
        self.num_i = 6
        self.num_h = 256
        self.num_o = 2
        self.model = MLP_for_0_1
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''

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

        echo_loss = []
        NUM_ALL_HAVE_SOLUTION = []
        NUM_ALL_HAVE_SOLUTION_test = []

        epochs = self.args.epochs
        data_loader_train = self.data_loader_train
        learning_rate = self.args.learning_rate
        model = self.model.MLP_self(num_i , num_h, num_o, num_heads) 
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.000)  # 定义优化器
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=6, min_lr=0.03)
        model_path = self.model_path

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

        # 开始训练
        for epoch in range(start_epoch , start_epoch + epochs):
  
            sum_loss = 0.0
            sum_loss_test = 0.0
            numError1 = 0
            numError2 = 0
            num_incorrect = 0
            num_correct = 0
            NUM_all_have_solution = 0
            NUM_all_have_solution_test = 0
            num_dipan_in_tabel = 0
            num_all_correct_but_dipan_in_tabel = 0

            for data in data_loader_train:  # 读入数据开始训练
                data, lables = data
                inputs_bxxx6 = data
                num_zu_in_epoch = 0

                # 计算 FK_loss_batch
                FK_loss_batch = torch.tensor(0.0, requires_grad=True)
                loss_fn = torch.nn.MSELoss()

                for inputs_xx6 in inputs_bxxx6:
                    # print(inputs_xx6)
                    num_zu_in_epoch += 1
                    # inputs = inputs_xx6
                    # 将7x6打乱并转换为1x42
                    inputs_xx6_no_random = inputs_xx6
                    inputs_xx6 = inputs_xx6[torch.randperm(inputs_xx6.size(0))]
                    # print(inputs_xx6_no_random, inputs_xx6_no_random.size())
                    # inputs = shaping_inputs_xx6_to_1xx(inputs_xx6)
                    inputs = inputs_xx6

                    intermediate_outputs_panduan = model(inputs)
                    FK_loss_batch = FK_loss_batch + loss_fn(intermediate_outputs_panduan, lables[num_zu_in_epoch - 1])
                    if loss_fn(intermediate_outputs_panduan, lables[num_zu_in_epoch - 1]) <= 0.01:
                        # print(intermediate_outputs_panduan)
                        NUM_all_have_solution += 1



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


            sum_loss_array = np.array(sum_loss)
            # echo_loss.append(sum_loss_array / len(data_loader_train))
            echo_loss.append(sum_loss_array)

            NUM_ALL_HAVE_SOLUTION.append(NUM_all_have_solution / self.args.num_train)

            print("correct", NUM_all_have_solution / self.args.num_train)


            model.eval()

            data_loader_test = self.data_loader_test

            for data_test in data_loader_test:
                data_test, lables_test = data_test
                num_zu_in_epoch_test = 0
                with torch.no_grad():
                    inputs_bxxx6_test = data_test
                    # print("data_test", data_test, "data_test[0]", inputs_bxxx6_test)
                    for inputs_xx6_test in inputs_bxxx6_test:
                        num_zu_in_epoch_test += 1
                        inputs_xx6_test = inputs_xx6_test[torch.randperm(inputs_xx6_test.size(0))]
                        inputs_test = inputs_xx6_test
                        intermediate_outputs_panduan_test = model(inputs_test)
                        if loss_fn(intermediate_outputs_panduan_test, lables_test[num_zu_in_epoch_test - 1]) <= 0.01:
                            NUM_all_have_solution_test += 1


            print("correct_test", NUM_all_have_solution_test / self.args.num_test)
            NUM_ALL_HAVE_SOLUTION_test.append(NUM_all_have_solution_test / self.args.num_test)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")

            print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss), "-" * 100)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        

        # 画图
        plot_train_loss(self.checkpoint_dir, start_epoch, epochs, echo_loss)
        # plot_train_fk(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUMError1, NUMError2, NUM_incorrect, NUM_correct)
        # plot_test_fk(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUM_incorrect_test, NUM_correct_test)
        plot_no_not_have_solution(self.checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION)
        plot_no_not_have_solution_test(self.checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION_test)
        # plot_correct_but_dipan_in_tabel(self.checkpoint_dir, start_epoch, epochs, NUM_all_correct_but_dipan_in_tabel)

if __name__ == "__main__":
    a = main()
    a.train()
