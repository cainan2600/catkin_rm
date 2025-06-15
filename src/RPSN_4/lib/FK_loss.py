import torch
import torch.nn as nn
from torchviz import make_dot
import numpy as np

def calculate_FK_loss(angles, FK_results, input_target,intermediate_output):

    relu = nn.ReLU(inplace=True)
    num_Error1 = 0
    num_Error2 = 0
    num_NOError1 = 0
    num_NOError2 = 0
    fanwei = torch.tensor([torch.pi * 178/180, torch.pi * 130/180, torch.pi * 135/180, torch.pi * 178/180, torch.pi * 128/180, torch.pi])

    # print(angles)
    FK_loss = torch.tensor([0.0], requires_grad=True)
    FK_loss_11 = torch.tensor([0.0], requires_grad=True)
    FK_loss_22 = torch.tensor([0.0], requires_grad=True)
    FK_loss_33 = torch.tensor([0.0], requires_grad=True)

    FK_loss_11 = FK_loss_11 + torch.sum(relu(- fanwei - angles)) + torch.sum(relu(angles - fanwei))
    if FK_loss_11.detach().numpy() != 0:
        num_Error1 = num_Error1 + 1

    # 计算损失
    MSELoss = nn.MSELoss()
    # 网络输出的底盘位置和真实物品位置距离
    if np.sqrt(np.sum(np.square(intermediate_output.detach().numpy() - input_target[3:5].detach().numpy()))) > 0.74:
        num_Error2 = num_Error2 + 1
        # FK_loss = FK_loss + (MSELoss(intermediate_output , input_target[3:5])) *10

    FK_loss_22 = FK_loss_22 + relu(MSELoss(intermediate_output , input_target[3:5]) - torch.tensor([0.2738])) * torch.tensor([10]) # 1826


    # FK输出和真实的差距
    if np.sqrt(np.sum(np.square(FK_results[:3, 3].detach().numpy() - input_target[3:6].detach().numpy()))) > 0.1:
        num_NOError1 = num_NOError1 + 1
        # print('FK_results[:3, 3]', FK_results[:3, 3], FK_results)
        # print(input_target , input_target[3:6])

        # FK_loss = FK_loss + MSELoss(FK_results[:3, 3] , input_target[3:6])
        # print('2:', MSELoss(FK_results[:3, 3] , input_target[3:6]))
    else:
        num_NOError2 = num_NOError2 + 1

    FK_loss_33 = FK_loss_33 + relu(MSELoss(FK_results[:3, 3] , input_target[3:6]) - torch.tensor([0.0001]))
    # FK_loss_33 = FK_loss_33 + torch.sqrt(torch.sum(torch.square(FK_results[:3, 3] - input_target[3:6])))

    FK_loss = FK_loss + FK_loss_11 + FK_loss_22 + FK_loss_33

    # make_dot(FK_loss_33).view()

    return FK_loss, num_Error1, num_Error2, num_NOError1, num_NOError2, FK_loss_11, FK_loss_22, FK_loss_33


def calculate_FK_loss_test(angles, FK_results, input_target,intermediate_output):
    relu = nn.ReLU(inplace=True)
    fanwei = torch.tensor([torch.pi * 178/180, torch.pi * 130/180, torch.pi * 135/180, torch.pi * 178/180, torch.pi * 128/180, torch.pi])
    IK_loss_test_incorrect = 0
    IK_loss_test_correct = 0
    # print(angles, FK_results, input_target)
    # FK_loss_test = torch.tensor([0.0], requires_grad=True)
    # FK_loss_11_test = torch.tensor([0.0], requires_grad=True)
    # FK_loss_22_test = torch.tensor([0.0], requires_grad=True)
    # FK_loss_33_test = torch.tensor([0.0], requires_grad=True)


    FK_loss_11_test = torch.sum(relu(- fanwei - angles)) + torch.sum(relu(angles - fanwei))

    # MSELoss = nn.MSELoss()
    if FK_loss_11_test.detach().numpy() != 0 or np.sqrt(np.sum(np.square(intermediate_output.detach().numpy() - input_target[3:5].detach().numpy()))) > 0.74 or np.sqrt(np.sum(np.square(FK_results[:3, 3].detach().numpy() - input_target[3:6].detach().numpy()))) > 0.1:
        IK_loss_test_incorrect = IK_loss_test_incorrect + 1

    else:
        IK_loss_test_correct = IK_loss_test_correct + 1
    return FK_loss_11_test, IK_loss_test_incorrect, IK_loss_test_correct