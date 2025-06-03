import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        # 输出1x3
        self.output1_head = nn.Sequential(
            nn.Linear(num_i, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, 2*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(2*num_h, 4*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            # nn.Linear(4*num_h, 8*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            # nn.Linear(8*num_h, 10*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            # nn.Linear(10*num_h, 8*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            # nn.Linear(8*num_h, 4*num_h),
            # nn.LayerNorm(num_h),
            # nn.ReLU(),
            nn.Linear(4*num_h, 2*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(2*num_h, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_o)
        )

        # 输出7x6
        self.output2_head = nn.Sequential(
            nn.Linear(num_i, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, 2*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(2*num_h, 4*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(4*num_h, 8*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            # nn.Linear(8*num_h, 10*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            # nn.Linear(10*num_h, 8*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            nn.Linear(8*num_h, 4*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(4*num_h, 2*num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(2*num_h, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_i)
        )

        # self.xyz = PhysicsProjectionLayer()



    def forward(self, input):

        out1 = self.output1_head(input)
        out1 = out1.mean(dim=0)
        # out1 = torch.sigmoid(out1)
        # out1 = self.xyz(out1)

        out2 = self.output2_head(input)

        return out1, out2


# class PhysicsProjectionLayer(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         # 输入x的形状：[batch_size, 2]，两个参数：rho_raw, theta_raw
#         # print(x)
#         # 约束1：半径必须≥0.44m
#         rho = x[0] * torch.tensor(0.05) + torch.tensor(0.79)  # 映射到0.44~0.64m范围
        
#         # 约束2：角度归一化到0~2π
#         theta = x[1] * torch.pi + torch.pi # 映射到0~360°
        
#         # 转换为笛卡尔坐标
#         x_out = rho * torch.cos(theta)
#         y_out = rho * torch.sin(theta)
        
#         # 约束3：航向角与位置方向一致
#         yaw_out = theta  # 让机器人朝向桌子中心
#         # print([yaw_out, x_out, y_out])
        
#         return torch.stack([yaw_out, x_out, y_out], dim=0)