import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()
        # self.norm_input = input_norm()
        # self.norm_output1 = output1_norm()
        # self.norm_output2 = output2_norm()

        self.constrained_layer = ConstrainedOutputLayer(x_min=-0.55, x_max=2.05, y_min=0.503, y_max=1.953)

        self.shared_layers = nn.Sequential(
            nn.Linear(num_i, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            # nn.Linear(2*num_h, 4*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            # nn.Linear(4*num_h, 2*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            # nn.Linear(2*num_h, num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
        )

        # 输出1x3
        self.output1_head = nn.Sequential(
            # nn.Linear(2*num_h, num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            nn.Linear(num_h, num_o)
        )

        # 输出7x6
        self.output2_head = nn.Sequential(
            # nn.Linear(2*num_h, num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU(),
            nn.Linear(num_h, num_i)
        )

    def forward(self, input):
        # input = self.norm_input(input)

        x = self.shared_layers(input)

        out1 = self.output1_head(x)
        out1 = out1.mean(dim=0)
        # out1 = self.norm_output1(out1)
        out1 = self.constrained_layer(out1)

        out2 = self.output2_head(x)
        # out2 = self.norm_output2(out2)

        return out1, out2

# class input_norm(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input_tensor):

#         all_zero = torch.all(input_tensor == 0, dim=1)
#         # 提取7x6的矩阵
#         matrix_1 = input_tensor[~all_zero]
#         matrix_0 = input_tensor[all_zero]

#         # print(input_tensor, matrix_1, matrix_0)
#         input_roll_pitch_yaw_max = torch.pi  # 角度范围 [-π, π]
#         # input_coord_min_x = -1
#         # input_coord_max_x = 1
#         # input_coord_min_y = -0.425
#         # input_coord_max_y = 0.425
#         # input_coord_min_z = 0.13
#         # input_coord_max_z = 0.15

#         norm_angels = matrix_1[..., :3] / input_roll_pitch_yaw_max
#         # norm_x = matrix_1[..., 3] - 0.75
#         # norm_y = (matrix_1[..., 4] - 1.228) / 0.425
#         # norm_z = (matrix_1[..., 5] - 0.13) / 0.02

#         norm_x = (matrix_1[..., 3] - (-0.99)) * 2 / 3.48 - 1
#         norm_y = (matrix_1[..., 4] - 0.063) * 2 / 2.33 - 1
#         norm_z = (matrix_1[..., 5] - 0.13) * 2  / 0.02 - 1

#         # print(norm_z.size(), norm_z)
#         norm_coords = torch.stack([norm_x, norm_y, norm_z], dim=1)

#         # print(norm_coords)
#         # print(torch.cat([norm_angels, norm_coords], dim=-1))
#         norm_matrics_1 = torch.cat([norm_angels, norm_coords], dim=-1)
#         # print(torch.cat([norm_matrics_1, matrix_0], dim=0))

#         return torch.cat([norm_matrics_1, matrix_0], dim=0)

# class output1_norm(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, output1_tensor):
#         # print(output1_tensor.size())
#         norm_x = (output1_tensor[1] - (-0.99)) * 2 / 3.48 - 1
#         norm_y = (output1_tensor[2] - 0.063) * 2 / 2.33 - 1
#         norm_w = output1_tensor[0] / torch.pi

#         return torch.stack([torch.tanh(norm_w), torch.tanh(norm_x), torch.tanh(norm_y)], dim=-1)

# class output2_norm(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, output2_tensor):

#         return torch.tanh(output2_tensor / torch.pi)

class ConstrainedOutputLayer(torch.nn.Module):
    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
    def forward(self, raw_output):
        """
        raw_output: 形状为(batch_size, 7)
                    [prob_x_or_y, prob_right_or_left, x_offset, y_offset, x_free, y_free, yaw]
        """
        # 约束x还是y
        prob_x_or_y = torch.sigmoid(raw_output[0])
        choose_x_or_y = torch.bernoulli(prob_x_or_y) # 1:约束x轴，0:约束y轴
        # 大的一边还是小的
        prob_right_or_left = torch.sigmoid(raw_output[1])
        choose_right_or_left = torch.bernoulli(prob_right_or_left) # 1为大，0为小

        # 安全偏移量
        x_offset = torch.exp(raw_output[2])
        y_offset = torch.exp(raw_output[3])

        # 约束x轴时，x在左右外部，y自由
        x_constrained = (
            (self.x_min - x_offset) * (1 - choose_right_or_left) +
            (self.x_max + x_offset) * choose_right_or_left
            # self.x_max + x_offset
        )
        y_free = raw_output[5]

        # 约束y轴时，y在上下外部，x自由
        y_constrained = (
            (self.y_min - y_offset) * (1 - choose_right_or_left) +
            (self.y_max + y_offset) * choose_right_or_left
            # self.y_max + y_offset
        )
        x_free = raw_output[4]

        # 混合输出
        x = choose_x_or_y * x_constrained + (1 - choose_x_or_y) * x_free
        y = choose_x_or_y * y_free + (1 - choose_x_or_y) * y_constrained

        # print(x_constrained, y_constrained)
        # print(x_offset, y_offset)
        
        return torch.stack([raw_output[6], x, y], dim=-1)



















        # # 分离输入参数
        # x_dir_raw = raw_output[0]  # x方向选择（左/右）
        # x_dist_raw = raw_output[1] # x方向距离
        # y_dir_raw = raw_output[2]  # y方向选择（下/上）
        # y_dist_raw = raw_output[3] # y方向距离
        # yaw = raw_output[4]        # 底盘朝向
        
        # # 计算x坐标（左/右外部）
        # x_dir = torch.sigmoid(x_dir_raw)  # 0~1，趋近0选择左，趋近1选择右
        # x_dist = torch.exp(x_dist_raw)    # 确保距离为正
        # x = (self.x_min - x_dist) * (1 - x_dir) + (self.x_max + x_dist) * x_dir
        
        # # 计算y坐标（下/上外部）
        # y_dir = torch.sigmoid(y_dir_raw)  # 0~1，趋近0选择下，趋近1选择上
        # y_dist = torch.exp(y_dist_raw)    # 确保距离为正
        # y = (self.y_min - y_dist) * (1 - y_dir) + (self.y_max + y_dist) * y_dir

        # # x或y在桌子外时为true
        # x_violation = (x <= self.x_min) | (x >= self.x_max)
        # y_violation = (y <= self.y_min) | (y >= self.y_max)
        # valid_mask = x_violation | y_violation


        # print(x, y, valid_mask, -valid_mask.float())
        
        # return torch.stack([yaw, x, y], dim=-1)
