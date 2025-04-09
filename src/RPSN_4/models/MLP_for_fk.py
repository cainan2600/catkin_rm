import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()
        self.norm_input = input_norm()
        # self.norm_output1 = output1_norm()
        self.norm_output2 = output2_norm()

        self.shared_layers = nn.Sequential(
            nn.Linear(num_i, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            # nn.Linear(2*num_h, 2*num_h),
            # # nn.LayerNorm(num_h),
            # nn.ReLU()
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

        out2 = self.output2_head(x)
        # out2 = self.norm_output2(out2)

        return out1, out2

class input_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):

        all_zero = torch.all(input_tensor == 0, dim=1)
        # 提取7x6的矩阵
        matrix_1 = input_tensor[~all_zero]
        matrix_0 = input_tensor[all_zero]

        # print(input_tensor, matrix_1, matrix_0)
        input_roll_pitch_yaw_max = torch.pi  # 角度范围 [-π, π]
        # input_coord_min_x = -1
        # input_coord_max_x = 1
        # input_coord_min_y = -0.425
        # input_coord_max_y = 0.425
        # input_coord_min_z = 0.13
        # input_coord_max_z = 0.15

        norm_angels = matrix_1[..., :3] / input_roll_pitch_yaw_max
        norm_x = matrix_1[..., 3] - 0.75
        norm_y = (matrix_1[..., 4] - 1.228) / 0.425
        norm_z = (matrix_1[..., 5] - 0.13) / 0.02
        # print(norm_z.size(), norm_z)
        norm_coords = torch.stack([norm_x, norm_y, norm_z], dim=1)

        # print(norm_coords)
        # print(torch.cat([norm_angels, norm_coords], dim=-1))
        norm_matrics_1 = torch.cat([norm_angels, norm_coords], dim=-1)
        # print(torch.cat([norm_matrics_1, matrix_0], dim=0))

        return torch.cat([norm_matrics_1, matrix_0], dim=0)

class output1_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output1_tensor):
        norm_x = output1_tensor[1] - 0.75



        return torch.cat([norm_matrics_1, matrix_0], dim=0)

class output2_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output2_tensor):

        # norm_x = output2_tensor / torch.pi
        # print(output2_tensor, norm_x)


        return output2_tensor / torch.pi
