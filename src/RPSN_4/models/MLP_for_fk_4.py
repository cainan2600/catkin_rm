import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.constrained_layer = ConstrainedOutputLayer(x_min=-0.55, x_max=2.05, y_min=0.503, y_max=1.953)

        # 输出1x3
        self.output1_head = nn.Sequential(
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
            nn.Linear(num_h, num_o)
        )

        # 输出7x6
        self.output2_head = nn.Sequential(
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
            nn.Linear(num_h, num_i)
        )

    def forward(self, input):

        out1 = self.output1_head(input)
        out1 = out1.mean(dim=0)
        out1 = self.constrained_layer(out1)

        out2 = self.output2_head(input)

        return out1, out2

class ConstrainedOutputLayer(torch.nn.Module):
    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
    def forward(self, raw_output):
        """
        raw_output: 形状为(7)
                    [prob_x, prob_y, prob_big, prob_small, x_offset, y_offset, yaw]
        """
        # # 约束x还是y
        # prob_x_or_y = torch.sigmoid(raw_output[0])
        # choose_x_or_y = torch.bernoulli(prob_x_or_y) # 1:约束x轴，0:约束y轴
        # # 大的一边还是小的
        # prob_right_or_left = torch.sigmoid(raw_output[1])
        # choose_right_or_left = torch.bernoulli(prob_right_or_left) # 1为大，0为小

        choose_x = F.gumbel_softmax(raw_output[0], tau=1.0, hard=not self.training)
        choose_big = F.gumbel_softmax(raw_output[2], tau=1.0, hard=not self.training)
        choose_y = F.gumbel_softmax(raw_output[1], tau=1.0, hard=not self.training)
        choose_small = F.gumbel_softmax(raw_output[3], tau=1.0, hard=not self.training)

        # 安全偏移量
        x_offset = torch.exp(raw_output[4])
        y_offset = torch.exp(raw_output[5])

        # 约束x轴时，x在左右外部，y自由
        x_constrained = (
            (self.x_min - x_offset) * choose_small +
            (self.x_max + x_offset) * choose_big
        )
        y_free = self.y_min - y_offset + (self.y_max - self.y_min + 2*y_offset) * torch.sigmoid(raw_output[5])

        # 约束y轴时，y在上下外部，x自由
        y_constrained = (
            (self.y_min - y_offset) * choose_small +
            (self.y_max + y_offset) * choose_big
        )
        x_free = self.x_min - x_offset + (self.x_max - self.x_min + 2*x_offset) * torch.sigmoid(raw_output[4])

        # 混合输出
        x = choose_x * x_constrained + choose_y * x_free
        y = choose_x * y_free + choose_y * y_constrained

        # print(x_constrained, y_constrained)
        # print(x_offset, y_offset)
        # print(choose_right_or_left, choose_x_or_y)
        # print(raw_output[2], raw_output[3])
        
        return torch.stack([raw_output[6], x, y], dim=-1)