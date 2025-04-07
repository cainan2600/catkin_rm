import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(num_i, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_h),
            # nn.LayerNorm(num_h),
            nn.ReLU(),
            # nn.Linear(2*num_h, num_h),
            # nn.ReLU()
        )

        # 输出1x3
        self.output1_head = nn.Sequential(
            nn.Linear(num_h, num_o)
        )

        # 输出7x6
        self.output2_head = nn.Sequential(
            nn.Linear(num_h, num_i)
        )

    def forward(self, input):

        x = self.shared_layers(input)

        out1 = self.output1_head(x)
        out1 = out1.mean(dim=0)

        out2 = self.output2_head(x)

        return out1, out2
