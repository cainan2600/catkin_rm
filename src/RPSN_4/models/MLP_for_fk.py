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

        # self.linear1 = torch.nn.Linear(num_i, num_h)
        # self.relu1 = torch.nn.ReLU()
        # # self.tanh1 = torch.nn.Tanh()
        # # self.leak_relu1 = torch.nn.PReLU()
        # self.linear2 = torch.nn.Linear(num_h, num_h)
        # self.relu2 = torch.nn.ReLU()
        # # self.tanh2 = torch.nn.Tanh()
        # # self.leak_relu2 = torch.nn.PReLU()
        # self.linear3 = torch.nn.Linear(num_h, num_o)



    def forward(self, input):

        x = self.shared_layers(input)

        out1 = self.output1_head(x)
        out1 = out1.mean(dim=0)
        
        out2 = self.output2_head(x)

        # x = self.linear1(input)
        # # x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        # x = self.linear3(x)
        # x = x.mean(dim=0)

        return out1, out2
