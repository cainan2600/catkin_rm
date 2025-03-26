import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()


        self.encoder = nn.Sequential(
            nn.Linear(7*6, 256),  # 展平7个物体的6维位姿
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # x,y,yaw
        )
        
    def forward(self, x):
        x = x.view(-1, 7*6)  # 展平输入
        x = self.encoder(x)
        x = self.regressor(x)
        # print(x.size())
        return x.squeeze()