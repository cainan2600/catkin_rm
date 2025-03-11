import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.mask = Masklayer()

        self.attention1 = nn.MultiheadAttention(embed_dim=num_i, num_heads=num_heads, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(num_features=num_i)

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.linear1.activation = 'relu'
        self.relu1 = torch.nn.ReLU()
        # self.tanh1 = torch.nn.Tanh()
        # self.leak_relu1 = torch.nn.PReLU()

        # MLP 层
        self.linear2 = torch.nn.Linear(num_h, num_h)
        self.linear2.activation = 'relu'
        self.relu2 = torch.nn.ReLU()
        # self.tanh2 = torch.nn.Tanh()
        # self.leak_relu2 = torch.nn.PReLU()

        self.linear3 = torch.nn.Linear(num_h, num_o)
        # self.dropout = torch.nn.Dropout(0.75)

        self._initialize_weights()



    def forward(self, input):

        # input = self.batch_norm(input)
        # attn_output, _ = self.attention1(input, input, input)
        # input = input + attn_output


        x = self.mask(input)

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        x = x.mean(dim=0)

        return x

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'activation') and layer.activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                    # print("我HE初始化拉!!!!!!!!!", "{}".format(layer))
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                    # print("我XA初始化拉!!!!!!!!!", "{}".format(layer))


class Masklayer(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):
        # 初始化全1的掩码（保留所有数据）
        mask = torch.ones_like(input_tensor)
        # 遍历每一行（从第二行开始检查）
        for i in range(1, input_tensor.size(0)):
            # 将当前行与之前所有行比较
            current_row = input_tensor[i]
            previous_rows = input_tensor[:i]
            # 计算重复性（精确匹配）
            is_duplicate = (previous_rows == current_row).all(dim=1).any()
            # 如果是重复行则将掩码置零
            if is_duplicate:
                mask[i] = 0
        # print(mask)
        return input_tensor * mask