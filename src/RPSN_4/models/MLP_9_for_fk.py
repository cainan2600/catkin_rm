import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.adj = build_sysmmetric_adj()

        self.gc1 = GraphConvolution(num_i, num_h)
        self.relu1 = nn.ReLU()
        self.gc2 = GraphConvolution(num_h, num_h)
        self.relu2 = nn.ReLU()
        # self.gc3 = GraphConvolution(num_h, num_i)

        # 输出1x3
        self.output1_head = nn.Sequential(
            nn.Linear(num_h, num_o)
        )

        # 输出7x6
        self.output2_head = nn.Sequential(
            nn.Linear(num_h, num_i)
        )


    def forward(self, input):
        all_zero = torch.all(input == 0, dim=1)
        # 提取7x6的矩阵
        matrix = input[~all_zero]
        adj = self.adj(input)
        x = self.gc1(matrix, adj)
        x = self.relu1(x)
        x = self.gc2(x, adj)
        x = self.relu2(x)

        out1 = self.output1_head(x)
        out1 = out1.mean(dim=0)
        
        out2 = self.output2_head(x)

        return out1, out2


class GraphConvolution(nn.Module):
    def __init__(self, feature_num, hide_size):
        super(GraphConvolution, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(feature_num, hide_size))
        self.b = nn.Parameter(torch.FloatTensor(hide_size))

        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x = torch.mm(x, self.w)
        output = torch.spmm(adj, x)
        return output + self.b




class build_sysmmetric_adj(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        all_zero = torch.all(input_tensor == 0, dim=1)
        # 提取7x6的矩阵
        matrix = input_tensor[~all_zero]
        # print(matrix)

        # n = matrix.size(0)

        # # 判断每个物品是否非全零
        # is_non_zero = [not torch.all(row == 0) for row in matrix]
        # # print(is_non_zero)

        # # 创建邻接矩阵并指定为Float类型
        # adj_matrix = torch.zeros((n, n), dtype=torch.float32)  # 关键修改：使用float32类型

        # for i in range(n):
        #     for j in range(n):
        #         if is_non_zero[j]:
        #             adj_matrix[i, j] = 1.0  # 赋值浮点数
        #         if i == j:
        #             adj_matrix[i, j] = 1.0 if is_non_zero[i] else 0.0

        m = matrix.size(0)  # 获取输入的行数
        adj_matrix = torch.eye(m)       # 创建m×m单位矩阵
        adj_matrix[0, :] = 1            # 第一行全设为1
        adj_matrix[:, 0] = 1       



        # print("生成的邻接矩阵：")
        # print(adj_matrix)

        row_sums = adj_matrix.sum(dim=1)
        col_sums = adj_matrix.sum(dim=0)
        # print(row_sums, col_sums)
        
        # 处理零和情况（避免计算负数次幂出错）
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        col_sums = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums)
        
        # 计算-0.5次方
        row_inv_sqrt = row_sums.pow(-0.5)
        col_inv_sqrt = col_sums.pow(-0.5)
        
        # 创建对角矩阵
        D_row = torch.diag(row_inv_sqrt)
        D_col = torch.diag(col_inv_sqrt)
        # print("11", D_col,"22", D_row)

        adj = torch.mm(D_row, adj_matrix)
        adj = torch.mm(adj, D_col)
        # print(adj)

        return adj
