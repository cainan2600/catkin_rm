import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.mask = Masklayer()

        self.mask_encoder = nn.Embedding(2, num_i)

        # self.combined_f = DiffProjectionAttention(num_h)
        # self.attention = nn.MultiheadAttention(embed_dim=num_h, num_heads=num_heads, batch_first=False)
        # self.norm1 = nn.LayerNorm(num_h)

        self.global_mlp = nn.Sequential(
            nn.Linear(num_i, num_h),
            nn.LayerNorm(num_h),
            nn.Tanh()
        )
        # self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(

            nn.Linear(7*num_h, 112),
            nn.LayerNorm(112),
            nn.Tanh(),

            # nn.Linear(112, 56),
            # # nn.BatchNorm1d(56),
            # nn.Tanh(),

            # nn.Linear(56, 28),
            # # nn.BatchNorm1d(28),
            # nn.Tanh(),

            # nn.Linear(28, 14),
            # # nn.BatchNorm1d(14),
            # nn.Tanh(),

            nn.Linear(112, num_o)
        )



    def forward(self, input):

        mask = self.mask(input)
        mask_indices = mask.long()
        mask_feat = self.mask_encoder(mask_indices)

        combined = input + mask_feat

        # # 生成注意力掩码（阻止填充位置间的相互关注）
        # # attn_mask = mask.unsqueeze(0)
        # # key_padding_mask = (attn_mask == 0)
        # # print(attn_mask, key_padding_mask)
        # attn_mask = mask.unsqueeze(0).repeat(7, 1) 
        # attn_input = combined.unsqueeze(1) # 7,1,128

        # # attn_input_Q, attn_input_K, attn_input_V = self.combined_f(combined)
        # # attn_output, _ = self.attention(attn_input_Q, attn_input_K, attn_input_V, attn_mask=attn_mask) # 7,1,128
        # attn_output, _ = self.attention(attn_input, attn_input, attn_input, attn_mask=attn_mask) # 7,1,128
        # # print(attn_output.size())
        # attn_output = attn_output.squeeze(1) # 7,128
        # global_feature = combined + attn_output
        # global_feature = self.norm1(global_feature)

        fused_feature = self.global_mlp(combined) # 7,256
        # fused_feature = self.norm2(global_feature + fused_feature)

        # print(fused_feature.size())
        # fused_feature = self.flatten(fused_feature)
        fused_feature = fused_feature.view(-1)
        # print(fused_feature.size())

        x = self.regressor(fused_feature)
        return x




class Masklayer(nn.Module):
    def __init__(self, esp = 1e-6):
        super().__init__()
        self.esp = esp

    def forward(self, input_tensor):

        # mask = torch.ones(7)

        # for i in range(1, input_tensor.size(0)):
            # # 用前面的数据填充时
            # current_row = input_tensor[i]
            # previous_rows = input_tensor[:i]

            # is_duplicate = (previous_rows == current_row).all(dim=1).any()

            # if is_duplicate:
            #     mask[i] = 0
            # # 用0填充时
            # if input_tensor[i].all().any() == 0:
            #     mask[i] = 0

        mask = (input_tensor.abs().sum(dim=1) > self.esp).float()
        # print(mask)
        return mask

# class DynamicPositionEncoding(nn.Module):
#     def __init__(self, num_h):
#         super().__init__()
#         self.pe = nn.Parameter(torch.randn(7, num_h))  # 可学习位置编码
        
#     def forward(self, valid_mask):
#         return self.pe * valid_mask.unsqueeze(-1)  # 屏蔽无效位置

class DiffProjectionAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, feats):
        Q = self.query_proj(feats)  # 学习如何查询
        K = self.key_proj(feats)    # 学习如何被查询
        V = self.value_proj(feats)  # 学习如何贡献信息
        return Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1)

# class HybridAttention(nn.Module):
#     def __init__(self, num_i, num_h, num_o, num_heads):
#         super().__init__()
#         # 几何注意力分支
#         self.geo_attn = nn.Sequential(
#             nn.Linear(num_h, 2*num_h),
#             nn.GELU(),
#             nn.Linear(2*num_h, 1)
#         )
        
#         # 可学习注意力分支
#         self.learned_attn  = nn.MultiheadAttention(embed_dim=num_h, num_heads=num_heads, batch_first=True)
        
#     def forward(self, x):
#         # 几何重要性得分
#         geo_scores = self.geo_attn(x).squeeze(-1)  # (B,7)
        
#         # 可学习注意力
#         learned_attn = self.learned_attn(x)
        
#         # 混合注意力
#         combined = geo_scores.unsqueeze(-1) * learned_attn
#         return combined


    # def _initialize_weights(self):
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Linear):
    #             if hasattr(layer, 'activation') and layer.activation == 'relu':
    #                 nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    #                 if layer.bias is not None:
    #                     nn.init.constant_(layer.bias, 0)
    #                 # print("我HE初始化拉!!!!!!!!!", "{}".format(layer))
    #             else:
    #                 nn.init.xavier_uniform_(layer.weight)
    #                 if layer.bias is not None:
    #                     nn.init.constant_(layer.bias, 0)
    #                 # print("我XA初始化拉!!!!!!!!!", "{}".format(layer))