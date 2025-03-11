import torch
import math 
import numpy as np


# 用于逆运算的 转置t
def transpose(x): # 取出每个——batch_size中的一个数据集的一个参数

    x = x[0]

    a = x[0][:3]
    b = x[1][:3]
    c = x[2][:3]
    result = torch.stack([a, b, c], 0)

    d = x[0][3]
    e = x[1][3]
    f = x[2][3]
    D = torch.stack([d, e, f], dim=0)
    D = D.unsqueeze(1)

    result_trans = torch.t(result)
    result_mul = torch.mm(-result_trans, D)

    T_Transpose0 = torch.cat([torch.t(result_trans), torch.t(result_mul)], 0)

    P = torch.tensor([0, 0, 0, 1])
    P = P.unsqueeze(0)

    T_Transpose = torch.cat([torch.t(T_Transpose0), P], 0)


    return T_Transpose

 
def rot2euler(R):
    assert (isRotationMatrix(R))
 
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
 
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1]) 
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# 输入三个欧拉角，以tensor形式运算出3×3旋转矩阵
def euler_to_rotMat(yaw, pitch, roll):
    ffff = torch.tensor(0)

    gggg = torch.tensor(1)

    Rz_yaw0 = torch.stack([torch.cos(yaw), -torch.sin(yaw), ffff], 0)
    Rz_yaw1 = torch.stack([torch.sin(yaw), torch.cos(yaw), ffff], 0)
    Rz_yaw2 = torch.stack([ffff, ffff, gggg], 0)
    Rz_yaw = torch.stack([Rz_yaw0, Rz_yaw1, Rz_yaw2], 0)

    Ry_pitch0 = torch.stack([torch.cos(pitch), ffff, torch.sin(pitch)], 0)
    Ry_pitch1 = torch.stack([ffff, gggg, ffff], 0)
    Ry_pitch2 = torch.stack([-torch.sin(pitch), ffff, torch.cos(pitch)], 0)
    Ry_pitch = torch.stack([Ry_pitch0, Ry_pitch1, Ry_pitch2], 0)

    Rx_roll0 = torch.stack([gggg, ffff, ffff], 0)
    Rx_roll1 = torch.stack([ffff, torch.cos(roll), -torch.sin(roll)], 0)
    Rx_roll2 = torch.stack([ffff, torch.sin(roll), torch.cos(roll)], 0)
    Rx_roll = torch.stack([Rx_roll0, Rx_roll1, Rx_roll2], 0)

    rotMat = torch.mm(Rz_yaw, torch.mm(Ry_pitch, Rx_roll))
    return rotMat




# 输入1×6tensor形式数据，数据前3个是欧拉角（转换为旋转矩阵），后三个是位置，输出是shaping后的4×4tensor齐次矩阵
def shaping(x):
    T_shapings = []
    for i in x: # 取出每个——batch_size中的一个数据集的一个参数
        a = i[0]
        b = i[1]
        c = i[2]
        result = euler_to_rotMat(c, b, a)
        # print(result)

        d = i[3]
        e = i[4]
        f = i[5]

        D = torch.stack([d, e, f], dim=0)
        D = D.unsqueeze(1) # 1x3变为3x1

        T_shaping0 = torch.cat([torch.t(result), torch.t(D)], 0)
        P = torch.tensor([0.0, 0.0, 0.0, 1.0])
        P = P.unsqueeze(0)

        T_shaping = torch.cat([torch.t(T_shaping0), P], 0)
        T_shaping = T_shaping.unsqueeze(0)
        T_shapings.append(T_shaping)

    T_shapings = torch.cat(T_shapings, dim=0)
    # print(T_shapings.size())
    return T_shapings



def rotation_matrix_to_quaternion(R):
    # 确保输入是3x3矩阵
    assert R.shape == (3, 3), "输入必须是3x3矩阵"
    
    # 计算四元数分量
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    
    # 构造四元数并归一化
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)  # 确保单位四元数
    
    # 约定w非负（可选）
    if q[0] < 0:
        q *= -1
    
    return q
