import torch

def cos(a):
    return torch.cos(a)

def sin(a):
    return torch.sin(a)

def THT(Theta, A, D, Alpha):
    # T = torch.tensor([
    #     [cos(Theta), -sin(Theta)*cos(Alpha), sin(Alpha)*sin(Theta), A*cos(Theta)],
    #     [sin(Theta), cos(Theta)*cos(Alpha), -cos(Theta)*sin(Alpha), A*sin(Theta)],
    #     [0, sin(Alpha), cos(Alpha), D],
    #     [0, 0, 0, 1]
    # ])
    T = torch.stack([
        torch.stack([torch.cos(Theta), -torch.sin(Theta) * torch.cos(Alpha), torch.sin(Alpha) * torch.sin(Theta),
                     A * torch.cos(Theta)], dim=-1),
        torch.stack([torch.sin(Theta), torch.cos(Theta) * torch.cos(Alpha), -torch.cos(Theta) * torch.sin(Alpha),
                     A * torch.sin(Theta)], dim=-1),
        torch.stack([torch.tensor(0.0), torch.sin(Alpha), torch.cos(Alpha), D], dim=-1),
        torch.tensor([0.0, 0.0, 0.0, 1.0])
    ])
    return T




def THT2(Theta, A, D, Alpha):
    # T = torch.tensor([
    #     [cos(Theta), -sin(Theta)*cos(Alpha), sin(Alpha)*sin(Theta), A*cos(Theta)],
    #     [sin(Theta), cos(Theta)*cos(Alpha), -cos(Theta)*sin(Alpha), A*sin(Theta)],
    #     [0, sin(Alpha), cos(Alpha), D+torch.tensor([0.5])],
    #     [0, 0, 0, 1]
    # ])
    T = torch.stack([
        torch.stack([torch.cos(Theta), -torch.sin(Theta) * torch.cos(Alpha), torch.sin(Alpha) * torch.sin(Theta),
                     A * torch.cos(Theta)], dim=-1),
        torch.stack([torch.sin(Theta), torch.cos(Theta) * torch.cos(Alpha), -torch.cos(Theta) * torch.sin(Alpha),
                     A * torch.sin(Theta)], dim=-1),
        torch.stack([torch.tensor(0.0), torch.sin(Alpha), torch.cos(Alpha), D+torch.tensor(0.5)], dim=-1),
        torch.tensor([0.0, 0.0, 0.0, 1.0])
    ])
    return T

def FK(theta, base, a, d, alpha):

    T01 = THT(theta[0], a[0], d[0], alpha[0])
    T12 = THT(theta[1], a[1], d[1], alpha[1])
    T23 = THT(theta[2], a[2], d[2], alpha[2])
    T34 = THT(theta[3], a[3], d[3], alpha[3])
    T45 = THT(theta[4], a[4], d[4], alpha[4])
    T56 = THT(theta[5], a[5], d[5], alpha[5])
    # print(base, T01)
    T0 = torch.mm(base, T01)
    T1 = torch.mm(T0, T12)
    T2 = torch.mm(T1, T23)
    T3 = torch.mm(T2, T34)
    T4 = torch.mm(T3, T45)
    T5 = torch.mm(T4, T56)

    return T5
    # print('T56', T56)
    # return [T01, T12, T23, T34, T45, T56]
    # return [T0, T1, T2, T3, T4, T5]
