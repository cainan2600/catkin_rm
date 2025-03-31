import torch
from lib.trans_all import *
from torchviz import make_dot
import random


grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

# 输入两个4×4tensor（世界坐标系下目标位置、世界坐标系下底盘位置）
def calculate_IK(input_tar, MLP_output_base, a, d, alpha):

    fanwei1 = torch.tensor([math.pi * 178/180, math.pi * 130/180, math.pi * 135/180, math.pi * 178/180, math.pi * 128/180, math.pi])

    num_Error1 = 0
    num_Error2 = 0
    num_Error3 = 0
    EPSILON = 1e-7

    save_what_caused_Error2_as_Nan = []
    the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0], requires_grad=True)
    the_loss_of_over = torch.tensor([0.0], requires_grad=True)

    TT = torch.mm(transpose(MLP_output_base), input_tar)
    nx = TT[0, 0]
    ny = TT[1, 0]
    nz = TT[2, 0]
    ox = TT[0, 1]
    oy = TT[1, 1]
    oz = TT[2, 1]
    ax = TT[0, 2]
    ay = TT[1, 2]
    az = TT[2, 2]
    px = TT[0, 3]
    py = TT[1, 3]
    pz = TT[2, 3]

    # nx.register_hook(save_grad('nx'))
    # # print("[grads]nx:", grads)
    # ny.register_hook(save_grad('ny'))
    # # print("[grads]ny:", grads)
    # nz.register_hook(save_grad('nz'))
    # # print("[grads]nz:", grads)
    # ox.register_hook(save_grad('ox'))
    # # print("[grads]ox:", grads)
    # oy.register_hook(save_grad('oy'))
    # # print("[grads]oy:", grads)
    # oz.register_hook(save_grad('oz'))
    # # print("[grads]oz:", grads)
    # ax.register_hook(save_grad('ax'))
    # # print("[grads]ax:", grads)
    # ay.register_hook(save_grad('ay'))
    # # print("[grads]ay:", grads)
    # az.register_hook(save_grad('az'))
    # # print("[grads]az:", grads)
    # px.register_hook(save_grad('px'))
    # # print("[grads]px:", grads)
    # py.register_hook(save_grad('py'))
    # # print("[grads]py:", grads)
    # pz.register_hook(save_grad('pz'))
    # print("[grads]pz:", grads)

    # 求角1
    m = py - ay*d[5]
    n = px - ax*d[5]
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" ,m, n, input_tar, MLP_output_base)
    theta11 = atan2(m, n)
    t1 = torch.stack([theta11, theta11, theta11, theta11, theta11, theta11, theta11, theta11], 0)

    # t1.register_hook(save_grad('t1'))
    # print("[grads]t1:", grads)
    # theta11.register_hook(save_grad('theta11'))
    # print("[grads]theta11:", grads)
    # m.register_hook(save_grad('m'))
    # print("[grads]m:", grads)
    # n.register_hook(save_grad('n'))
    # print("[grads]n:", grads)


    # 求角6
    A = - oz**2 + (ny**2 +ox**2)*cos(theta11)**2 + (nx**2+oy**2)*sin(theta11)**2 - (2*nx*ny-2*ox*oy)*cos(theta11)*sin(theta11)
    B = nz**2+ (nx**2+oy**2)*cos(theta11)**2+ (ny**2+ox**2)*sin(theta11)**2 + (2*nx*ny-2*ox*oy)*cos(theta11)*sin(theta11)
    C = 2*nz*oz+ (2*nx*ox-2*ny*oy)*cos(theta11)**2- (2*nx*ox - 2*ny*oy)*sin(theta11)**2 + (4*nx*oy+4*ny*ox)*cos(theta11)*sin(theta11)
    D = ay**2*cos(theta11)**2 + ax**2*sin(theta11)**2 - 2*ax*ay*sin(theta11)*cos(theta11)
    AA = B + D
    BB = C
    CC = A + D
    # print(AA,BB,CC)
    # BB.register_hook(save_grad('BB'))
    # print("[grads]BB:", grads)

    if BB**2 - 4*AA*CC >= 0:
        theta61 = atan2(-BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
        theta62 = atan2(-BB - torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
        t6 = torch.stack([theta61, theta62, theta61, theta62, theta61, theta62, theta61, theta62], 0)

    else:
        num_Error1_loss = (abs(BB**2 - 4*AA*CC) - torch.tensor([0.0])) * 1000
        # num_Error1_loss = torch.tensor([0.0], requires_grad=True)
        # angle_solution = (abs(BB**2 - 4*AA*CC) - torch.tensor([0.0])) * 1000
        angle_solution = torch.tensor([0.0], requires_grad=True)
        num_Error2_loss = torch.tensor([0.0], requires_grad=True)
        num_Error3_loss = torch.tensor([0.0], requires_grad=True)
        # num_Error1_loss = torch.tensor([0.0], requires_grad=True)
        num_Error1 += 1
        # print("角62推出来了", num_Error1_loss)
        # angle_solution.register_hook(save_grad('angle_solution'))
        # print("[grads]angle_solution:", grads)
        return angle_solution, num_Error1, num_Error1_loss, num_Error2_loss, num_Error3_loss, num_Error2, num_Error3, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    # print(theta61)

    # t6.register_hook(save_grad('t6'))
    # print("[grads]t6:", grads)
    # print("t6", t6)

    # 求角4
    DD1 = -(oy*cos(theta11)*cos(t6[0]) + ny*cos(theta11)*sin(t6[0]) - ox*sin(theta11)*cos(t6[0]) - nx*sin(theta11)*sin(t6[0]))
    DD2 = -(oy*cos(theta11)*cos(t6[1]) + ny*cos(theta11)*sin(t6[1]) - ox*sin(theta11)*cos(t6[1]) - nx*sin(theta11)*sin(t6[1]))
    # DD =  [DD1, DD2]
    # for dd in DD:
    #     if dd > 1:
    #         print(dd)
    theta41 = torch.acos(DD1)
    theta42 = torch.acos(DD2)
    t4 = torch.stack([theta41, theta41, theta42, theta42, theta41, theta41, theta42, theta42], 0)

    # t4.register_hook(save_grad('t4'))
    # print("[grads]t4:", grads)
    # print("t4", t4)

    # 求角5

    EE1 = (ax*sin(theta11) - ay*cos(theta11)) / (sin(t4[0]) + EPSILON)
    EE2 = (ax*sin(theta11) - ay*cos(theta11)) / (sin(t4[2]) + EPSILON)
    # EE1 = EE1.torch.clamp(-1.0 + EPSILON, 1.0 - EPSILON)
    # EE2 = EE1.torch.clamp(-1.0 + EPSILON, 1.0 - EPSILON)
    EE1 = torch.clamp(EE1, -1.0 + EPSILON, 1.0 - EPSILON)
    EE2 = torch.clamp(EE2, -1.0 + EPSILON, 1.0 - EPSILON)
    theta51 = torch.asin(EE1)
    theta52 = torch.asin(EE2)
    t5 = torch.stack([theta51, theta52, theta51, theta52, theta51, theta52, theta51, theta52], 0)
    # t5.register_hook(save_grad('t5'))
    # print("[grads]t5:", grads)
    # print("t5", t5)
    
    # 求角2

    FF1 = (oz*cos(t6[0]) + nz*sin(t6[0])) / (sin(t4[0]) + EPSILON)
    FF2 = (oz*cos(t6[0]) + nz*sin(t6[0])) / (sin(t4[2]) + EPSILON)
    FF3 = (oz*cos(t6[1]) + nz*sin(t6[1])) / (sin(t4[0]) + EPSILON)
    FF4 = (oz*cos(t6[1]) + nz*sin(t6[1])) / (sin(t4[2]) + EPSILON)

    GG1 = (az + FF1*cos(t4[0])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG2 = (az + FF1*cos(t4[0])*sin(t5[1])) / (cos(t5[1]) + EPSILON)

    GG3 = (az + FF2*cos(t4[2])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG4 = (az + FF2*cos(t4[2])*sin(t5[1])) / (cos(t5[1]) + EPSILON)

    GG5 = (az + FF3*cos(t4[0])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG6 = (az + FF3*cos(t4[0])*sin(t5[1])) / (cos(t5[1]) + EPSILON)

    GG7 = (az + FF4*cos(t4[2])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG8 = (az + FF4*cos(t4[2])*sin(t5[1])) / (cos(t5[1]) + EPSILON)
    GG = [0,0,0,0,0,0,0,0]
    GG = [GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG8]
    # print(GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG8)

    # FF1.register_hook(save_grad('FF1'))
    # print("[grads]FF1:", grads)
    # GG1.register_hook(save_grad('GG1'))
    # print("[grads]GG1:", grads)

    theta21 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG1) / a[2]) - math.pi / 2
    theta22 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG2) / a[2]) - math.pi / 2
    theta23 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG3) / a[2]) - math.pi / 2
    theta24 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG4) / a[2]) - math.pi / 2
    theta25 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG5) / a[2]) - math.pi / 2
    theta26 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG6) / a[2]) - math.pi / 2
    theta27 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG7) / a[2]) - math.pi / 2
    theta28 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG8) / a[2]) - math.pi / 2

    # rad_t2 = [0,0,0,0,0,0,0,0]
    # for iiii, rad_t2_single in enumerate(rad_t2):
        
    #     rad_t2[iiii] = (pz - d[0] - az*d[5] - d[3]*GG[iiii]) / a[2]
    # # print(rad_t2)
    # rad_t2 = torch.FloatTensor(rad_t2)
    # index_t2_over = torch.where(rad_t2 < -0.642788)

    # print("index_t2_over",index_t2_over)


    t2 = torch.stack([theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28], 0)

    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    # t2.register_hook(save_grad('t2'))
    # print("[grads]t2:", grads)
    # az.register_hook(save_grad('az'))
    # print("[grads]az:", grads)

    nan_index = torch.isnan(t2).nonzero()
    # print("nan_index", nan_index, t2)
    for i in nan_index:
        # print(i)
        if abs(save_what_caused_Error2_as_Nan[i]) < 2:
            the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                                                abs(abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 100
        else:
            the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                                                abs((abs(save_what_caused_Error2_as_Nan[i]) * 0.005 + torch.tensor([1.99])) - torch.tensor([1])) * 100
            # the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + torch.tensor([0.0], requires_grad=True)

    if len(nan_index) == 8:
        # aaabbb = nan_index[0].item()
        # cccddd = (pz - d[0] - az*d[5] - d[3]*GG[aaabbb]) / a[2]
        num_Error2_loss = the_NANLOSS_of_illegal_solution_with_num_and_Nan
        # num_Error2_loss = torch.tensor([0.0], requires_grad=True)
        num_Error1_loss = torch.tensor([0.0], requires_grad=True)
        num_Error3_loss = torch.tensor([0.0], requires_grad=True)
        angle_solution = torch.tensor([0.0], requires_grad=True)
        num_Error2 += 1

        return angle_solution, num_Error1, num_Error1_loss, num_Error2_loss, num_Error3_loss, num_Error2, num_Error3, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    else:
        pass

    FFF1 = (oz*cos(theta61) + nz*sin(theta61)) / (sin(theta41) + EPSILON)
    FFF2 = (oz*cos(theta61) + nz*sin(theta61)) / (sin(theta42) + EPSILON)
    FFF3 = (oz*cos(theta62) + nz*sin(theta62)) / (sin(theta41) + EPSILON)
    FFF4 = (oz*cos(theta62) + nz*sin(theta62)) / (sin(theta42) + EPSILON)

    GGG1 = (az + FFF1*cos(theta41)*sin(theta51)) / (cos(theta51) + EPSILON)
    GGG2 = (az + FFF1*cos(theta41)*sin(theta52)) / (cos(theta52) + EPSILON)

    GGG3 = (az + FFF2*cos(theta42)*sin(theta51)) / (cos(theta51) + EPSILON)
    GGG4 = (az + FFF2*cos(theta42)*sin(theta52)) / (cos(theta52) + EPSILON)

    GGG5 = (az + FFF3*cos(theta41)*sin(theta51)) / (cos(theta51) + EPSILON)
    GGG6 = (az + FFF3*cos(theta41)*sin(theta52)) / (cos(theta52) + EPSILON)

    GGG7 = (az + FFF4*cos(theta42)*sin(theta51)) / (cos(theta51) + EPSILON)
    GGG8 = (az + FFF4*cos(theta42)*sin(theta52)) / (cos(theta52) + EPSILON)

    FFF = [FFF1, FFF1, FFF2, FFF2, FFF3, FFF3, FFF4, FFF4]
    GGG = [GGG1, GGG2, GGG3, GGG4, GGG5, GGG6, GGG7, GGG8]

    # FFF1.register_hook(save_grad('FFF1'))
    # # print("[grads]FFF1:", grads)
    # GGG1.register_hook(save_grad('GGG1'))
    # # print("[grads]GGG1:", grads)
    # GGG2.register_hook(save_grad('GGG2'))
    # # print("[grads]GGG1:", grads)
    # FFF2.register_hook(save_grad('FFF2'))
    # # print("[grads]FFF2:", grads)
    # GGG3.register_hook(save_grad('GGG3'))
    # # print("[grads]GGG3:", grads)
    # GGG4.register_hook(save_grad('GGG4'))
    # # print("[grads]GGG4:", grads)
    # FFF3.register_hook(save_grad('FFF3'))
    # # print("[grads]FFF3:", grads)
    # GGG5.register_hook(save_grad('GGG5'))
    # # print("[grads]GGG5:", grads)
    # GGG6.register_hook(save_grad('GGG6'))
    # # print("[grads]GGG6:", grads)
    # FFF4.register_hook(save_grad('FFF4'))
    # # print("[grads]FFF4:", grads)
    # GGG7.register_hook(save_grad('GGG7'))
    # # print("[grads]GGG7:", grads)
    # GGG8.register_hook(save_grad('GGG8'))
    # print("[grads]GGG8:", grads)

    theta210 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG1) / a[2])
    theta220 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG2) / a[2])
    theta230 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG3) / a[2])
    theta240 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG4) / a[2])
    theta250 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG5) / a[2])
    theta260 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG6) / a[2])
    theta270 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG7) / a[2])
    theta280 = torch.acos((pz - d[0] - az*d[5] - d[3]*GGG8) / a[2])

    # pz.register_hook(save_grad('pz'))
    # print("[grads]pz:", grads)
    

    tt2 = [theta210, theta220, theta230, theta240, theta250, theta260, theta270, theta280]

    # 求角3
    theta2_3_1 = atan2(FFF1, GGG1)
    theta2_3_2 = atan2(FFF1, GGG2)
    theta2_3_3 = atan2(FFF2, GGG3)
    theta2_3_4 = atan2(FFF2, GGG4)
    theta2_3_5 = atan2(FFF3, GGG5)
    theta2_3_6 = atan2(FFF3, GGG6)
    theta2_3_7 = atan2(FFF4, GGG7)
    theta2_3_8 = atan2(FFF4, GGG8)
    # theta238 = [theta2_3_1, theta2_3_2, theta2_3_3, theta2_3_4, theta2_3_5, theta2_3_6, theta2_3_7, theta2_3_8]
 
    theta31 = theta2_3_1 - theta210 - math.pi / 2
    theta32 = theta2_3_2 - theta220 - math.pi / 2
    theta33 = theta2_3_3 - theta230 - math.pi / 2
    theta34 = theta2_3_4 - theta240 - math.pi / 2
    theta35 = theta2_3_5 - theta250 - math.pi / 2
    theta36 = theta2_3_6 - theta260 - math.pi / 2
    theta37 = theta2_3_7 - theta270 - math.pi / 2
    theta38 = theta2_3_8 - theta280 - math.pi / 2

    tt3 = [theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38]
    t3 = torch.stack([theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38], 0)
    # print(tt3)
    # theta2_3_1.register_hook(save_grad('theta2_3_1'))
    # print("[grads]theta2_3_1:", grads)
    # FFF1.register_hook(save_grad('FFF1'))
    # print("[grads]FFF1:", grads)
    # GGG1.register_hook(save_grad('GGG1'))
    # print("[grads]GGG1:", grads)


    cout_not_ok = 0
    index_ok_t2 = torch.where(torch.abs(t2) <= fanwei1[1])
    # print(index_ok_t2[0], len(index_ok_t2[0]))
    for index_ok_t2_ii in index_ok_t2[0]:
        if not torch.abs(t3[index_ok_t2_ii]) <= fanwei1[2]:
            # diff = max(0, GGG[index_ok_t2_ii]*sin(tt2[index_ok_t2_ii] - fanwei1[2])- FFF[index_ok_t2_ii]*cos(tt2[index_ok_t2_ii] - fanwei1[2])) + \
            #     max(0, FFF[index_ok_t2_ii]*cos(fanwei1[2] + tt2[index_ok_t2_ii]) - GGG[index_ok_t2_ii]*sin(fanwei1[2] + tt2[index_ok_t2_ii]))
            # the_loss_of_over = the_loss_of_over + diff * 1000
            # diff = FFF[index_ok_t2_ii]*cos(fanwei1[2] + tt2[index_ok_t2_ii]) - GGG[index_ok_t2_ii]*sin(fanwei1[2] + tt2[index_ok_t2_ii])
            # the_loss_of_over = the_loss_of_over + (diff - 0) * 1000
            # the_loss_of_over = the_loss_of_over + (abs(tt3[index_ok_t2_ii]) - fanwei1[2]) * 1000
            # the_loss_of_over = the_loss_of_over + 1000 * (max(0, FFF[index_ok_t2_ii]*cos(fanwei1[2] + tt2[index_ok_t2_ii]) - GGG[index_ok_t2_ii]*sin(fanwei1[2] + tt2[index_ok_t2_ii]))**2)

            # the_loss_of_over = the_loss_of_over + torch.tensor([(abs(t3[index_ok_t2_ii]) - fanwei1[2]) * 1000], requires_grad=True)
            # the_loss_of_over = loss_fn_t3(torch.abs(t3[index_ok_t2_ii]), fanwei1[2]) * 1000 + torch.tensor([0.0], requires_grad=True)
            cout_not_ok += 1
    if not len(index_ok_t2[0]) == 0:
        if cout_not_ok == len(index_ok_t2[0]):
            # num_Error3_loss = the_loss_of_over + the_NANLOSS_of_illegal_solution_with_num_and_Nan
            # num_Error3_loss = the_loss_of_over
            # make_dot(num_Error3_loss).view()
            # the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + the_loss_of_over
            # angle_solution = the_NANLOSS_of_illegal_solution_with_num_and_Nan
            num_Error3 += 1
            # print(the_loss_of_over)
            # the_loss_of_over.register_hook(save_grad('the_loss_of_over'))
            # print("[grads]the_loss_of_over:", grads)
            # diff.register_hook(save_grad('diff'))
            # print("[grads]diff:", grads)
            # FFF1.register_hook(save_grad('FFF1'))
            # print("[grads]FFF1:", grads)
            # GGG1.register_hook(save_grad('GGG1'))
            # print("[grads]GGG1:", grads)
            # angle_solution = torch.tensor([0.0], requires_grad=True)
            # num_Error2_loss = torch.tensor([0.0], requires_grad=True)
            # num_Error1_loss = torch.tensor([0.0], requires_grad=True)
            # return angle_solution, num_Error1, num_Error1_loss, num_Error2_loss, num_Error3_loss, num_Error2, num_Error3, the_NANLOSS_of_illegal_solution_with_num_and_Nan


    # index_ok = torch.where(~torch.isnan(t3))[0]
    # print("index_ok", index_ok, t3)
    # for index_ok_i in index_ok:
    for index_ok_i in range(8):
        the_loss_of_over = the_loss_of_over + 10 * (max(0, - fanwei1[2] - tt3[index_ok_i])**2 + max(0, tt3[index_ok_i] - fanwei1[2])**2)


    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)
    num_Error2_loss = torch.tensor([0.0], requires_grad=True)
    num_Error1_loss = torch.tensor([0.0], requires_grad=True)
    # num_Error3_loss = torch.tensor([0.0], requires_grad=True)
    num_Error3_loss = the_loss_of_over

    # print(max(0, tt3[0]))
    # print("角3推出来了", angle_solution)


    return angle_solution, num_Error1, num_Error1_loss, num_Error2_loss, num_Error3_loss, num_Error2, num_Error3, the_NANLOSS_of_illegal_solution_with_num_and_Nan


def calculate_IK_test(input_tar, MLP_output_base, a, d, alpha):

    IK_test_incorrect = 0
    EPSILON = 1e-7
    save_what_caused_Error2_as_Nan = []
    the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0], requires_grad=True)

    TT = torch.mm(transpose(MLP_output_base), input_tar)
    nx = TT[0, 0]
    ny = TT[1, 0]
    nz = TT[2, 0]
    ox = TT[0, 1]
    oy = TT[1, 1]
    oz = TT[2, 1]
    ax = TT[0, 2]
    ay = TT[1, 2]
    az = TT[2, 2]
    px = TT[0, 3]
    py = TT[1, 3]
    pz = TT[2, 3]
    # print(TT, MLP_output_base)

    # 求角1
    m = py - ay*d[5]
    n = px - ax*d[5]
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" ,m, n, input_tar, MLP_output_base)
    theta11 = atan2(m, n)
    t1 = torch.stack([theta11, theta11, theta11, theta11, theta11, theta11, theta11, theta11], 0)

    # 求角6
    A = - oz**2 + (ny**2 +ox**2)*cos(theta11)**2 + (nx**2+oy**2)*sin(theta11)**2 - (2*nx*ny-2*ox*oy)*cos(theta11)*sin(theta11)
    B = nz**2+ (nx**2+oy**2)*cos(theta11)**2+ (ny**2+ox**2)*sin(theta11)**2 + (2*nx*ny-2*ox*oy)*cos(theta11)*sin(theta11)
    C = 2*nz*oz+ (2*nx*ox-2*ny*oy)*cos(theta11)**2- (2*nx*ox - 2*ny*oy)*sin(theta11)**2 + (4*nx*oy+4*ny*ox)*cos(theta11)*sin(theta11)
    D = ay**2*cos(theta11)**2 + ax**2*sin(theta11)**2 - 2*ax*ay*sin(theta11)*cos(theta11)
    AA = B + D
    BB = C
    CC = A + D


    if BB**2 - 4*AA*CC >= 0:
        theta61 = atan2(-BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
        theta62 = atan2(-BB - torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
        t6 = torch.stack([theta61, theta62, theta61, theta62, theta61, theta62, theta61, theta62], 0)
            # print(t6, -BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)

    else:
        angle_solution = (abs(4*AA*CC - BB**2) - torch.tensor([0.0], requires_grad=True)) * 100
        # angle_solution = torch.tensor([0.0], requires_grad=True)
    #         num_Error1 += 1
    #         # print("角62推出来了", angle_solution)
    #         # if abs(4*AA*CC - BB**2) > 1e-12:
    #         #     print('!!!!!!!!!!!!!!!!!!!!!!!')

        return angle_solution

    # t6.register_hook(save_grad('t6'))
    # print("[grads]t6:", grads)
    # print("t6", t6)
    # for theta in t6:
    #     if cos(theta) == 0:
    #         print(cos(theta))

    # 求角4
    DD1 = -(oy*cos(theta11)*cos(t6[0]) + ny*cos(theta11)*sin(t6[0]) - ox*sin(theta11)*cos(t6[0]) - nx*sin(theta11)*sin(t6[0]))
    DD2 = -(oy*cos(theta11)*cos(t6[1]) + ny*cos(theta11)*sin(t6[1]) - ox*sin(theta11)*cos(t6[1]) - nx*sin(theta11)*sin(t6[1]))
    # DD =  [DD1, DD2]
    # for dd in DD:
    #     if dd > 1:
    #         print(dd)
    theta41 = torch.acos(DD1)
    theta42 = torch.acos(DD2)
    t4 = torch.stack([theta41, theta41, theta42, theta42, theta41, theta41, theta42, theta42], 0)

    # t4.register_hook(save_grad('t4'))
    # print("[grads]t4:", grads)
    # print("t4", t4)

    # 求角5
    # for ii in range(3):
    #     if sin(t4[ii]) < EPSILON:
    #         num_Error1 += 1
    #         angle_solution = t4[ii] * 100 - torch.tensor([0])
    #         # print("角5{}推出来了".format(ii), angle_solution)

    #         return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    EE1 = (ax*sin(theta11) - ay*cos(theta11)) / (sin(t4[0]) + EPSILON)
    EE2 = (ax*sin(theta11) - ay*cos(theta11)) / (sin(t4[2]) + EPSILON)
    EE1 = torch.clamp(EE1, -1.0 + EPSILON, 1.0 - EPSILON)
    EE2 = torch.clamp(EE2, -1.0 + EPSILON, 1.0 - EPSILON)
    theta51 = torch.asin(EE1)
    theta52 = torch.asin(EE2)
    t5 = torch.stack([theta51, theta52, theta51, theta52, theta51, theta52, theta51, theta52], 0)

    # t5.register_hook(save_grad('t5'))
    # print("[grads]t5:", grads)
    # print("t5", t5)
    
    # 求角2
    # for iii in range(2):
    #     if cos(t5[iii]) < EPSILON:
    #         num_Error1 += 1
    #         angle_solution = t5[iii] * 100 - torch.tensor([0])
    #         # print("角2{}推出来了".format(iii), angle_solution)

    #         return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    FF1 = (oz*cos(t6[0]) + nz*sin(t6[0])) / (sin(t4[0]) + EPSILON)
    FF2 = (oz*cos(t6[0]) + nz*sin(t6[0])) / (sin(t4[2]) + EPSILON)
    FF3 = (oz*cos(t6[1]) + nz*sin(t6[1])) / (sin(t4[0]) + EPSILON)
    FF4 = (oz*cos(t6[1]) + nz*sin(t6[1])) / (sin(t4[2]) + EPSILON)

    GG1 = (az + FF1*cos(t4[0])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG2 = (az + FF1*cos(t4[0])*sin(t5[1])) / (cos(t5[1]) + EPSILON)

    GG3 = (az + FF2*cos(t4[2])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG4 = (az + FF2*cos(t4[2])*sin(t5[1])) / (cos(t5[1]) + EPSILON)

    GG5 = (az + FF3*cos(t4[0])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG6 = (az + FF3*cos(t4[0])*sin(t5[1])) / (cos(t5[1]) + EPSILON)

    GG7 = (az + FF4*cos(t4[2])*sin(t5[0])) / (cos(t5[0]) + EPSILON)
    GG8 = (az + FF4*cos(t4[2])*sin(t5[1])) / (cos(t5[1]) + EPSILON)
    # GG = [0,0,0,0,0,0]
    # GG = [GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG]
    # print(GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG8)

    theta21 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    theta22 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    theta23 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    theta24 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    theta25 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    theta26 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    theta27 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    theta28 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    t2 = torch.stack([theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28], 0)

    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    # t2.register_hook(save_grad('t2'))
    # print("[grads]t2:", grads)
    # for iiiii in save_what_caused_Error2_as_Nan:
    #     print(abs(abs(iiiii) - torch.tensor([1.0])) * 1000)

    nan_index = torch.isnan(t2).nonzero()
    for i in nan_index:

        if abs(save_what_caused_Error2_as_Nan[i]) < 2:
            the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                        abs(abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 1000     
        else:
            the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                            abs((abs(save_what_caused_Error2_as_Nan[i]) * 0.005 + torch.tensor([1.99])) - torch.tensor([1])) * 1000
    # if the_NANLOSS_of_illegal_solution_with_num_and_Nan > 10000:
    #     print("太大了")
    if len(nan_index) == 8:
        GG = 0
        mini_nan = 5000
        for echo_loss in save_what_caused_Error2_as_Nan:
            if abs(echo_loss) < mini_nan:
                mini_nan = abs(echo_loss)
        GG = abs(mini_nan - torch.tensor([1]))
        angle_solution = GG * 1000
        # angle_solution = torch.tensor([0.0], requires_grad=True)
        IK_test_incorrect += 1
        # print("从角2出去的angle_solution: ", angle_solution)
        # assert torch.isnan(angle_solution).sum() == 0

        return angle_solution

    else:
        pass

    # assert torch.isnan(the_NANLOSS_of_illegal_solution_with_num_and_Nan).sum() == 0
    # t2.register_hook(save_grad('t2'))
    # print("[grads]t2:", grads)
    # print("t2", t2)
    # print(the_NANLOSS_of_illegal_solution_with_num_and_Nan)

    # 求角3
    theta2_3_1 = atan2(FF1, GG1)
    theta2_3_2 = atan2(FF1, GG2)
    theta2_3_3 = atan2(FF2, GG3)
    theta2_3_4 = atan2(FF2, GG4)
    theta2_3_5 = atan2(FF3, GG5)
    theta2_3_6 = atan2(FF3, GG6)
    theta2_3_7 = atan2(FF4, GG7)
    theta2_3_8 = atan2(FF4, GG8)
    # theta238 = [theta2_3_1, theta2_3_2, theta2_3_3, theta2_3_4, theta2_3_5, theta2_3_6, theta2_3_7, theta2_3_8]
    # print(theta238)

    theta31 = theta2_3_1 - theta21
    theta32 = theta2_3_2 - theta22
    theta33 = theta2_3_3 - theta23
    theta34 = theta2_3_4 - theta24
    theta35 = theta2_3_5 - theta25
    theta36 = theta2_3_6 - theta26
    theta37 = theta2_3_7 - theta27
    theta38 = theta2_3_8 - theta28

    t3 = torch.stack([theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38], 0)

    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)
    # print("角3推出来了", angle_solution)

    return angle_solution