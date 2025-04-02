def calculate_FK_loss(angles, FK_results, input_target,intermediate_output):
    # print(angles, FK_results, input_target)
    FK_loss = torch.tensor([0.0], requires_grad=True)

    for angle in angles:
        if -torch.pi < angle and angle < torch.pi:
            pass
        elif -torch.pi > angle:
            FK_loss = FK_loss + (-torch.pi-angle)
        else:
            FK_loss = FK_loss + (angle - torch.pi)
    if FK_loss != 0:
        global num_Error1
        num_Error1 = num_Error1 + 1
    # print('!',FK_results[:3, 3])
    # print('!',input_target[3:6])
    # 计算损失
    MSELoss = nn.MSELoss()

    if MSELoss(intermediate_output , input_target[3:5]) > 2.5:
        global num_Error2
        num_Error2 = num_Error2 + 1
        FK_loss = FK_loss + (MSELoss(intermediate_output , input_target[3:5])-torch.tensor([2.5]))*10
        # print('1:', MSELoss(intermediate_output , input_target[1:3])-torch.tensor([1.5]))
    # MSELoss = nn.MSELoss()
    # FK_loss = FK_loss + MSELoss(FK_results[:3, 3], input_target[3:6])
    # if MSELoss(FK_results[:3, 3], input_target[3:6]) > 0.01:
    #     global num_NOError1
    #     num_NOError1 = num_NOError1 + 1
    # else:
    #     global num_NOError2
    #     num_NOError2 = num_NOError2 + 1

    if MSELoss(FK_results[:3, 3] , input_target[3:6]) > 0.001:
        global num_NOError1
        num_NOError1 = num_NOError1 + 1
        # print('FK_results[:3, 3]', FK_results[:3, 3])
        # print('input_target[3:6]', input_target[3:6])
        # print(MSELoss(FK_results[:3, 3] , input_target[3:6]))

        FK_loss = FK_loss + MSELoss(FK_results[:3, 3] , input_target[3:6])
        # print('2:', MSELoss(FK_results[:3, 3] , input_target[3:6]))
    else:
        global num_NOError2
        num_NOError2 = num_NOError2 + 1


    return FK_loss


def calculate_FK_loss_test(angles, FK_results, input_target,intermediate_output):
    # print(angles, FK_results, input_target)
    FK_loss = torch.tensor([0.0], requires_grad=True)

    for angle in angles:
        if -torch.pi < angle and angle < torch.pi:
            pass
        elif -torch.pi > angle:
            FK_loss = FK_loss + (-torch.pi-angle)
        else:
            FK_loss = FK_loss + (angle - torch.pi)
    MSELoss = nn.MSELoss()
    if FK_loss != 0 or MSELoss(intermediate_output , input_target[3:5]) > 2.5 or MSELoss(FK_results[:3, 3] , input_target[3:6]) > 0.005:
        global incorrect
        incorrect = incorrect + 1

    else:
        global correct
        correct = correct + 1
    return FK_loss