#!/usr/bin/env python3
# -*- coding=UTF-8 -*-

import openai
import os
import rospy
from std_msgs.msg import String


def test_openai_api(usr_input):

    with open('/home/cn/catkin_rm/src/vi_grab/scripts/prompt_file/task.prompt', 'r', encoding='utf-8') as f:
        prompt_task = f.read()
    with open('/home/cn/catkin_rm/src/vi_grab/scripts/prompt_file/example.prompt', 'r', encoding='utf-8') as f:
        example_task = f.read()

    try:
        # 调用GPT-4 Turbo模型
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {
                "role": "system", 
                "content": prompt_task
                },
                {
                "role": "assistant", 
                "content": 'yes'
                },
                {
                "role": "user", 
                "content": '现在我给你三个具体的推理例子帮助你理解'
                },
                {
                "role": "assistant", 
                "content": '好的，我已经准备好了'
                },
                {
                "role": "user", 
                "content": example_task
                },
                {
                "role": "assistant", 
                "content": 'yes'
                },
                {
                "role": "user", 
                "content": usr_input
                }
            ],
            max_tokens=150  # 限制返回的最大token数
        )
        
        # 提取并返回模型的回复
        return response.choices[0].message['content'].strip()
    
    except Exception as e:
        return f"发生错误：{str(e)}"

def str_to_num(all_object_name, xxxxxxxxxxx):
    """
    将LLM输出中物品名词换为对应坐标值
    """
    result1 = all_object_name
    for i, step_object in enumerate(all_object_name):
        for ii, couple_object in enumerate(step_object):
            for iii, object_zone in enumerate(couple_object):
                if not object_zone == "target_zone":

                    result1[i][ii][iii] = xxxxxxxxxxx["Location_of_objects"]["{}".format(object_zone)] # + xy!!!!!!!!
                else:
                    result1[i][ii][iii] = target_zone_to_position(xxxxxxxxxxx["target_zone"]) # + xy!!!!!!!!

                return result1

def str_to_num_input_to_RPSN(result_gpt_2, Location_of_objects_list, xxxxxxxxxxx):
    """
    将LLM输出转换为输入RPSN的形式
    """
    result2 = []
    for step_objects in result_gpt_2:
        step_objectss = []
        for step_of_objects in step_objects:
            for item in step_of_objects:

                if item not in step_objectss and item is not None:
                    step_objectss.append(item)
        result2.append(step_objectss)

    result3 = []
    for result2_to_num in result2:
        result3_step = []
        for result2_to_num_object in result2_to_num:
            if result2_to_num_object == "plate":
                aaaaaaa = Location_of_objects_list[0]
            elif result2_to_num_object == "apple":
                aaaaaaa = Location_of_objects_list[1]
            elif result2_to_num_object == "orange":
                aaaaaaa = Location_of_objects_list[2]
            elif result2_to_num_object == "peach":
                aaaaaaa = Location_of_objects_list[3]
            elif result2_to_num_object == "milk":
                aaaaaaa = Location_of_objects_list[4]
            elif result2_to_num_object == "juice":
                aaaaaaa = Location_of_objects_list[5]
            elif result2_to_num_object == "soda":
                aaaaaaa = Location_of_objects_list[6]
            
            elif result2_to_num_object == "target_zone":
                aaaaaaa = target_zone_to_position(xxxxxxxxxxx["target_zone"])

            result3_step.append(aaaaaaa)
        result3.append(result3_step)

    return result3

def target_zone_to_position(target_zone):
    # 取中心点
    x_zuo = target_zone[0][0]
    y_zuo = target_zone[0][1]
    x_you = target_zone[1][0]
    y_you = target_zone[1][1]

    zhongxing = [(x_zuo + x_you) / 2, (y_zuo + y_you) / 2]

    return zhongxing


# 测试函数
def main_LLM():
    openai.api_key = "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"
    # 1.输入命令
    Location_of_objects_lists = [
        [2.2614665929415785, 0.19939435727009533, 0.06444841347386539, 0.17984060864068507, 1.3798559213305333, 0.13171522506225303],
        [1.541372559977209, -0.618561892382764, -0.8951714497663478, -0.030351880384557495, 1.4131248117808244, 0.13974098022410492],
        [2.408354330487878, 0.8148508180052912, 0.03703070064853679, -0.033120757976618176, 1.5598670776384305, 0.14619308217614937],
        [1.74752881537881, -1.424395932924747, -0.484452801557166, 0.09478099533697348, 1.2890599182139901, 0.14989546656065625],
        [2.464884158938659, 0.07189036860878878, -0.6892274155425099, -0.15625375065112102, 1.5992589427437653, 0.13165024902566866],
        [-2.3626866590121716, -0.7637248846919906, -2.6488281853706956, 0.19637785827121557, 1.4012840143271035, 0.13442900940869726],
        [1.3371326602672535, 0.14717602870970417, -1.364890027629934, -0.09157310805369989, 1.52468503262029, 0.14363987212258827]
    ]

# 2.2614665929415785, 0.19939435727009533, 0.06444841347386539, 0.17984060864068507, 1.3798559213305333, 0.13171522506225303
# 1.541372559977209, -0.618561892382764, -0.8951714497663478, -0.030351880384557495, 1.4131248117808244, 0.13974098022410492
# 2.408354330487878, 0.8148508180052912, 0.03703070064853679, -0.033120757976618176, 1.5598670776384305, 0.14619308217614937
# 1.74752881537881, -1.424395932924747, -0.484452801557166, 0.09478099533697348, 1.2890599182139901, 0.14989546656065625
# 2.464884158938659, 0.07189036860878878, -0.6892274155425099, -0.15625375065112102, 1.5992589427437653, 0.13165024902566866
# -2.3626866590121716, -0.7637248846919906, -2.6488281853706956, 0.19637785827121557, 1.4012840143271035, 0.13442900940869726
# 1.3371326602672535, 0.14717602870970417, -1.364890027629934, -0.09157310805369989, 1.52468503262029, 0.14363987212258827

    Location_of_objects_lists_copy = Location_of_objects_lists
    for ii, tensor in enumerate(Location_of_objects_lists):
        tensor = [round(val, 3) for val in tensor]
        Location_of_objects_list[ii] = tensor
    xxxxxxxxxxx = {
        "Target": "Put all items back in the target area, regardless of the plates.",
        "Initial_position_of_AMMR": [0, 0],
        "target_zone": [[-0.25, 0.803],[0.15, 1.203]],
        "Location_of_objects": {
            "plate":[Location_of_objects_list[0][3], Location_of_objects_list[0][4]], 
            "apple":[Location_of_objects_list[1][3], Location_of_objects_list[1][4]], "orange":[Location_of_objects_list[2][3], Location_of_objects_list[2][4]], 
            "peach":[Location_of_objects_list[3][3], Location_of_objects_list[3][4]], "milk":[Location_of_objects_list[4][3], Location_of_objects_list[4][4]], 
            "juice":[Location_of_objects_list[5][3], Location_of_objects_list[5][4]], "soda":[Location_of_objects_list[6][3], Location_of_objects_list[6][4]],
            "None":None
        }
    }

    # xxxxxxxxxxx = "你理解了吗"
    test_prompt = "{}".format(xxxxxxxxxxx)
    result_gpt = test_openai_api(test_prompt)
    print("模型回复：")
    print(result_gpt)

    # 2.处理初步的答案，提取出只包含名词的列表(需要检查是否需要)
    result_gpt_2 = result_gpt

    # 3.str_to_num
    result1 = str_to_num(result_gpt_2, xxxxxxxxxxx)

    # 4.str_to_num_input_to_RPSN
    result22 = str_to_num_input_to_RPSN(result_gpt_2, Location_of_objects_lists_copy, xxxxxxxxxxx)

    print(result_gpt_2, result1, result22)
    # return result_gpt_2, result1, result22

if __name__ == "__main__":
    main_LLM()

    # all_object_name = [[[apple, plate], [orange, plate]], [[milk, None]], [[None, tar_zone]]]
    # all_object_position = [[[[x, y, z, w], [x, y, z, w]], [[x, y, z, w], [x, y, z, w]]], [[[x, y, z, w], [None]]], [[[None], [x, y, z, w]]]]

    # rospy.init_node('msg_pub_gpt', anonymous=True)
    # pub_arm_output = rospy.Publisher("msg_gpt_action", String, queue_size=10) # queue_size????
    # gpt_action = String()
    # main()
    # gpt_action.gpt_output = "{}".format(result_gpt)
    # pub_arm_output.publish(gpt_action.gpt_output)