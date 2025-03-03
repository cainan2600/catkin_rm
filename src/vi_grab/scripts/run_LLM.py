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

                result1[i][ii][iii] = xxxxxxxxxxx["Location_of_objects"]["{}".format(object_zone)]

                return result1

def str_to_num_input_to_RPSN(result_gpt_2):
    """
    将LLM输出转换为输入RPSN的形式
    """
    step_objectss = []
    result2 = []
    for step_objects in result_gpt_2:
        for step_of_objects in step_objects:
            for object_1 in step_of_objects:
                item = object_1
                num_none = 0
                if item is None:
                    num_none += 1
                if item not in step_objectss:
                    
                    return

    return result2


# 测试函数
def main_LLM():
    openai.api_key = "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"
    # 1.输入命令
    xxxxxxxxxxx = {
        "Target": "引号内是需要实现的目标",
        "Initial_position_of_AMMR": [AMMR的x轴坐标, AMMR的y轴坐标],
        "target_zone": [[目标区域的x轴坐标, 目标区域的y轴坐标],[目标区域的x轴坐标, 目标区域的y轴坐标]],
        "Location_of_objects": {"苹果":[苹果x轴坐标, 苹果y轴坐标], "橘子":[橘子x轴坐标, 橘子y轴坐标], "桃子":[桃子x轴坐标, 桃子y轴坐标], "牛奶":[牛奶x轴坐标, 牛奶y轴坐标], "果汁":[果汁x轴坐标, 果汁y轴坐标], "苏打水":[苏打水x轴坐标, 苏打水y轴坐标]}
    }
    test_prompt = "{}".format(xxxxxxxxxxx)
    result_gpt = test_openai_api(test_prompt)
    print("模型回复：")
    print(result_gpt)

    # 2.处理初步的答案，提取出只包含名词的列表(需要检查是否需要)
    result_gpt_2 = result_gpt

    # 3.str_to_num
    result1 = str_to_num(result_gpt_2, xxxxxxxxxxx)

    # 4.str_to_num_input_to_RPSN
    result2 = str_to_num_input_to_RPSN(result_gpt_2)

    return result_gpt_2, result1, result2

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