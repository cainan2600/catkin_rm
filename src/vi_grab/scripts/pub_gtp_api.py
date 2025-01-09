#!/usr/bin/env python3
# -*- coding=UTF-8 -*-

import openai
import os
import rospy
from std_msgs.msg import String

# os.environ["http_proxy"] = "http://localhost:2017"
# os.environ["https_proxy"] = "http://localhost:2017"
# 设置 API 密钥和自定义基地址
openai.api_key = "sk-lDVvQaV6h9u0p40f90007449Bf834a878cF4A946Da705c17"
openai.api_base = "https://free.gpt.ge/v1"

def test_openai_api(prompt):
    try:
        # 调用GPT-4 Turbo模型
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # gpt-3.5-turbo模型
            # model="gpt-4o",  # gpt-3.5-turbo模型
            messages=[
                {
                "role": "system", 
                "content":"You are a specialist in robot dynamics."
                # "content": "You will assist user in controlling a mobile robot combined with a robotic arm, focusing on technical precision. \
                #     It helps users plan optimal retrieval and placement strategies for multiple objects on a tabletop. \
                #         Initially, the GPT will ask users if they would like to modify the default conditions. \
                #             The robot operates under specific constraints: the robotic arm has an effective radius of 1300mm, and the table dimensions are 4000mm in length and 2600mm in width. \
                #                 The table's origin (0, 0) is located at the bottom-left corner, with the X-axis aligned with the length and the Y-axis along the width. \
                #                     The target placement area is a 500mm by 500mm square, with its bottom-left corner located at (0, 0). \
                #                         Objects on the table include an apple, banana, orange, milk, yogurt, juice, and a plate."
                },
                {
                "role": "user", 
                "content": "hello"
                },
                {
                "role": "system", 
                "content": "tell me your problem"
                },
                {
                "role": "user", 
                "content": prompt
                }
            ],
            max_tokens=150  # 限制返回的最大token数
        )
        
        # 提取并返回模型的回复
        return response.choices[0].message['content'].strip()
    
    except Exception as e:
        return f"发生错误：{str(e)}"

# 测试函数
def main():
    test_prompt = "回归模型怎么提高精度"
    result_gpt = test_openai_api(test_prompt)
    print("模型回复：")
    print(result_gpt)

if __name__ == "__main__":
    # rospy.init_node('msg_pub_gpt', anonymous=True)
    # pub_arm_output = rospy.Publisher("msg_gpt_action", String, queue_size=10) # queue_size????
    # gpt_action = String()
    # main()
    # gpt_action.gpt_output = "{}".format(result_gpt)
    # pub_arm_output.publish(gpt_action.gpt_output)










    main()