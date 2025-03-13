#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
from std_msgs.msg import String, Bool,Empty
import time
import rospy, sys
from rm_msgs.msg import MoveJ_P,Arm_Current_State,Gripper_Set, Gripper_Pick,ArmState,MoveL,MoveJ,set_modbus_mode,write_register,write_single_register,Tool_Analog_Output
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import 
from vi_msgs.msg import ObjectInfo
from geometry_msgs.msg import TransformStamped,PointStamped
from geometry_msgs.msg import Point, Quaternion
import actionlib
# from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
from actionlib import GoalStatus
from .trans_all_rm import *
from .run_LLM import main_LLM
from .run_RPSN import main_RPSN

# 相机坐标系到机械臂末端坐标系的旋转矩阵，通过手眼标定得到
rotation_matrix = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]])
# 相机坐标系到机械臂末端坐标系的平移向量，通过手眼标定得到
translation_vector = np.array([-0.08039019, 0.03225555, -0.09756825])

#move to point
def navigateToGoal(x, y, orientation_z, orientation_w):
    ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    ac.wait_for_server(rospy.Duration(5.0))

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.z = orientation_z
    goal.target_pose.pose.orientation.w = orientation_w
    ac.send_goal(goal)
    ac.wait_for_result()
    state = ac.get_state()
    if state == GoalStatus.SUCCEEDED:
        rospy.loginfo("Successfully reached the goal location")
    else:
        rospy.loginfo("Failed to reach the goal location")

# 接收到识别物体的回调函数
def object_pose_callback(data):
    """
    函数功能：每帧图像经过识别后的回调函数，若有抓取指令，则判断当前画面帧中是否有被抓物体，如果有则将物体坐标进行转换，并让机械臂执行抓取动作
    输入参数：无
    返回值：无
    """
    global object_msg, grasp_state, grasp_plate, x_chasis, y_chasis
    # 判断当前帧的识别结果是否有要抓取的物体
    if data.object_class == object_msg:

        # 等待当前的机械臂末端位姿（相对于机械臂基座）
        arm_pose_msg = rospy.wait_for_message("/rm_driver/Arm_Current_State", Arm_Current_State, timeout=None)
        print(arm_pose_msg)
        #rospy.sleep(1)
        # 等待接收当前机械臂位姿四元数形式（相对与世界坐标系）
        arm_orientation_msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=None)
        print(arm_orientation_msg)
        # 计算机械臂基坐标系下的物体坐标
        result = convert(data.x,data.y,data.z,arm_pose_msg.Pose[0],arm_pose_msg.Pose[1],arm_pose_msg.Pose[2],arm_pose_msg.Pose[3],arm_pose_msg.Pose[4],arm_pose_msg.Pose[5])
        print(data.object_class,':',result)
        # 抓取物体0
        if grasp_state:
            catch(result,arm_orientation_msg)
        else:
            # 放置物体
            if object_msg == "target_zone":
                if grasp_plate:
                    # 得到当前底盘位置，取目标区域中心点与底盘位置，得到两者间位置关系，判断放在哪一边合适
                    x_diff = result[0] - x_chasis
                    y_diff = result[1] - y_chasis
                    if x_diff > 0:
                        if y_diff > 0:
                            result[0] = result[0] - 0.3
                        else:
                            result[1] = result[1] + 0.3
                    else:
                        if  y_diff > 0:
                            result[0] = result[0] + 0.3
                        else:
                            result[1] = result[1] + 0.3
                else:
                    x_diff = result[0] - x_chasis
                    y_diff = result[1] - y_chasis
                    if x_diff > 0:
                        if y_diff > 0:
                            result[0] = result[0] - 0.1
                        else:
                            result[1] = result[1] + 0.1
                    else:
                        if  y_diff > 0:
                            result[0] = result[0] + 0.1
                        else:
                            result[1] = result[1] + 0.1

            elif object_msg == "plate":
                x_diff = result[0] - x_chasis
                y_diff = result[1] - y_chasis
                if x_diff > 0:
                    if y_diff > 0:
                        result[0] = result[0] - 0.05
                    else:
                        result[1] = result[1] + 0.05
                else:
                    if  y_diff > 0:
                        result[0] = result[0] + 0.05
                    else:
                        result[1] = result[1] + 0.05

            else: # 将盘子放到目标物体中央
                if grasp_plate:



                    return
            place(result,arm_orientation_msg)
        # # 清除object_msg的信息，之后二次发布抓取物体信息可以再执行
        # object_msg = ''

# 相机坐标系物体到机械臂基坐标系转换函数
def convert(x,y,z,x1,y1,z1,rx,ry,rz):
    """
    函数功能：我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
    机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）
    输入参数：深度相机识别到的物体坐标（x, y, z）和机械臂末端的位姿（x1,y1,z1,rx,ry,rz）
    返回值：物体在机械臂基座坐标系下的位置（x, y, z）
    """
    global rotation_matrix,translation_vector
    obj_camera_coordinates = np.array([x, y, z])

    # 机械臂末端的位姿，单位为弧度
    end_effector_pose = np.array([x1, y1, z1,
                                  rx, ry, rz])
    # 将旋转矩阵和平移向量转换为齐次变换矩阵
    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector
    # 机械臂末端的位姿转换为齐次变换矩阵
    position = end_effector_pose[:3]
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position
    # 计算物体相对于机械臂基座的位姿
    obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
    #obj_end_effector_coordinates_homo = np.linalg.inv(T_camera_to_end_effector).dot(obj_camera_coordinates_homo)
    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
    obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    obj_base_coordinates = obj_base_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标
    # 计算物体的旋转
    obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(rotation_matrix)
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)
    # 组合结果
    obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
    obj_base_pose[3:] = rx,ry,rz
    return obj_base_pose

def catch(result,arm_orientation_msg):
    '''
    函数功能：机械臂执行抓取动作
    输入参数：经过convert函数转换得到的‘result’和机械臂当前的四元数位姿‘arm_orientation_msg’
    返回值：无
    '''
    # 上一步通过pic_joint运动到了识别较好的姿态，然后就开始抓取流程
    # 流程第一步：经过convert转换后，得到了机械臂坐标系下的物体位置坐标result，通过movej_p运动到result目标附近，因为不能一下就到达
    movejp_type([result[0]+0.07,result[1],result[2],arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************catching  step1*************************')
    time.sleep(4)
    # 抓取第二步：通过抓取第一步已经到达了物体前方，后续使用movel运动方式让机械臂直线运动到物体坐标处
    movel_type([result[0],result[1],result[2],arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************catching  step2*************************')
    time.sleep(4)
    # 抓取第三步：到达目标处，闭合夹爪
    gripper_close()
    print('*************************catching  step3*************************')
    time.sleep(4)
    # gripper_open()
    arm_ready_pose()
    print('*************************catching  step4*************************')

def place(result,arm_orientation_msg):
    '''
    函数功能：机械臂执行抓取动作
    输入参数：经过convert函数转换得到的‘result’和机械臂当前的四元数位姿‘arm_orientation_msg’
    返回值：无
    '''
    # 上一步通过pic_joint运动到了识别较好的姿态，然后就开始抓取流程
    # 流程第一步：经过convert转换后，得到了机械臂坐标系下的物体位置坐标result，通过movej_p运动到result目标附近，因为不能一下就到达
    movejp_type([result[0]+0.07,result[1],result[2],arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************placeing  step1*************************')
    movel_type([result[0],result[1],result[2],arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************placeing  step2*************************')
    # 抓取第三步：到达目标处，闭合夹爪
    time.sleep(4)
    gripper_open()
    print('*************************placeing  step3*************************')
    time.sleep(4)
    arm_ready_pose()


def movejp_type(pose,speed):
    '''
    函数功能：通过输入机械臂末端的位姿数值，让机械臂以指定速度（0-1，最好小于0.5，否则太快）运动到指定位姿
    输入参数：pose（position.x、position.y、position.z、orientation.x、orientation.y、orientation.z、orientation.w）、speed
    返回值：无
    '''
    moveJ_P_pub = rospy.Publisher("rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=1)
    rospy.sleep(0.5)
    move_joint_pose = MoveJ_P()
    move_joint_pose.Pose.position.x = pose[0]
    move_joint_pose.Pose.position.y = pose[1]
    move_joint_pose.Pose.position.z = pose[2]
    move_joint_pose.Pose.orientation.x = pose[3]
    move_joint_pose.Pose.orientation.y = pose[4]
    move_joint_pose.Pose.orientation.z =  pose[5]
    move_joint_pose.Pose.orientation.w =  pose[6]
    move_joint_pose.speed = speed
    moveJ_P_pub.publish(move_joint_pose)


def movel_type(pose,speed):
    '''
    函数功能：通过输入机械臂末端的位姿数值，让机械臂以指定速度（0-1，最好小于0.5，否则太快）直线运动到指定位姿
    输入参数：pose（position.x、position.y、position.z、orientation.x、orientation.y、orientation.z、orientation.w）、speed
    返回值：无
    '''
    moveL_pub = rospy.Publisher("rm_driver/MoveL_Cmd", MoveL, queue_size=1)
    rospy.sleep(0.5)
    move_line_pose = MoveL()
    move_line_pose.Pose.position.x = pose[0]
    move_line_pose.Pose.position.y = pose[1]
    move_line_pose.Pose.position.z = pose[2]
    move_line_pose.Pose.orientation.x = pose[3]
    move_line_pose.Pose.orientation.y = pose[4]
    move_line_pose.Pose.orientation.z =  pose[5]
    move_line_pose.Pose.orientation.w =  pose[6]
    move_line_pose.speed = speed
    moveL_pub.publish(move_line_pose)

def arm_ready_pose():
    '''
    函数功能：执行整个抓取流程前先运动到一个能够稳定获取物体坐标信息的姿态，让机械臂在此姿态下获取识别物体的三维坐标，机械臂以关节运动的方式到达拍照姿态，
    此关节数值可以根据示教得到，将机械臂通过按住绿色按钮拖动到能够获取较好效果的姿态
    输入参数：角度（弧度）
    返回值：无
    0, -50, 110, 0, 60, 0
    '''
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    pic_joint = MoveJ()
    pic_joint.joint = [0, -0.8726646, 1.91986218, 0, 1.2217304764, 0]
    pic_joint.speed = 0.3
    moveJ_pub.publish(pic_joint)

    
def gripper_open():
    '''
    函数功能：打开4C2夹爪
    输入参数：无
    返回值：无
    '''
    set_pub = rospy.Publisher("rm_driver/Gripper_Set", Gripper_Set, queue_size=1)
    rospy.sleep(1)
    set = Gripper_Set()
    set.position = 1000
    set_pub.publish(set)

def gripper_close():
    '''
    函数功能：闭合4C2夹爪
    输入参数：无
    返回值：无
    '''
    pick_pub = rospy.Publisher("rm_driver/Gripper_Pick_On", Gripper_Pick, queue_size=1)
    rospy.sleep(1)
    pick1 = Gripper_Pick()
    pick1.speed = 200
    pick1.force = 1000
    pick_pub.publish(pick1)

def read_value(input):
    if len(input) == 4:
        x = input[0]
        y = input[1]
        z = input[2]
        w = input[3]
        return x, y, z, w
    if len(input) == 2:
        x = input[0]
        y = input[1]
        return x, y

def get_see_pose():
    '''
    使机械臂末端执行器Z轴与目标物体处于同一直线，找到适合观察的位置
    '''
    global object_msg, tar_chasis_position, object_position, rotation_matrix, translation_vector
    corrent_chasis_position = tar_chasis_position
    tar_object_name = object_msg
    tar_object_position = object_position

    # 神经网络输出的基座位置
    x_chasis, y_chasis, z_chasis, w_chasis = read_value(tar_chasis_position)
    x_boject, y_object = read_value(object_position)
    position_object_to_world = [x_boject, y_object, 0.52, 1]

    # T_base_to_end_effector
    arm_pose_msg = rospy.wait_for_message("/rm_driver/Arm_Current_State", Arm_Current_State, timeout=None)
    end_effector_pose = np.array(
                                    [arm_pose_msg.Pose[0],arm_pose_msg.Pose[1],arm_pose_msg.Pose[2],
                                    arm_pose_msg.Pose[3],arm_pose_msg.Pose[4],arm_pose_msg.Pose[5]]
                            )
    position = end_effector_pose[:3]
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position

    # T_world_to_end_effector
    arm_orientation_msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=None)
    # quaternion = [
    #                 arm_orientation_msg.Pose.orientation.x, arm_orientation_msg.Pose.orientation.y,
    #                 arm_orientation_msg.Pose.orientation.z, arm_orientation_msg.Pose.orientation.w
    #     ]
    # rotation = R.from_quat(quaternion)
    # position_1 = [arm_orientation_msg.position.x, arm_orientation_msg.position.y, arm_orientation_msg.position.z]
    # T_world_to_end_effector = np.eye(4)
    # T_world_to_end_effector[:3, :3] = rotation
    # T_world_to_end_effector[:3, 3] = position_1

    # T_world_to_base
    T_world_to_base = np.array([
        [1 - 2 * z_chasis**2, -2 * w_chasis * z_chasis, 0, x_chasis],
        [2 * w_chasis * z_chasis, 1 - 2 * z_chasis**2, 0, y_chasis],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # # T_world_to_base
    # inv_T_base_to_end_effector = np.linalg.inv(T_base_to_end_effector)
    # T_world_to_base = T_world_to_end_effector.dot(inv_T_base_to_end_effector)

    # position_object_to_base(4x4 * 4x1)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # position_object_to_world = np.linalg.inv(position_object_to_world)
    # position_object_to_base41 = T_world_to_base.dot(position_object_to_world)
    T_world_to_base_inv = np.linalg.inv(T_world_to_base)
    position_object_to_base41 = T_world_to_base_inv.dot(position_object_to_world)
    position_object_to_base_x = position_object_to_base41[0][0]
    position_object_to_base_y = position_object_to_base41[1][0]
    position_object_to_base_z = position_object_to_base41[2][0]
    position_object_to_base = [position_object_to_base_x, position_object_to_base_y, position_object_to_base_z]

    # 方向向量
    position_end_effector_to_base = position
    xiangliang_d = position_object_to_base - position_end_effector_to_base
    d_norm = xiangliang_d / np.linalg.norm(xiangliang_d)

    # # T_camera_to_end_effector
    # T_camera_to_end_effector = np.eye(4)
    # T_camera_to_end_effector[:3, :3] = rotation_matrix
    # T_camera_to_end_effector[:3, 3] = translation_vector

    # # position_object_to_camera
    # position_object_to_camera = T_camera_to_end_effector.dot(position_object_to_base)

    # 计算新旋转矩阵
    # 选择参考向量，确保不与z_new共线
    ref_vec = np.array([1, 0, 0], dtype=float)
    if np.linalg.norm(np.cross(ref_vec, d_norm)) < 1e-6:
        ref_vec = np.array([0, 1, 0], dtype=float)
    # 计算新的X轴
    x_new = ref_vec - np.dot(ref_vec, d_norm) * d_norm
    x_new /= np.linalg.norm(x_new)
    # 计算新的Y轴
    y_new = np.cross(d_norm, x_new)
    # 构造旋转矩阵
    R_new = np.column_stack((x_new, y_new, d_norm))

    # 转换为四元数形式
    q_target = rotation_matrix_to_quaternion(R_new)

    # move
    movejp_type(
        [
            arm_orientation_msg.position.x, arm_orientation_msg.position.y, arm_orientation_msg.position.z,
            q_target[0], q_target[1], q_target[2], q_target[3]
        ],
        0.3
    )
    time.sleep(4)


# def movej_type(joint,speed):
#     '''
#     函数功能：通过输入机械臂每个关节的数值（弧度），让机械臂以指定速度（0-1，最好小于0.5，否则太快）运动到指定姿态
#     输入参数：[joint1,joint2,joint3,joint4,joint5,joint6]、speed
#     返回值：无
#     '''
#     moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
#     rospy.sleep(0.5)
#     move_joint = MoveJ()
#     move_joint.joint = joint
#     move_joint.speed = speed
#     moveJ_pub.publish(move_joint)

# def set_mode():
#     '''
#     函数功能：设置modbus模式
#     输入参数：无
#     返回值：无
#     '''
#     pub_modbus_mode = rospy.Publisher("/rm_driver/Set_Modbus_Mode_Cmd", set_modbus_mode, queue_size=1)
#     rospy.sleep(1)
#     set_modbus_mode = set_modbus_mode()
#     set_modbus_mode.port = 1
#     set_modbus_mode.baudrate = 9600
#     set_modbus_mode.timeout = 5
#     pub_modbus_mode.publish(set_modbusmode)

# def modbus_gripper_set(num1, num2, num3, num4, num5 ):
#     '''
#     函数功能：写多个寄存器
#     输入参数：port、address、num、data、device
#     返回值：无
#     '''
#     pub_write_register = rospy.Publisher("/rm_driver/Write_Register_Cmd", write_register, queue_size=1)
#     rospy.sleep(1)
#     write_reg = write_register()
#     write_reg.port = num1
#     write_reg.address = num2
#     write_reg.num = num3
#     write_reg.data = num4
#     write_reg.device = num5
#     pub_write_register.publish(write_reg)

# def modbus_gripper_control():
#     '''
#     函数功能：写单个寄存器
#     输入参数：无
#     返回值：无
#     '''
#     pub_write_single_register = rospy.Publisher("/rm_driver/Write_Single_Register_Cmd", write_single_register, queue_size=1)
#     rospy.sleep(1)
#     write_sin_reg = write_register()
#     write_sin_reg.port = 1
#     write_sin_reg.address = 45
#     write_sin_reg.data = 0
#     write_sin_reg.device = 1
#     pub_write_single_register.publish(write_sin_reg)

# def set_tool():
#     '''
#     函数功能：设置工具端电压输出
#     输入参数：无
#     返回值：无
#     '''
#     pub_tool_voltage = rospy.Publisher("/rm_driver/Tool_Analog_Output",Tool_Analog_Output,queue_size=1)
#     rospy.sleep(1)
#     set_vol = Tool_Analog_Output()
#     set_vol.voltage = 24
#     pub_tool_voltage.publish(set_vol)


if __name__ == '__main__':
    
    print('*************************step1————————LLM*************************')

    all_object_name, all_object_position, all_object_position_input_to_RPSN = main_LLM()

    print('*************************step2————————RPSN*************************')

    all_tar_chasis_position = main_RPSN(all_object_position_input_to_RPSN)

    print('*************************step3——————开始验证*************************')

    # all_object_name = [[[apple, plate], [orange, plate]], [[milk, None]], [[None, tar_zone]]]
    # all_object_position = [[[[x, y, z], [x, y, z]], [[x, y, z], [x, y, z]]], [[[x, y, z], [None]]], [[[None], [x, y, z]]]]
    # all_tar_chasis_position = [[x, y, z, w], [x, y, z, w], [x, y, z, w]]

    rospy.init_node('object_catch')
    arm_ready_pose()
    gripper_open()
    for i_chasis_positon, step in enumerate(len(all_tar_chasis_position)):

        tar_chasis_position = all_tar_chasis_position[step]
        x_chasis, y_chasis, z_chasis, w_chasis = read_value(tar_chasis_position)
        # # 导航之前需要将机械臂基座位姿转换为底盘位姿
        # T_world_to_base = np.array([
        #     [1 - 2 * z_chasis**2, -2 * w_chasis * z_chasis, 0],
        #     [2 * w_chasis * z_chasis, 1 - 2 * z_chasis**2, 0],
        #     [0, 0, 1]
        # ])
        # # T_chasis_to_base 仅绕Z旋转
        # T_chasis_to_base = [
        #     '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        # ]
        # T_chasis_to_world = T_world_to_base.dot(np.linalg.inv(T_chasis_to_base))
        # # 旋转矩阵转四元数
        # navigateToGoal_4 = rotation_matrix_to_quaternion(T_chasis_to_world)
        # assert navigateToGoal_4==0 and navigateToGoal_4[1] == 0
        # # 导航
        # navigateToGoal(x_chasis, y_chasis, navigateToGoal_4[2], navigateToGoal_4[3])
        navigateToGoal(x_chasis, y_chasis, z_chasis, w_chasis)

        sub_step = all_object_position[step]

        for i_sub_position, sub_sub_step in enumerate(sub_step):

            object_position = sub_sub_step[0]
            tar_place_position = sub_sub_step[1]

            if object_position is None:
                # 放下---true为打开-false为关闭
                grasp_state = False
                object_msg = str(all_object_name[i_chasis_positon][i_sub_position][1])
                # rospy.init_node('object_catch')
                get_see_pose()
                sub_object_pose = rospy.Subscriber("/object_pose", ObjectInfo, object_pose_callback, queue_size=1)
                rospy.spin()
                #time.sleep(5)
            else:
                # 拿起来
                grasp_state = True

                object_msg = str(all_object_name[i_chasis_positon][i_sub_position][0])
                if object_msg == "plate":
                    grasp_plate = True
                else:
                    grasp_plate = False
                # rospy.init_node('object_catch')
                get_see_pose()
                sub_object_pose = rospy.Subscriber("/object_pose", ObjectInfo, object_pose_callback, queue_size=1)
                rospy.spin()
                #time.sleep(5)

            if not tar_place_position is None:
                # 放下
                grasp_state = False
                object_msg = str(all_object_name[i_chasis_positon][i_sub_position][1])
                # rospy.init_node('object_catch')
                get_see_pose()
                sub_object_pose = rospy.Subscriber("/object_pose", ObjectInfo, object_pose_callback, queue_size=1)
                rospy.spin()
                #time.sleep(5)
            else:
                pass

