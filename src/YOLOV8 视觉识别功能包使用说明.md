# YOLOV8 视觉识别功能包使用说明

## 1.项目概述

机械臂作为当前社会生产中必不可少的执行单元，已经融入到我们的各类生产生活中，如何让机械臂更加智能的执行任务是提高生产效率与生产智能化的持续目标，因此本项目为机械臂添加“眼睛”以增加机械臂的感知信息，并赋予”眼睛“识别周围环境信息的能力，以提升其执行智能化程度，为后续集成为机器人完成自动化生产任务提供基础。

<img src="././pic/抓取.png" alt="pic" style="zoom:50%;" />

### 1.1项目背景

机器人在传统应用中多以执行器的角色被使用，例如机械臂在流水线上以固定姿态顺序操作工件，这就使得其姿态设定必须非常精确严格，并且工作过程中出现任何偏移都会打乱整个流程，导致巨大的损失。为使机器人能够更加智能的对目标物体进行操作，就需要赋予机器人以视觉，传统的工业相机能够提供物体的轮廓颜色等信息，满足部分工厂生产条件，但由于其无法理解物体的语义信息，导致无法走出工厂，为日常生活提供智慧服务。

深度学习的出现加速了机器人进行图像理解，依据各种算法，已经能从二维图像获取高级语义信息，极大地帮助机器人理解周围环境。YOLOv8（You Only Look Once version 8）是一个深度学习框架，用于实现实时对象检测。它是 YOLO 系列的最新迭代，旨在提供更高的准确性和速度。其特点如下：

- **实时性能**: YOLOv8 继续保持 YOLO 系列的实时检测特性，即使在较低的硬件配置上也能达到很高的帧率（FPS）。
- **高准确度**: 通过更深更复杂的网络结构和改进的训练技巧，YOLOv8 在保持高速度的同时，也大幅提高了检测的准确度。
- **多尺度预测**: YOLOv8 引入了改进的多尺度预测技术，可以更好地检测不同大小的对象。
- **自适应锚框**: 新版在自适应调整锚框方面做了优化，可以更准确地预测对象的位置和大小。

通过YOLOV8赋能机器人，是当前机器人智能化进程的通用做法，而ROS操作系统作为当前使用最广泛的机器人操作系统，基于其点对点设计以及服务和节点管理器等机制，能够分散由计算机视觉和语音识别等功能带来的实时计算压力，能够适应多机器人遇到的挑战。基于以上背景制作了YOLOV8的视觉功能包，方便开发者在ROS中获取视觉信息。

### 1.2主要目标

- **视觉识别功能集成**：将视觉识别算法与机械臂执行功能集成，完成信息互通。
- **智能执行**：通过视觉识别的信息，让机械臂运动到指定位置，并抓取指定物体。

### 1.3核心功能

- **视觉识别物体信息发布**：将经过视觉识别算法得到的物体信息通过ROS话题发布，方便用户直接调用。
- **多类型轨迹执行**：机械臂有多种轨迹规划算法，能够让机械臂通过不同的运动方式到达目标点，保护设备不受损坏。

### 1.4技术亮点

- **高性能机械臂**：睿尔曼RM65-6f-v机械臂以其6自由度和5kg负载能力，提供广泛的操作范围和高精度作业能力。
- **基于ROS的软件架构**：采用ROS noetic版本，构建模块化、可扩展的软件系统，支持二次开发和功能扩展。

### 1.5应用前景

该视觉识别抓取系统能够快速集成到各类机器人上，方便完成如无人商超、智慧农业等指定场景功能需求。

### 1.6更新日志

|  更新日期  |         更新内容         | 版本号 |
| :--------: | :----------------------: | :----: |
| 2024/11/21 | YOLOV8视觉识别功能包发布 | v1.0.0 |

## 2.软硬件概述

YOLOV8 视觉识别功能包基于RM产品开发，利用YOLOV8和D435相机视觉识别物体，得到物体的三维坐标，最终让RM机械臂完成一个抓取动作。使用此功能包仅仅需要将之加入原RM产品的功能包之中，编译通过后即可使用，提供单机械臂抓取水瓶并倒水的demo。

```
​```

vi_work/

   └─src                                     
   │   ├─pic                                    // 图片库
   │   ├─rm_robot                               // 机械臂ROS功能包
   │   ├─vi_grab
   │   │   └─CMakeLists.txt
   │   │   └─package.xml
   │   │   └─include
   │   │   └─launch            
   │   │      └─ vi_pour_demo.launch              // 视觉识别水瓶并执行倒水demo 启动文件
   │   │   └─model
   │   │      └─ yolov8n.pt                       // YOLOV8模型权重
   │   │   └─ requirements.txt
   │   │   └─scripts
   │   │      └─ LICENSE
   │   │      └─ pub.py                           //发布获取机械臂状态的节点
   │   │      └─ README.md
   │   │      └─ vision_pour_water.py             //视觉抓取demo执行文件
   │   │      └─vi_catch_yolov8.py               //YOLOV8视觉识别运行节点
   │   │   └─src
   │   ├─vi_msgs
   │   │   └─ CMakeLists.txt
   │   │   └─ package.xml
   │   │   └─ include
   │   │      └─ vi_msgs
   │   │   └─msg
   │   │      └─ ObjectInfo.msg                  //视觉识别物体三维坐标的消息类型
   │   │   └─src

\```
```

硬件采用RM65系列机械臂，intel realsense D435相机及因时EG2-4C2两指电动夹爪，机械臂可以是RM65-B带视觉转接板版本，也可以是RM65-B-V视觉版机械臂（如下图，内部走线集成D435相机），主控采用arm或者X86架构的PC均可。

<img src="././pic/视觉臂.png" alt="pic" style="zoom:50%;" />

机械臂通过网线连接主控，相机通过数据线接到主控的USB接口，夹爪由RM特制的末端接口线与机械臂末端连接。RM机械臂末端通信接口是一个6芯的连接器，它为连接到机器人的不同夹持器和传感器提供电源和控制信号。因此可以替换不同的末端执行器，集成时请参考RM机械臂二次开发手册。

<img src="././pic/硬件连接.jpg" alt="pic" style="zoom:100%;" />

## 3.功能包测试环境

### 3.1.ROS noetic 环境

ROS环境安装可以参考：[ROS的最简单安装——鱼香一键安装_鱼香ros一键安装-CSDN博客](https://blog.csdn.net/m0_73745340/article/details/135281023)

### 3.2.Ubuntu20.04系统

本功能包示例使用安装好jatpack的jetson xariver NX，cuda

Jetson NX刷机安装cuda等深度学习环境可以参考[英伟达官方源Jetson Xavier NX安装Ubuntu20.04，配置CUDA，cuDNN，Pytorch等环境教程](https://blog.csdn.net/m0_53717069/article/details/128536837)

安装好基本环境后，在src下打开终端，根据requirements.txt安装python三方库，脚本均基于Ubuntu20系统自带的python3.8开发，可使用conda虚拟环境

```
pip3 install -r requirements.txt
```

### 3.3.YOLOV8 依赖需求

```
pip3 install ultralytics
```

### 3.4.RealSense D435 驱动以及pyrealsense2

1.注册服务器的公钥

```
sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
```


2.将服务器添加到存储库列表中

2.将服务器添加到存储库列表中

```
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
```

3.安装SDK2

```
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
```

4.测试安装结果

```
realsense-viewer
```

<img src="././pic/435测试画面.png" alt="pic" style="zoom:100%;" />

5.安装pyrealsense2

```
python -m pip install --upgrade pip
pip install pyrealsense2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 4.测试步骤

1.首先确认机械臂ROS功能包是否可以正常使用，因为ROS功能包与机械臂的版本存在对应关系，可以进入[realman资料官网](https://develop.realman-robotics.com/robot/summarize/)查看具体版本对应信息。若ROS包存在问题，请下载官网最新的ROS包替换src目录下的rm_robot。

```
sudo apt-get update    # 更新在线包列表
sudo apt-get install python-catkin-tools    # 安装catkin
```

```
mkdir -p rm_work    # 创建功能包
cd rm_work    # 进入工作空间目录
```

将下载好的src解压后放到rm_work目录下

2.编译vi_msgs、rm_msgs

```
catkin build vi_msgs rm_msgs
```

vi_msgs下的ObjectInfo定义的物体信息如下

```
string object_class                //物体类别
float64 x                          //物体距离相机坐标系的X轴值
float64 y                          //物体距离相机坐标系的Y轴值
float64 z                          //物体距离相机坐标系的Z轴值
```

3.功能包整体编译

```
catkin build 
```

4.运行 vi_grab_demo.launch 

 vi_grab_demo.launch  里一共启动了四个节点，分别是msg_pub（主动获取机械臂状态）、robot_driver（机械臂功能启动）、object_detect（视觉识别信息发布）、object_catch（抓取任务脚本）。

```
# 声明环境变量
source devel/setup.bash 
# 运行launch文件
roslaunch vi_grab vi_grab_demo.launch 
```

5.发布需要抓取的物体

打开终端，在终端通过rostopic pub 发布你想抓取的物体名字（使用的是coco模型），例如抓取水瓶就是

```
rostopic pub /choice_object std_msgs/String "bottle"
```

6.机械臂接收到需要抓取物体的信息，开始执行vision_grab.py文件里的运动逻辑

## 5.关键代码解析

视觉功能包是建立了一个ROS节点，用于发布YOLOV8的视觉识别结果，用户可以基于此节点进行多种二次开发，只需要将功能包放入原先的工作空间即可。在RM产品中，我们通过订阅物体信息话题，将物体信息转换到RM机械臂的坐标系下，完成视觉抓取等功能。

视觉识别及信息发布部分：

```
model_path = os.path.join('model', 'yolov8n.pt')
model = YOLO('model_path')  #通过加载不同的模型，使用yolov8的不同模式，例如yolov8n-pose.pt是人体姿态识别模式，yolov8n.pt是普通检测框模式
rospy.init_node("object_detect",anonymous=True)   #建立ROS节点
object_pub = rospy.Publisher("object_pose",ObjectInfo,queue_size=10)   #定义话题发布器
# 循环检测图像流
try:
    while True:
        # 等待获取一对连续的帧：深度和颜色
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()
        if not depth_image.any() or not color_image.any():
            continue
        # 使用 YOLOv8 进行目标检测
        results = model.predict(color_image, conf=0.5)
        detected_boxes = results[0].boxes.xyxy  # 获取边界框坐标
        data = results[0].boxes.data.cpu().tolist()
        canvas = results[0].plot()

        for i, (row, box) in enumerate(zip(data, detected_boxes)):
            id = int(row[5])
            name = results[0].names[id]
            x1, y1, x2, y2 = map(int, box)  # 获取边界框坐标
            # 显示中心点坐标
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)
            dis, camera_coordinate = get_3d_camera_coordinate([ux, uy], aligned_depth_frame, depth_intrin) #得到中心点的深度值，当作距离

            formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f},{camera_coordinate[2]:.2f})"
            # 展示检测界面
            cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
            cv2.putText(canvas, str(formatted_camera_coordinate), (ux + 20, uy + 10), 0, 1,
                        [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            # ROS话题发送物体坐标
            object_info_msg.object_class = str(name)
            object_info_msg.x = float(camera_coordinate[0])
            object_info_msg.y = float(camera_coordinate[1])
            object_info_msg.z = float(camera_coordinate[2])
            rospy.loginfo(object_info_msg)
            object_pub.publish(object_info_msg)

        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                               cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('detection', canvas)
        key = cv2.waitKey(1)
        # 按下 esc 或者 'q' 退出程序和图像界面
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break 
finally:
    # 关闭相机图像流
    pipeline.stop()
```

物体在相机坐标系下的三维坐标转换到机械臂基坐标系下的坐标函数：

```
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
```

等待接收choice_object话题信息，如果一直没有向choice_object话题发布信息，则持续等待。

```
object_msg = rospy.wait_for_message('/choice_object', String, timeout=None)
```

# 常见问题

![image-20241120164150206](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20241120164150206.png)

此错误将导致无法通过话题读取机械臂当前状态信息，因此不可忽略。应当修改网络设置。

例如虚拟机