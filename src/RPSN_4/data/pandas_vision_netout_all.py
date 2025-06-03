import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import shutil


matplotlib.use("Agg")
# 文件名，可替换为实际使用的文件名
def pandas_plt(dir):
    filename = "{}.txt".format(dir)

    # 用于存储解析后的数据
    data = []
    current_row = 0
    line_num = 0
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_num += 1
            if not line_num == 11200:
                line = line.strip()
                if line:  # 只处理非空行
                    parts = line.split(' ')
                    x = float(parts[1])
                    y = float(parts[2])
                    data.append([x, y])
                    current_row += 1
                    if current_row == 7:  # 每处理7行数据后，下一行是空格行，直接跳过
                        current_row = 0
                        continue
            else:
                break


    # 将数据转换为DataFrame格式
    df = pd.DataFrame(data, columns=['x', 'y'])

    # # 使用pandas绘制散点图
    ax = df.plot.scatter(x='x', y='y', c='blue', label='Data Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter Plot from 1400 data')
    # plt.plot([-0.25, 1.75, 1.75, -0.25, -0.25], [0.803, 0.803, 1.653, 1.653, 0.803], 'r')
    plt.plot([-1, 1, 1, -1, -1], [-0.425, -0.425, 0.425, 0.425, -0.425], 'r')


    # theta = np.linspace(0, 2 * np.pi, 1000)  # 生成1000个角度采样点
    # r = 0.44  # 指定半径
    # # x = r * np.cos(theta) + 0.75
    # # y = r * np.sin(theta) + 1.228
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)

    # plt.plot(x, y, 'r-',  label='r=0.44 circle')
    # # # plt.plot([-1.4, 1.4, 1.4, -1.4, -1.4], [-0.825 -0.825, 0.825, 0.825, -0.825], 'r')


    fig = ax.get_figure()
    fig.savefig('{}.png'.format(dir))

    # plt.show()

if __name__ == "__main__":

    source_dir = "/home/cn/catkin_rm/src/RPSN_4/work_dir/squre/test02-12-0.0ori-7random-7copy-10chasisloss"

    for i in range(1, 51):
        if  i % 1 == 0:
            # print("{}/{}/NET_output".format(source_dir,i))
            pandas_plt("{}/{}/NET_output".format(source_dir,i))
            # pandas_plt("{}/{}/NET_output_test".format(source_dir,i))


    # 目标目录路径（与源目录相同）
    target_dir = source_dir
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 生成200以内（包含200）除0以外的2的倍数列表
    valid_folders = {str(num) for num in range(1, 51, 1)}
    
    # 遍历源目录下的所有条目
    for entry in os.scandir(source_dir):
        # 检查是否为目录且名称是有效的文件夹名
        if entry.is_dir() and entry.name in valid_folders:
            folder_name = entry.name
            image_path = os.path.join(entry.path, "NET_output.png")
            # image_path = os.path.join(entry.path, "NET_output_test.png")
            
            # 检查图片文件是否存在
            if os.path.isfile(image_path):
                # 新文件名：文件夹名 + .png
                new_filename = f"{folder_name}.png"
                target_path = os.path.join(target_dir, new_filename)
                
                try:
                    # 移动并重命名图片
                    shutil.move(image_path, target_path)
                    print(f"已将 {image_path} 重命名并移动至 {target_path}")
                except Exception as e:
                    print(f"处理 {image_path} 时出错: {e}", file=sys.stderr)
            else:
                print(f"警告: 文件夹 {folder_name} 中未找到 NET_output.png 文件", file=sys.stderr)
    
    print("处理完成!")
