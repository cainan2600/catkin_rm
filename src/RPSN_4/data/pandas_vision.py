import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")
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
                    # x = float(parts[1])
                    # y = float(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
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
    ax.set_title('Scatter Plot from 1000 data')
    # plt.plot([-0.25, 1.75, 1.75, -0.25, -0.25], [0.803, 0.803, 1.653, 1.653, 0.803], 'r')
    plt.plot([-1, 1, 1, -1, -1], [-0.425, -0.425, 0.425, 0.425, -0.425], 'r')


    # theta = np.linspace(0, 2 * np.pi, 1000)  # 生成1000个角度采样点
    # r = 0.44  # 指定半径
    # # x = r * np.cos(theta) + 0.75
    # # y = r * np.sin(theta) + 1.228
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)

    # plt.plot(x, y, 'r-',  label='r=0.44 circle')
    # plt.plot([-1.4, 1.4, 1.4, -1.4, -1.4], [-0.825 -0.825, 0.825, 0.825, -0.825], 'r')


    fig = ax.get_figure()
    fig.savefig('{}.png'.format(dir))

    plt.show()

if __name__ == "__main__":
    pandas_plt("/home/cn/catkin_rm/src/RPSN_4/data/data_cainan/rm-fk-ik-all-random-with-dipan-norm-squre-2000data/train-2000/train_dataset_dipan_2000")


