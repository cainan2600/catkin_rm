import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def plot_correct_and_incorrct(filename):
    # 文件名，可替换为实际使用的文件名
    filename_CORRECT_chasis = "{}/CORRECT_chasis.txt".format(filename)
    filename_CORRECT_obj = "{}/CORRECT_obj.txt".format(filename)
    filename_INCORRECT_chasis = "{}/INCORRECT_chasis.txt".format(filename)
    filename_INCORRECT_obj = "{}/INCORRECT_obj.txt".format(filename)

    # print(filename_CORRECT_chasis)
    with open(filename_CORRECT_chasis, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        len_lines = len(lines) - 1
    num_data = np.random.randint(0, len_lines) +1
    print("num_data", num_data)
    # print(len_lines)
    # num_data = 1
    

    # 用于存储解析后的数据
    data_co_obj = []
    x_all = []
    y_all = []
    # current_row = 0
    line_num = 0
    with open(filename_CORRECT_obj, 'r') as file:
        # print("1111111111111111111")
        lines = file.readlines()
        for line in lines:
            line_num += 1
            # print(line_num)
            if 7 * (num_data -1) + 1 <= line_num < 7 * (num_data) + 1:
                line = line.strip()
                # print('111111111111111111111111')
                if line:
                    parts = line.split(' ')
                    x = float(parts[3])
                    y = float(parts[4])
                    # print(x, y)
                    x_all.append(x)
                    y_all.append(y)

    data_co_obj.append([x_all, y_all])
            # else:
            #     pass

    # 用于存储解析后的数据
    data_co_chasis = []
    x_all = []
    y_all = []
    # current_row = 0
    line_num = 0
    with open(filename_CORRECT_chasis, 'r') as file:
        # print("1111111111111111111")
        lines = file.readlines()
        for line in lines:
            line_num += 1
            # print(line_num)
            if line_num == num_data:
                line = line.strip()
                # print('111111111111111111111111')
                if line:
                    parts = line.split(' ')
                    x = float(parts[3])
                    y = float(parts[4])
                    x_all.append(x)
                    y_all.append(y)

    data_co_chasis.append([x_all, y_all])
            # else:
            #     pass

    # 用于存储解析后的数据
    data_inco_obj = []
    x_all = []
    y_all = []
    # current_row = 0
    line_num = 0
    with open(filename_INCORRECT_obj, 'r') as file:
        # print("1111111111111111111")
        lines = file.readlines()
        for line in lines:
            line_num += 1
            # print(line_num)
            if 7 * (num_data -1) + 1 <= line_num < 7 * (num_data) + 1:
                line = line.strip()
                # print('111111111111111111111111')
                if line:
                    parts = line.split(' ')
                    x = float(parts[0])
                    y = float(parts[1])
                    x_all.append(x)
                    y_all.append(y)

    data_inco_obj.append([x_all, y_all])
            # else:
            #     pass

    # 用于存储解析后的数据
    data_inco_chasis = []
    x_all = []
    y_all = []
    # current_row = 0
    line_num = 0
    with open(filename_INCORRECT_chasis, 'r') as file:
        # print("1111111111111111111")
        lines = file.readlines()
        for line in lines:
            line_num += 1
            # print(line_num)
            if line_num == num_data:
                line = line.strip()
                # print('111111111111111111111111')
                if line:
                    parts = line.split(' ')
                    x = float(parts[3])
                    y = float(parts[4])
                    x_all.append(x)
                    y_all.append(y)

    data_inco_chasis.append([x_all, y_all])
            # else:
            #     pass



    # # 将数据转换为DataFrame格式
    # df = pd.DataFrame(data_co_obj, columns=['x', 'y'])
    # # dff = pd.DataFrame(data_co_chasis, columns=['x', 'y'])

    # # # 使用pandas绘制散点图
    # ax = df.plot.scatter(x='x', y='y', c='blue', label='Data Points')
    # # ax = dff.plot.scatter(x='x', y='y', c='red', label='Data_chasis Points')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('Scatter Plot from 1400 data')
    # plt.plot([-0.25, 1.75, 1.75, -0.25, -0.25], [0.803, 0.803, 1.653, 1.653, 0.803], 'r')
    # # plt.plot([-1.4, 1.4, 1.4, -1.4, -1.4], [-0.825 -0.825, 0.825, 0.825, -0.825], 'r')


    # fig = ax.get_figure()
    # fig.savefig('{}/pic_incorr-corr.png'.format(filename))

    # plt.show()

    print(data_inco_obj[0],data_inco_obj[0])

    plt.figure()
    plt.plot([-0.25, 1.75, 1.75, -0.25, -0.25], [0.803, 0.803, 1.653, 1.653, 0.803], 'r')
    
    plt.scatter(data_co_obj[0][0], data_co_obj[0][1], c='y', label='data_co_obj')
    plt.scatter(data_co_chasis[0][0], data_co_chasis[0][1], c='k', label='data_co_chasis')
    plt.scatter(data_inco_obj[0][0], data_inco_obj[0][1], c='b', label='data_inco_obj')
    plt.scatter(data_inco_chasis[0][0], data_inco_chasis[0][1], c='g', label='data_inco_chasis')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Last-Training-epoc')
    plt.legend()

    plt.savefig('{}/pic_incorr-corr-1.png'.format(filename))
    plt.show()



if __name__ == "__main__":
    filename = "/home/cn/catkin_rm/src/RPSN_4/work_dir/test01-1-output5-1111111"
    plot_correct_and_incorrct(filename)


