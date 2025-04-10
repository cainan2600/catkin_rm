import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# 文件名，可替换为实际使用的文件名
filename = "/home/cn/catkin_rm/src/RPSN_4/work_dir/test01-2same-weight-loss/NET_output.txt"

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
plt.plot([-0.25, 1.75, 1.75, -0.25, -0.25], [0.803, 0.803, 1.653, 1.653, 0.803], 'r')
# plt.plot([-1.4, 1.4, 1.4, -1.4, -1.4], [-0.825 -0.825, 0.825, 0.825, -0.825], 'r')


fig = ax.get_figure()
fig.savefig('/home/cn/catkin_rm/src/RPSN_4/work_dir/test01-2same-weight-loss/NET_output.png')

plt.show()


