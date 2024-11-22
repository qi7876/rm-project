import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from fitter import Fitter

# 主文件夹路径
main_folder_path = r"C:\Users\David\Desktop\Data-for-students\UWB-channel-data"

# 不需要的数据索引
excluded_indices = {
    "第一个人": [],
    "第二个人": [1, 2, 3, 4],
    "第三个人": [1, 3, 4],
    "第四个人": [1, 2],
    "第五个人": [1, 2],
    "第六个人": [],
    "第七个人": [4],
    "第八个人": [2]
}

# 用于存储所有有效数据
all_data = []

# 文件名称列表
file_names = [
    "g_norm_1.mat",
    "g_norm_2.mat",
    "g_norm_3.mat",
    "g_norm_4.mat",
    "g_norm_5.mat"
]

# 遍历每个人的文件夹
for person_index, person_folder in enumerate(os.listdir(main_folder_path)):
    person_path = os.path.join(main_folder_path, person_folder)

    if not os.path.isdir(person_path):
        continue  # 如果不是文件夹则跳过
    
    # 计算每个人对应的文件范围
    excluded = excluded_indices.get(person_folder, [])
    
    for file_index in range(len(file_names)):  # 遍历指定的文件名
        if file_index in excluded:
            continue  # 如果该索引被排除，则跳过
        
        file_name = file_names[file_index]
        file_path = os.path.join(person_path, file_name)
        
        if os.path.isfile(file_path):
            # 读取 .mat 文件
            data = scipy.io.loadmat(file_path)
            # 假设信道幅值存储在名为 'g_norm' 的变量中
            if 'g_norm' in data:
                g_norm = data['g_norm'].flatten()  # 展平数据
                # 忽略零点
                filtered_data = g_norm[g_norm != 0]
                all_data.append(filtered_data)  # 将非零数据添加到列表中
            else:
                print(f"{file_name} 中没有找到 'g_norm' 变量。")

# 将所有有效数据合并为一个数组
if all_data:  # 确保有数据可合并
    all_data = np.concatenate(all_data)

    # 使用 fitter 进行概率分布拟合
    f = Fitter(all_data, distributions=['norm', 't', 'laplace'])
    f.fit()

    # 输出拟合结果
    f.summary()

    # 可视化每种分布的拟合结果
    distributions = ['norm', 't', 'laplace']
    for distribution in distributions:
        plt.figure(figsize=(10, 6))
        f.plot_pdf(names=[distribution])  # 仅绘制指定的分布
        plt.title(f"Probability Density Function Fitting - {distribution.capitalize()}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid()
        plt.show()
else:
    print("没有有效的数据可以进行拟合。")
