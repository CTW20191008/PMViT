import matplotlib.pyplot as plt
import numpy as np


loss_file = '/home/yons/disk/zhuhao/ViT_P/results/train/loss_pretrain_bvit_10_6_20_flip_10_mask'
# 存储每行的 loss 值
loss_values = []

# 打开文件
with open(loss_file, 'r') as file:
    # 逐行读取文件内容
    for line in file:
        if line.find('[0/5004]') != -1:
            # 寻找每行中的 loss 字段
            index = line.find('loss')
            if index != -1:
                # 找到 loss 字段后面的数值
                start_index = line.find(' ', index) + 1
                end_index = line.find(' ', start_index)
                if end_index == -1:
                    end_index = len(line)
                loss_value = float(line[start_index:end_index])  # 将字符串转换为浮点数
                loss_values.append(loss_value)

# # 归一化处理
# # loss_values = [(x - min(loss_values)) / (max(loss_values) - min(loss_values)) for x in loss_values]
# loss_values = [x / max(loss_values) for x in loss_values]
# print(min(loss_values))

# # 计算均值
# mean_value = np.mean(loss_values)

# # 计算方差
# variance_value = np.var(loss_values)

# print("均值:", mean_value)
# print("方差:", variance_value)

# loss_values = [(x-mean_value) / variance_value for x in loss_values]

# 绘制图表
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.show()
