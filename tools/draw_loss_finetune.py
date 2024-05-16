import matplotlib.pyplot as plt


loss_file = '/home/yons/disk/zhuhao/ViT_P/results/train/result_pvit_finetune_10_6_20_10'
# 存储每行的 loss 值
train_loss_values = []
test_loss_values = []

# 打开文件
with open(loss_file, 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 寻找每行中的 loss 字段
        if line.find('[0/5004]') != -1:  # 'Train'
            index = line.find('loss')
            if index != -1:
                # 找到 loss 字段后面的数值
                start_index = line.find(' ', index) + 1
                end_index = line.find(' ', start_index)
                if end_index == -1:
                    end_index = len(line)
                loss_value = float(line[start_index:end_index])  # 将字符串转换为浮点数
                train_loss_values.append(loss_value)
        elif line.find('[0/196]') != -1:    # 'Test'
            index = line.find('Loss')
            if index != -1:
                # 找到 loss 字段后面的数值
                start_index = line.find(' ', index) + 1
                end_index = line.find(' ', start_index)
                if end_index == -1:
                    end_index = len(line)
                loss_value = float(line[start_index:end_index])  # 将字符串转换为浮点数
                test_loss_values.append(loss_value)

print(f"[INFO]: train_loss_values is {train_loss_values}")
print(f"[INFO]: test_loss_values is {test_loss_values}")

# 绘制图表
plt.plot(train_loss_values, label='Train Loss')
plt.plot(test_loss_values, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Loss Values')
plt.legend()
plt.show()

