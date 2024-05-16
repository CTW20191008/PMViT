import os
import random
import shutil

# 源文件夹路径和目标文件夹路径
source_folder = '/home/yons/dataset/all/val'
target_folder = source_folder.replace('all', 'all_1_10')

if not os.path.exists(target_folder):
    os.mkdir(target_folder)

# 获取源文件夹下所有子文件夹的路径
subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

for index, folder in enumerate(subfolders):
    category = folder.split('/')[-1]
    new_folder = f"{target_folder}/{category}"
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    # 获取当前子文件夹下所有图片文件的路径
    image_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.JPEG')]

    # 计算需要提取的图片数量
    num_images_to_copy = len(image_files) // 10

    # 随机选择1/10数量的图片文件
    selected_images = random.sample(image_files, num_images_to_copy)

    # 将选中的图片复制到目标文件夹
    for image in selected_images:
        shutil.copy(image, new_folder)

    print(f"[INFO]: finished {index+1} category - {category}")