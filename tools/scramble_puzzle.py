import numpy as np
import cv2
import random

# 读取原始图片
image_size = 560 # 224
image = cv2.imread('D:\\algo\\ghost\\PViT\\test_001.jpg')
image = cv2.resize(image, (image_size, image_size))  # 将图片调整为224*224大小 

# 将图片切成196个14*14大小的图片块
stride = 56 # 14
image_blocks = [image[i:i+stride, j:j+stride] for i in range(0, image_size, stride) for j in range(0, image_size, stride)]

# 随机翻转图片块
image_blocks_new = []
for block in image_blocks:
    random_number = random.randint(0, 3)
    if random_number == 0:
        print("[INFO]: Do nothing")
    elif random_number == 1:
        print("[INFO]: Do Back")
        block = block.copy()[::-1, ::-1, :]
    elif random_number == 2:
        print("[INFO]: Do Left")
        block = cv2.transpose(block)[::-1]
    elif random_number == 3:
        print("[INFO]: Do Right")
        block = np.flip(cv2.transpose(block), 1)
    else:
        print(f"[INFO]: Random number is {random_number}, not supported")
    image_blocks_new.append(block)

random.shuffle(image_blocks_new)

# 将翻转后的图片块拼接成224*224大小的图片
reconstructed_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
index = 0
for i in range(0, image_size, stride):
    for j in range(0, image_size, stride):
        reconstructed_image[i:i+stride, j:j+stride] = image_blocks_new[index]
        index += 1

# 显示拼接后的图片
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
