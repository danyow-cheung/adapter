import cv2
import numpy as np

def create_black_mask(image_path):
    # 读取输入图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 创建白色蒙版
    white_mask = np.ones_like(image) * 255

    # 将非白色区域替换为黑色
    black_mask = cv2.bitwise_not(cv2.bitwise_xor(image, white_mask))
    # 保存黑色蒙版
    cv2.imwrite("black_mask.png", black_mask)

# 调用函数并传入图像路径
create_black_mask("image.jpg")