import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_mask_png(mask_path):
    """
    测试读取mask.png并分析关键信息
    :param mask_path: 掩码图片的路径（如 /data/xxx/mask.png）
    """
    # 1. 基础读取 单通道灰度模式
    # cv2.IMREAD_GRAYSCALE 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f" 读取失败！请检查路径是否正确：{mask_path}")
        print(f"   建议检查：1. 文件是否存在  2. 路径是否有拼写错误  3. 文件是否损坏")
        return
    
    # 2. 输出掩码核心信息
    print(" 掩码读取成功！")
    print(f" 掩码尺寸（高, 宽）：{mask.shape}")  # 应和原图尺寸一致
    print(f" 数据类型：{mask.dtype}")  # 应为 uint8
    print(f" 掩码中的像素值（类别ID）：{np.unique(mask)}")  # 显示所有类别ID
    print(f" 各类别像素数量：")
    for val in np.unique(mask):
        count = np.sum(mask == val)
        print(f"   类别ID {val}: {count} 个像素")
    
    # 3. 可视化掩码
    plt.figure(figsize=(10, 8))
    
    # 子图1：原始灰度掩码
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('原始灰度掩码', fontsize=12)
    plt.axis('off')
    
    # 子图2：彩色可视化掩码（不同类别用不同颜色，更易区分）
    plt.subplot(1, 2, 2)
    # 自定义颜色映射（可根据你的类别数扩展）
    color_map = {
        0: [0, 0, 0],       # 背景 - 黑色
        1: [255, 0, 0],     # 类别1 - 红色
        2: [0, 255, 0],     # 类别2 - 绿色
        3: [0, 0, 255],     # 类别3 - 蓝色
        4: [255, 255, 0],   # 类别4 - 黄色
        5: [255, 0, 255],   # 类别5 - 洋红
        6: [0, 255, 255],   # 类别6 - 青色
        7: [128, 0, 0],     # 类别7 - 深红
        8: [0, 128, 0]      # 类别8 - 深绿
    }
    # 转换为彩色图
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for val in np.unique(mask):
        mask_color[mask == val] = color_map.get(val, [128, 128, 128])  # 未知类别用灰色
    
    plt.imshow(mask_color)
    plt.title('彩色可视化掩码', fontsize=12)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 路径
    MASK_PATH = "/data/unet-attention-dsconv_github/data/train_mask/6_JPG.rf.7f22a52ca57bf0287362001a6a74a7be.png"
    test_mask_png(MASK_PATH)
