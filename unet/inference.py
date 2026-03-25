# unet/inference.py
import torch
import cv2
import numpy as np
from unet.model import AttentionUNet   # 你的 Attention UNet
from unet.utils import CLASS_COLORS   # 颜色映射

def predict_and_visualize(image_path, model_path="/data/unet-attention-dsconv_github/unet/model_save/unet_atten_dsconv_best.pth", output_vis_path="/data/unet-attention-dsconv_github/unet/inference_jpg/inference_output.jpg"):
    """
    1. 预测 mask (shape: H,W  值0~8)
    2. 生成彩色 overlay原图 + 半透明颜色填充患病区域
    3. 保存可视化图像，返回路径（供 LangChain 使用）
    """
    # 加载模型
    model = AttentionUNet(n_classes=9).cuda()  # 9类 (0背景 + 8病变)
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()

    # 读取原图
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    h, w = original.shape[:2]

    # 预处理为模型输入 (640x640)
    img = cv2.resize(original, (640, 640))
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0

    # 推理
    with torch.no_grad():
        pred = model(tensor)                    # [1,9,640,640]
        mask = torch.argmax(pred, dim=1)[0]     # [640,640] 值0~8
        mask = mask.cpu().numpy().astype(np.uint8)

    # 恢复原尺寸
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 生成彩色 overlay
    overlay = original.copy()
    color_mask = np.zeros_like(original, dtype=np.uint8)

    for cls_id in range(0, 8):  # 只填充病变区域
        cls_pixels = (mask == cls_id)
        color = CLASS_COLORS[cls_id]
        color_mask[cls_pixels] = color

    # 半透明融合 (alpha=0.4)
    cv2.addWeighted(color_mask, 0.5, overlay, 0.5, 0, overlay)

    # 保存可视化结果
    vis_img = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_vis_path, vis_img)

    # 返回 mask（用于统计）和可视化路径
    return mask, output_vis_path

if __name__ == "__main__":
    image_path = "/data/unet-attention-dsconv_github/data/train/6_JPG.rf.7f22a52ca57bf0287362001a6a74a7be.jpg"
    try:
        mask, vis_path = predict_and_visualize(image_path)
        print(f"Mask: {mask.shape},唯一值:{np.unique(mask)},save_path:{vis_path}")
    except Exception as e:
        print(f"出现错误: {e}")