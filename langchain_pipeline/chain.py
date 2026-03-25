from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64
import numpy as np
import os
import sys
sys.path.append("/data/unet-attention-dsconv_github")
from unet.inference import predict_and_visualize

# vLLM OpenAI-compatible client (你服务器地址)
llm = ChatOpenAI(
    model="/data/qwen3.5_9b_huatuo",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    max_retries=0
)
CLASS_NAMES = [
    "Background",
    "Brain SOL For assessment",
    "Craniopharyngioma",
    "Extradural-hemorrhage",
    "Intraparenchymal-hemorrhage",
    "Intraventricular-hemorrhage",
    "Meningioma",
    "Subarachnoid hemorrhage",
    "Subdural-hemorrhage"
]

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
        
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是脑部MRI辅助分析专家(仅供学术演示,不用于临床诊断)。请客观描述，不要下诊断结论，末尾必须加“请专业医师复核”。"),
    ("human", [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}},
        {"type": "text", "text": """
    彩色标注（不同颜色代表不同患病类型,自动除去图片周围噪声,只专注脑部中病变颜色区域）已提供。
    患病区域统计：
    {features}
    要求：
    1. 参考彩色区域的位置、大小、颜色对应关系进行描述
    2. 如果未检测到明显病灶，请明确说明
    3. 描述可能受影响的脑区（仅参考）
    4. 保持客观、谨慎，报告格式清晰"""}
 ])
])

chain = prompt | llm | StrOutputParser()
def generate_report(mask: np.ndarray, visualized_path: str) -> str:
    # 统计特征
    features = "检测到的患病区域总结：\n"
    total_pixels = np.prod(mask.shape)
    has_lesion = False
    for cls_id in range(1, 9):
        pixels = np.sum(mask == cls_id)
        if pixels > 100:
            has_lesion = True
            ratio = pixels / total_pixels * 100
            features += f"- {CLASS_NAMES[cls_id]}：约 {ratio:.2f}% 面积\n"
    if not has_lesion:
        features += "- 未检测到明显患病区域\n"

    # 转 base64
    image_base64 = image_to_base64(visualized_path)

    # 直接调用 chain（这就是你想要的“使用chain”）
    report = chain.invoke({
        "image_base64": image_base64,
        "features": features
    })
    return report

if __name__ == "__main__":

    test_img = "/data/unet-attention-dsconv_github/data/train/6_JPG.rf.7f22a52ca57bf0287362001a6a74a7be.jpg"
    
    mask, vis_path = predict_and_visualize(test_img)
    report = generate_report(mask, vis_path)
    print("=== 生成报告 ===")
    print(report)
