import gradio as gr
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8001/brain-detect"   # 确认这个能 curl 通

def upload_and_detect(file):
    if file is None:
        return "请上传图片", None, None

    try:
        with open(file, "rb") as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(API_URL, files=files, timeout=90)

        print("状态码:", response.status_code)
        print("响应头 Content-Type:", response.headers.get('content-type'))
        print("原始响应文本:", response.text[:1000])  # 截断避免终端爆炸

        if response.status_code != 200:
            return f"API 返回 {response.status_code}\n{response.text}", None, None

        try:
            result = response.json()
            print("解析后的 JSON:", result)
        except:
            return "后端返回的不是合法 JSON！\n" + response.text, None, None

        report = result.get("report", "[无报告字段]")
        vis = result.get("visualized_image", None)

        return report, vis, vis

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("异常:", err)
        return f"请求/处理异常:\n{str(e)}\n{err}", None, None


demo = gr.Interface(
    fn=upload_and_detect,
    inputs=gr.Image(type="filepath", label="上传脑部MRI图片"),
    outputs=[
        gr.Textbox(label="AI辅助报告"),
        gr.Image(label="可视化结果（彩色标注）"),
        gr.Image(label="可视化结果（放大查看）")
    ],
    title="脑部MRI辅助检测演示(学术用途，仅供参考)",
    description="上传脑部MRI图像，AI进行病灶分割并生成辅助报告。注意：本系统仅学术演示，不具备临床诊断能力。",
)

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,        
    share=False,
    allowed_paths=[
        "/data/unet-attention-dsconv_github/unet/inference_jpg",
        "/data/unet-attention-dsconv_github/backend/temp",
        "/tmp",
        "/data"                                                    
    ]
)