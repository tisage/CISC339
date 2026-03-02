import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import ast
import os
import glob
import json

def get_device():
    """选择最佳硬件加速设备"""
    if torch.backends.mps.is_available():
        print("硬件: Apple Silicon (MPS) 加速已启用")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("硬件: CUDA 加速已启用")
        return torch.device("cuda")
    else:
        print("硬件: 使用 CPU 运行")
        return torch.device("cpu")

def main():
    device = get_device()
    
    # 1. 载入 Donut 处理器和模型
    # 这个模型特别适合直接将收据/文档照片直接变成结构化 JSON
    model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
    print(f"正在从 HuggingFace 载入模型 {model_id}...")
    print("初次运行可能会下载模型权重（约 800MB），请耐心等待。")
    
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    
    # 将模型移动到加速硬件
    model.to(device)
    model.eval()

    # 2. 搜索要处理的图片 (支持多个图片，比如 sample_1.png, sample_2.jpg 等)
    search_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = []
    
    # 寻找目录下所有的常见图片格式
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(search_dir, ext)))
        
    if not image_paths:
        print(f"在 {search_dir} 没有找到任何图片！")
        print("演示模式：自动生成一张 sample_document.jpg 来确保代码走通...")
        demo_image = os.path.join(search_dir, "sample_document.jpg")
        Image.new('RGB', (800, 600), color='white').save(demo_image)
        image_paths = [demo_image]
        
    print(f"\n共找到 {len(image_paths)} 张图片准备进行文档分析。")
    
    # 提前准备模型所需的特定 Prompt
    task_prompt = "<s_cord-v2>" # cord-v2 数据集的特殊提示词
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 3. 循环处理每一张图片
    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        print("\n" + "="*50)
        print(f"正在分析图片: [{file_name}] ...")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"跳过图片，文件损坏或无法读取: {e}")
            continue

        # 图像预处理
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # 运行模型生成结构化文档 (JSON格式)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # 解码与后处理
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        # 提取出整理好的 Python 字典 (JSON数据结构)
        organized_data = processor.token2json(sequence)
        
        # 4. 生成同名的 JSON 或 Text 文件
        # 例如: sample_1.png -> sample_1.json
        base_name, _ = os.path.splitext(image_path)
        output_file_path = f"{base_name}.json"
        
        # 写入文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ 处理完成！特征提取内容已自动写入到了：\n---> {output_file_path}")
        print("以下为内容预览：")
        print(json.dumps(organized_data, indent=4, ensure_ascii=False))

    print("\n所有文档图片识别结束！")

if __name__ == "__main__":
    main()
