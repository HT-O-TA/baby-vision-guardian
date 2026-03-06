import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化 MTCNN 检测器，keep_all=True 表示检测图片中所有的人脸
mtcnn = MTCNN(keep_all=True, device=device)

def detect_and_crop_faces(input_dir, output_dir):
    """
    对输入文件夹中的每张图片检测人脸，并按全局递增序号保存到输出文件夹
    Args:
        input_dir: 输入图片文件夹路径
        output_dir: 裁剪后人脸保存文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 全局人脸计数器（从1开始）
    face_counter = 1
    
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法打开图片 {img_name}: {e}")
            continue
        
        # 检测图片中的所有人脸
        boxes, probs = mtcnn.detect(img)
        
        if boxes is None:
            print(f"未检测到人脸：{img_name}")
            continue
        
        # 遍历所有检测结果，过滤较小的人脸区域
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # 判断人脸大小是否符合要求
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # 裁剪人脸区域
            face = img.crop((x1, y1, x2, y2))
            
            # 使用全局计数器生成序号
            output_path = os.path.join(output_dir, f"face_{face_counter}.jpg")
            face.save(output_path)
            print(f"保存裁剪后的人脸到: {output_path}")
            
            # 计数器递增
            face_counter += 1

# 示例：指定输入输出目录
input_directory = "input_faces"        # 替换为你的图片文件夹路径
output_directory = "output_faces"      # 替换为你希望保存裁剪人脸的文件夹路径

detect_and_crop_faces(input_directory, output_directory)
