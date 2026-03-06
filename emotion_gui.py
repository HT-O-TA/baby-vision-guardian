import sys
import os
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facenet_pytorch import MTCNN
import torch

class EmotionAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("婴幼儿表情识别系统")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        

        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化人脸检测器
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # 创建界面
        self.create_widgets()
        
        # 当前图像路径和处理后的图像
        self.current_image_path = None
        self.processed_image = None
        self.emotion_scores = None
        self.face_detected = False
        
    def create_widgets(self):
        # 顶部标题
        header_frame = tk.Frame(self.root, bg="#4a7abc", height=60)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame, 
            text="婴幼儿表情识别系统", 
            font=("Arial", 20, "bold"), 
            bg="#4a7abc", 
            fg="white"
        )
        title_label.pack(pady=10)
        
        # 主内容区域
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左侧：图像显示区域
        self.image_frame = tk.Frame(content_frame, bg="white", width=500, height=500)
        self.image_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.prompt_label = tk.Label(
            self.image_frame, 
            text="请点击\"选择图片\"按钮上传婴幼儿面部照片", 
            font=("Arial", 12), 
            bg="white"
        )
        self.prompt_label.pack(pady=5)
        
        # 右侧：控制和结果显示区域
        right_frame = tk.Frame(content_frame, bg="#f0f0f0", width=500)
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        # 按钮区域
        control_frame = tk.Frame(right_frame, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=10)
        
        self.upload_btn = tk.Button(
            control_frame, 
            text="选择图片", 
            font=("Arial", 12),
            command=self.upload_image,
            bg="#4a7abc",
            fg="white",
            padx=20,
            pady=10
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(
            control_frame, 
            text="分析表情", 
            font=("Arial", 12),
            command=self.analyze_emotion,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # 情绪分数显示区域
        self.result_frame = tk.Frame(right_frame, bg="white", padx=10, pady=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 用于显示情绪条形图的区域
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 结果标签
        self.result_label = tk.Label(
            right_frame, 
            text="等待分析...", 
            font=("Arial", 14, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        self.result_label.pack(fill=tk.X, pady=10)
        
        # 状态栏
        self.status_bar = tk.Label(
            self.root, 
            text="就绪", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """处理图片上传"""
        filetypes = [
            ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("所有文件", "*.*")
        ]
        
        self.current_image_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=filetypes
        )
        
        if self.current_image_path:
            self.status_bar.config(text=f"已加载: {os.path.basename(self.current_image_path)}")
            self.load_and_resize_image()
            self.analyze_btn.config(state=tk.NORMAL)
            self.prompt_label.config(text="点击\"分析表情\"按钮开始分析")
    
    def load_and_resize_image(self):
        """加载并调整图像尺寸以适应显示区域"""
        try:
            # 使用PIL打开图像
            image = Image.open(self.current_image_path)
            
            # 获取图像标签尺寸
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            
            # 如果标签尚未调整（初始加载时），使用默认尺寸
            if label_width < 10:
                label_width = 400
                label_height = 400
            
            # 调整图像大小以适应标签
            img_width, img_height = image.size
            ratio = min(label_width/img_width, label_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter可用的格式
            tk_image = ImageTk.PhotoImage(image)
            
            # 更新图像标签
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # 保持引用以防止垃圾回收
            
            # 重置分析状态
            self.face_detected = False
            self.emotion_scores = None
            self.clear_emotion_chart()
            self.result_label.config(text="等待分析...")
            
        except Exception as e:
            self.status_bar.config(text=f"错误: {str(e)}")
    
    def detect_face(self):
        """检测图像中的人脸"""
        try:
            # 读取图像
            img = Image.open(self.current_image_path).convert('RGB')
            img_np = np.array(img)
            
            # 检测人脸
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                self.status_bar.config(text="未检测到人脸，请尝试其他图片")
                self.face_detected = False
                return False
            
            # 获取置信度最高的人脸
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            
            # 转换坐标为整数
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # 在图像上绘制人脸框
            img_with_box = img_np.copy()
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 转换回PIL图像并显示
            img_with_box_pil = Image.fromarray(img_with_box)
            
            # 调整大小以适应标签
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            
            # 如果标签尚未调整（初始加载时），使用默认尺寸
            if label_width < 10:
                label_width = 400
                label_height = 400
            
            # 调整图像大小以适应标签
            img_width, img_height = img_with_box_pil.size
            ratio = min(label_width/img_width, label_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            img_with_box_pil = img_with_box_pil.resize((new_width, new_height), Image.LANCZOS)
            
            # 更新图像显示
            tk_image = ImageTk.PhotoImage(img_with_box_pil)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image
            
            self.face_detected = True
            self.status_bar.config(text=f"人脸检测成功")
            return True
            
        except Exception as e:
            self.status_bar.config(text=f"人脸检测错误: {str(e)}")
            self.face_detected = False
            return False
    
    def analyze_emotion(self):
        """分析面部表情和情绪"""
        if not self.current_image_path:
            self.status_bar.config(text="请先选择图片")
            return
        
        self.status_bar.config(text="正在分析...")
        self.prompt_label.config(text="正在分析中，请稍候...")
        self.root.update()
        
        try:
            # 先进行人脸检测
            if not self.detect_face():
                self.result_label.config(text="未检测到人脸，请尝试其他图片")
                return
            
            # 使用DeepFace分析情绪
            result = DeepFace.analyze(
                self.current_image_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='retinaface'
            )
            
            # 获取情绪得分
            emotions = result[0]['emotion']
            
            # 保存情绪得分
            self.emotion_scores = {
                'angry': emotions.get('angry', 0),
                'disgust': emotions.get('disgust', 0),
                'fear': emotions.get('fear', 0),
                'happy': emotions.get('happy', 0),
                'sad': emotions.get('sad', 0),
                'surprise': emotions.get('surprise', 0),
                'neutral': emotions.get('neutral', 0)
            }
            
            # 找出得分最高的情绪
            dominant_emotion = max(self.emotion_scores, key=self.emotion_scores.get)
            dominant_score = self.emotion_scores[dominant_emotion]
            
            # 计算二级分类结果
            negative_score = self.emotion_scores['angry'] + self.emotion_scores['disgust'] + self.emotion_scores['sad']
            if self.emotion_scores['fear'] >= 15:
                negative_score += self.emotion_scores['fear'] * 0.6
                
            positive_score = self.emotion_scores['happy'] + self.emotion_scores['surprise']
            neutral_score = self.emotion_scores['neutral']
            
            # 确定二级分类结果
            if negative_score >= 45 and negative_score > positive_score + 5 and negative_score > neutral_score + 5:
                secondary_emotion = "负面情绪 (negative)"
            elif positive_score >= 25 and positive_score > neutral_score:
                secondary_emotion = "正面情绪 (positive)"
            else:
                secondary_emotion = "中性情绪 (neutral)"
            
            # 更新结果标签
            result_text = f"主要情绪: {self.translate_emotion(dominant_emotion)} ({dominant_score:.1f}%)\n"
            result_text += f"情绪分类: {secondary_emotion}"
            self.result_label.config(text=result_text)
            
            # 更新图表
            self.update_emotion_chart()
            
            self.status_bar.config(text="分析完成")
            self.prompt_label.config(text="分析完成")
            
        except Exception as e:
            self.status_bar.config(text=f"分析错误: {str(e)}")
            self.result_label.config(text="分析失败，请重试")
    
    def translate_emotion(self, emotion):
        """将英文情绪翻译为中文"""
        translations = {
            'angry': '愤怒',
            'disgust': '厌恶',
            'fear': '恐惧',
            'happy': '高兴',
            'sad': '伤心',
            'surprise': '惊讶',
            'neutral': '中性'
        }
        return translations.get(emotion, emotion)
    
    def update_emotion_chart(self):
        """更新情绪条形图"""
        if not self.emotion_scores:
            return
        
        # 清除当前图表
        self.ax.clear()
        
        # 准备数据
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        # 使用英文标签而不是中文
        emotions_display = [e.capitalize() for e in emotions]
        scores = [self.emotion_scores[e] for e in emotions]
        
        # 设置条形图颜色
        colors = ['#FF4500', '#9932CC', '#4B0082', '#FFD700', '#4682B4', '#32CD32', '#A9A9A9']
        
        # 创建条形图
        bars = self.ax.bar(emotions_display, scores, color=colors)
        
        # 为每个条形添加数据标签
        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
        
        # 设置图表标题和标签 - 使用英文
        self.ax.set_title('Emotion Analysis Results')
        self.ax.set_ylabel('Score (%)')
        self.ax.set_ylim(0, 100)
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45)
        
        # 自动调整布局
        self.fig.tight_layout()
        
        # 更新画布
        self.canvas.draw()
    
    def clear_emotion_chart(self):
        """清除情绪图表"""
        self.ax.clear()
        self.ax.set_title('Waiting for analysis...')
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop() 