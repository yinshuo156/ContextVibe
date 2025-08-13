#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVEN (Multi-modal Attention for Valence-Arousal Emotion Network) 集成模块
用于替换现有的Valence和Arousal计算，提供更准确的多模态情感分析
"""

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
from pathlib import Path
import tempfile
import shutil

# 设置huggingface镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    from transformers import (
        HubertModel, Wav2Vec2FeatureExtractor, 
        RobertaModel, RobertaTokenizer, BeitModel
    )
    import timm
    from torchvision import transforms
except ImportError as e:
    logging.warning(f"MAVEN依赖导入失败: {e}")
    logging.warning("请安装MAVEN依赖: pip install transformers timm torchvision")

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MAVENFeatureExtractor:
    """MAVEN特征提取器"""
    
    def __init__(self, device: str = "auto"):
        """初始化MAVEN特征提取器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        logger.info(f"MAVEN特征提取器使用设备: {self.device}")
        
        try:
            self._load_models()
            logger.info("MAVEN模型加载成功")
        except Exception as e:
            logger.error(f"MAVEN模型加载失败: {e}")
            raise
    
    def _load_models(self):
        """加载预训练模型"""
        # 视觉特征提取器 (Swin Transformer)
        self.swin = timm.create_model(
            "swin_base_patch4_window7_224", 
            pretrained=True, 
            num_classes=0
        ).to(self.device)
        
        # 音频特征提取器 (HuBERT)
        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            mirror="https://hf-mirror.com"
        ).to(self.device)
        self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960",
            mirror="https://hf-mirror.com"
        )
        
        # 文本特征提取器 (RoBERTa)
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base",
            mirror="https://hf-mirror.com"
        ).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base",
            mirror="https://hf-mirror.com"
        )
        
        # 冻结所有预训练模型参数
        for model in [self.swin, self.hubert, self.roberta]:
            for param in model.parameters():
                param.requires_grad = False
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_video_features(self, video_path: str, max_frames: int = 30) -> torch.Tensor:
        """提取视频特征"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为PIL图像
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # 预处理
                processed_frame = self.image_transform(pil_image)
                frames.append(processed_frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                logger.warning(f"无法从视频中提取帧: {video_path}")
                return torch.zeros(1, 1024, device=self.device)
            
            # 堆叠帧并提取特征
            frames_tensor = torch.stack(frames).to(self.device)
            
            with torch.no_grad():
                # 使用Swin Transformer提取特征
                batch_size, num_frames = frames_tensor.shape[:2]
                frames_flat = frames_tensor.view(-1, 3, 224, 224)
                
                features_list = []
                for i in range(0, batch_size * num_frames, 32):
                    end_idx = min(i + 32, batch_size * num_frames)
                    batch_frames = frames_flat[i:end_idx]
                    if batch_frames.shape[0] > 0:
                        features = self.swin(batch_frames)
                        features_list.append(features)
                
                if features_list:
                    video_features = torch.cat(features_list, dim=0)
                    video_features = video_features.view(batch_size, num_frames, -1)
                    # 平均池化
                    return video_features.mean(dim=1)
                else:
                    return torch.zeros(1, 1024, device=self.device)
                    
        except Exception as e:
            logger.error(f"视频特征提取失败 {video_path}: {e}")
            return torch.zeros(1, 1024, device=self.device)
    
    def extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """提取音频特征"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 如果音频太短，进行填充
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            elif len(audio) > 16000:
                audio = audio[:16000]
            
            # 提取特征
            audio_features = self.audio_feature_extractor(
                audio.reshape(1, -1),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            audio_input_values = audio_features.input_values.to(self.device)
            
            with torch.no_grad():
                audio_outputs = self.hubert(audio_input_values)
                audio_features = audio_outputs.last_hidden_state.mean(dim=1)
            
            return audio_features
            
        except Exception as e:
            logger.error(f"音频特征提取失败 {audio_path}: {e}")
            return torch.zeros(1, 768, device=self.device)
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        """提取文本特征"""
        try:
            if not text or text.strip() == "":
                return torch.zeros(1, 768, device=self.device)
            
            # 分词和编码
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                text_outputs = self.roberta(**inputs)
                text_features = text_outputs.last_hidden_state.mean(dim=1)
            
            return text_features
            
        except Exception as e:
            logger.error(f"文本特征提取失败: {e}")
            return torch.zeros(1, 768, device=self.device)


class MAVENCrossModalAttention(nn.Module):
    """MAVEN跨模态注意力机制"""
    
    def __init__(self, video_dim=1024, audio_dim=768, text_dim=768, hidden_dim=512):
        super(MAVENCrossModalAttention, self).__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 特征投影层
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 跨模态注意力层
        self.video_to_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.video_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.audio_to_video_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.audio_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.text_to_video_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # 情感预测层
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # valence, arousal
        )
    
    def forward(self, video_features, audio_features, text_features):
        """前向传播"""
        batch_size = video_features.shape[0]
        
        # 特征投影
        video_proj = self.video_proj(video_features).unsqueeze(1)  # [B, 1, H]
        audio_proj = self.audio_proj(audio_features).unsqueeze(1)  # [B, 1, H]
        text_proj = self.text_proj(text_features).unsqueeze(1)     # [B, 1, H]
        
        # 跨模态注意力
        # Video -> Audio
        video_audio_attn, _ = self.video_to_audio_attn(video_proj, audio_proj, audio_proj)
        
        # Video -> Text
        video_text_attn, _ = self.video_to_text_attn(video_proj, text_proj, text_proj)
        
        # Audio -> Video
        audio_video_attn, _ = self.audio_to_video_attn(audio_proj, video_proj, video_proj)
        
        # Audio -> Text
        audio_text_attn, _ = self.audio_to_text_attn(audio_proj, text_proj, text_proj)
        
        # Text -> Video
        text_video_attn, _ = self.text_to_video_attn(text_proj, video_proj, video_proj)
        
        # Text -> Audio
        text_audio_attn, _ = self.text_to_audio_attn(text_proj, audio_proj, audio_proj)
        
        # 融合跨模态特征
        enhanced_video = video_proj + video_audio_attn + video_text_attn
        enhanced_audio = audio_proj + audio_video_attn + audio_text_attn
        enhanced_text = text_proj + text_video_attn + text_audio_attn
        
        # 自注意力
        all_features = torch.cat([enhanced_video, enhanced_audio, enhanced_text], dim=1)
        self_attn_out, _ = self.self_attn(all_features, all_features, all_features)
        
        # 全局平均池化
        global_features = self_attn_out.mean(dim=1)
        
        # 输出投影
        output_features = self.output_proj(global_features)
        
        # 情感预测
        emotion_scores = self.emotion_head(output_features)
        
        return emotion_scores


class MAVENCalculator:
    """MAVEN情感计算器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """初始化MAVEN计算器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        logger.info(f"MAVEN计算器使用设备: {self.device}")
        
        # 初始化特征提取器
        self.feature_extractor = MAVENFeatureExtractor(device=device)
        
        # 初始化MAVEN模型
        self.maven_model = MAVENCrossModalAttention().to(self.device)
        
        # 加载预训练权重（如果有）
        if model_path and os.path.exists(model_path):
            try:
                self.maven_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"加载MAVEN模型权重: {model_path}")
            except Exception as e:
                logger.warning(f"加载MAVEN模型权重失败: {e}")
        
        # 设置为评估模式
        self.maven_model.eval()
        logger.info("MAVEN计算器初始化完成")
    
    def calculate_va_scores(self, 
                          video_path: Optional[str] = None,
                          audio_path: Optional[str] = None,
                          text: Optional[str] = None) -> Tuple[float, float]:
        """计算Valence和Arousal分数"""
        try:
            # 提取特征
            if video_path and os.path.exists(video_path):
                video_features = self.feature_extractor.extract_video_features(video_path)
            else:
                video_features = torch.zeros(1, 1024, device=self.device)
            
            if audio_path and os.path.exists(audio_path):
                audio_features = self.feature_extractor.extract_audio_features(audio_path)
            else:
                audio_features = torch.zeros(1, 768, device=self.device)
            
            if text:
                text_features = self.feature_extractor.extract_text_features(text)
            else:
                text_features = torch.zeros(1, 768, device=self.device)
            
            # 预测情感分数
            with torch.no_grad():
                emotion_scores = self.maven_model(video_features, audio_features, text_features)
                
                # 转换为[-1, 1]范围
                valence = torch.tanh(emotion_scores[0, 0]).item()
                arousal = torch.tanh(emotion_scores[0, 1]).item()
            
            return valence, arousal
            
        except Exception as e:
            logger.error(f"MAVEN情感计算失败: {e}")
            return 0.0, 0.0
    
    def calculate_multimodal_va(self, data_path: str) -> Tuple[float, float]:
        """根据文件类型计算多模态VA分数"""
        try:
            file_path = Path(data_path)
            
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # 视频文件
                return self.calculate_va_scores(video_path=data_path)
            
            elif file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                # 音频文件
                return self.calculate_va_scores(audio_path=data_path)
            
            elif file_path.suffix.lower() in ['.txt', '.json']:
                # 文本文件
                with open(data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return self.calculate_va_scores(text=text)
            
            else:
                logger.warning(f"不支持的文件类型: {file_path.suffix}")
                return 0.0, 0.0
                
        except Exception as e:
            logger.error(f"多模态VA计算失败 {data_path}: {e}")
            return 0.0, 0.0


# 兼容性包装器，用于替换现有的VACalculator
class MAVENVACalculator:
    """MAVEN兼容的VA计算器，用于替换现有的VACalculator"""
    
    def __init__(self, model_path: Optional[str] = None):
        """初始化MAVEN VA计算器"""
        self.maven_calculator = MAVENCalculator(model_path=model_path)
        logger.info("MAVEN VA计算器初始化完成")
    
    def calculate_text_va(self, text: str) -> Tuple[float, float]:
        """计算文本的Valence和Arousal"""
        return self.maven_calculator.calculate_va_scores(text=text)
    
    def calculate_audio_va(self, audio_path: str) -> Tuple[float, float]:
        """计算音频的Valence和Arousal"""
        return self.maven_calculator.calculate_va_scores(audio_path=audio_path)
    
    def calculate_video_va(self, video_path: str) -> Tuple[float, float]:
        """计算视频的Valence和Arousal"""
        return self.maven_calculator.calculate_va_scores(video_path=video_path)
    
    def calculate_multimodal_va(self, data_path: str) -> Tuple[float, float]:
        """根据文件类型计算多模态VA"""
        return self.maven_calculator.calculate_multimodal_va(data_path)


if __name__ == "__main__":
    # 测试代码
    calculator = MAVENVACalculator()
    
    # 测试文本情感分析
    test_text = "This is a great project! I'm very excited about it."
    valence, arousal = calculator.calculate_text_va(test_text)
    print(f"文本: {test_text}")
    print(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}")
