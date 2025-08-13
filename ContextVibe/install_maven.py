#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVEN安装脚本
下载和配置MAVEN多模态情感分析所需的预训练模型
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """运行命令并处理错误"""
    logger.info(f"正在{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"{description}成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description}失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def install_maven_dependencies():
    """安装MAVEN依赖"""
    logger.info("开始安装MAVEN依赖...")
    
    # 设置huggingface镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 安装必要的包
    packages = [
        "timm>=0.9.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "librosa>=0.10.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"安装{package}"):
            logger.error(f"安装{package}失败")
            return False
    
    logger.info("MAVEN依赖安装完成")
    return True

def download_pretrained_models():
    """下载预训练模型"""
    logger.info("开始下载预训练模型...")
    
    try:
        import torch
        from transformers import HubertModel, Wav2Vec2FeatureExtractor, RobertaModel, RobertaTokenizer
        import timm
        
        # 下载Swin Transformer
        logger.info("下载Swin Transformer模型...")
        swin_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        
        # 下载HuBERT模型
        logger.info("下载HuBERT模型...")
        hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960", mirror="https://hf-mirror.com")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960", mirror="https://hf-mirror.com")
        
        # 下载RoBERTa模型
        logger.info("下载RoBERTa模型...")
        roberta_model = RobertaModel.from_pretrained("roberta-base", mirror="https://hf-mirror.com")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", mirror="https://hf-mirror.com")
        
        logger.info("所有预训练模型下载完成")
        return True
        
    except Exception as e:
        logger.error(f"下载预训练模型失败: {e}")
        return False

def create_model_cache_dir():
    """创建模型缓存目录"""
    cache_dir = Path.home() / ".cache" / "maven_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"模型缓存目录: {cache_dir}")
    return cache_dir

def test_maven_integration():
    """测试MAVEN集成"""
    logger.info("测试MAVEN集成...")
    
    try:
        from contextvibe.core.maven_integration import MAVENVACalculator
        
        # 创建计算器
        calculator = MAVENVACalculator()
        
        # 测试文本情感分析
        test_text = "This is a great project! I'm very excited about it."
        valence, arousal = calculator.calculate_text_va(test_text)
        
        logger.info(f"测试文本: {test_text}")
        logger.info(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}")
        
        logger.info("MAVEN集成测试成功")
        return True
        
    except Exception as e:
        logger.error(f"MAVEN集成测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始MAVEN安装流程...")
    
    # 1. 安装依赖
    if not install_maven_dependencies():
        logger.error("依赖安装失败")
        sys.exit(1)
    
    # 2. 创建缓存目录
    cache_dir = create_model_cache_dir()
    
    # 3. 下载预训练模型
    if not download_pretrained_models():
        logger.error("预训练模型下载失败")
        sys.exit(1)
    
    # 4. 测试集成
    if not test_maven_integration():
        logger.error("MAVEN集成测试失败")
        sys.exit(1)
    
    logger.info("MAVEN安装完成！")
    logger.info("现在可以在ContextVibe中使用MAVEN进行多模态情感分析了")

if __name__ == "__main__":
    main()
