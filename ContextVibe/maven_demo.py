#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVEN多模态情感分析演示脚本
展示如何使用MAVEN进行Valence和Arousal计算
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from contextvibe.core.maven_integration import MAVENVACalculator, MAVENCalculator
from contextvibe.core.vae_calculator import VAE_CCalculator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_text_analysis():
    """演示文本情感分析"""
    print("\n" + "="*60)
    print("MAVEN文本情感分析演示")
    print("="*60)
    
    try:
        # 创建MAVEN计算器
        calculator = MAVENVACalculator()
        
        # 测试文本
        test_texts = [
            "This is a great project! I'm very excited about it.",
            "This plan is completely unfeasible, I strongly oppose it.",
            "We need to discuss this issue further.",
            "I'm so happy with the results! Everything worked perfectly!",
            "This is terrible news. I'm devastated by this outcome."
        ]
        
        print("\n文本情感分析结果:")
        for i, text in enumerate(test_texts, 1):
            valence, arousal = calculator.calculate_text_va(text)
            print(f"\n文本 {i}: {text}")
            print(f"  Valence (效价): {valence:.3f}")
            print(f"  Arousal (唤醒度): {arousal:.3f}")
            
            # 解释结果
            if valence > 0.5:
                valence_desc = "积极"
            elif valence < -0.5:
                valence_desc = "消极"
            else:
                valence_desc = "中性"
            
            if arousal > 0.5:
                arousal_desc = "高唤醒"
            elif arousal < -0.5:
                arousal_desc = "低唤醒"
            else:
                arousal_desc = "中等唤醒"
            
            print(f"  情感状态: {valence_desc}, {arousal_desc}")
        
        return True
        
    except Exception as e:
        logger.error(f"文本分析演示失败: {e}")
        return False

def demo_audio_analysis():
    """演示音频情感分析"""
    print("\n" + "="*60)
    print("MAVEN音频情感分析演示")
    print("="*60)
    
    try:
        # 创建MAVEN计算器
        calculator = MAVENVACalculator()
        
        # 查找音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(Path('.').glob(f"**/*{ext}"))
        
        if not audio_files:
            print("未找到音频文件，跳过音频分析演示")
            return True
        
        print(f"\n找到 {len(audio_files)} 个音频文件:")
        for i, audio_file in enumerate(audio_files[:3]):  # 只分析前3个
            print(f"\n音频文件 {i+1}: {audio_file.name}")
            
            try:
                valence, arousal = calculator.calculate_audio_va(str(audio_file))
                print(f"  Valence (效价): {valence:.3f}")
                print(f"  Arousal (唤醒度): {arousal:.3f}")
                
                # 解释结果
                if valence > 0.5:
                    valence_desc = "积极"
                elif valence < -0.5:
                    valence_desc = "消极"
                else:
                    valence_desc = "中性"
                
                if arousal > 0.5:
                    arousal_desc = "高唤醒"
                elif arousal < -0.5:
                    arousal_desc = "低唤醒"
                else:
                    arousal_desc = "中等唤醒"
                
                print(f"  情感状态: {valence_desc}, {arousal_desc}")
                
            except Exception as e:
                print(f"  分析失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"音频分析演示失败: {e}")
        return False

def demo_video_analysis():
    """演示视频情感分析"""
    print("\n" + "="*60)
    print("MAVEN视频情感分析演示")
    print("="*60)
    
    try:
        # 创建MAVEN计算器
        calculator = MAVENVACalculator()
        
        # 查找视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path('.').glob(f"**/*{ext}"))
        
        if not video_files:
            print("未找到视频文件，跳过视频分析演示")
            return True
        
        print(f"\n找到 {len(video_files)} 个视频文件:")
        for i, video_file in enumerate(video_files[:2]):  # 只分析前2个
            print(f"\n视频文件 {i+1}: {video_file.name}")
            
            try:
                valence, arousal = calculator.calculate_video_va(str(video_file))
                print(f"  Valence (效价): {valence:.3f}")
                print(f"  Arousal (唤醒度): {arousal:.3f}")
                
                # 解释结果
                if valence > 0.5:
                    valence_desc = "积极"
                elif valence < -0.5:
                    valence_desc = "消极"
                else:
                    valence_desc = "中性"
                
                if arousal > 0.5:
                    arousal_desc = "高唤醒"
                elif arousal < -0.5:
                    arousal_desc = "低唤醒"
                else:
                    arousal_desc = "中等唤醒"
                
                print(f"  情感状态: {valence_desc}, {arousal_desc}")
                
            except Exception as e:
                print(f"  分析失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"视频分析演示失败: {e}")
        return False

def demo_multimodal_analysis():
    """演示多模态情感分析"""
    print("\n" + "="*60)
    print("MAVEN多模态情感分析演示")
    print("="*60)
    
    try:
        # 创建VAEC计算器（使用MAVEN）
        vaec_calculator = VAE_CCalculator()
        
        # 查找各种类型的文件
        all_files = []
        for ext in ['.wav', '.mp3', '.flac', '.txt', '.json', '.mp4', '.avi']:
            all_files.extend(Path('.').glob(f"**/*{ext}"))
        
        if not all_files:
            print("未找到文件，跳过多模态分析演示")
            return True
        
        print(f"\n找到 {len(all_files)} 个文件，分析前5个:")
        for i, file_path in enumerate(all_files[:5]):
            print(f"\n文件 {i+1}: {file_path.name} ({file_path.suffix})")
            
            try:
                scores = vaec_calculator.calculate_vaec_scores(str(file_path))
                print(f"  Valence (效价): {scores['valence']:.3f}")
                print(f"  Arousal (唤醒度): {scores['arousal']:.3f}")
                print(f"  Energy (能量): {scores['energy']:.3f}")
                print(f"  Cohesion (一致性): {scores['cohesion']:.3f}")
                
            except Exception as e:
                print(f"  分析失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"多模态分析演示失败: {e}")
        return False

def demo_comparison():
    """演示MAVEN与传统方法的对比"""
    print("\n" + "="*60)
    print("MAVEN与传统方法对比演示")
    print("="*60)
    
    try:
        # 创建MAVEN计算器
        maven_calculator = MAVENVACalculator()
        
        # 创建传统计算器
        traditional_calculator = VAE_CCalculator()
        
        test_text = "This is an amazing breakthrough! I'm absolutely thrilled!"
        
        print(f"\n测试文本: {test_text}")
        
        # MAVEN方法
        print("\nMAVEN方法结果:")
        maven_valence, maven_arousal = maven_calculator.calculate_text_va(test_text)
        print(f"  Valence: {maven_valence:.3f}")
        print(f"  Arousal: {maven_arousal:.3f}")
        
        # 传统方法
        print("\n传统方法结果:")
        traditional_scores = traditional_calculator.calculate_vaec_scores("temp.txt")
        print(f"  Valence: {traditional_scores['valence']:.3f}")
        print(f"  Arousal: {traditional_scores['arousal']:.3f}")
        
        print("\n对比分析:")
        print("  MAVEN使用多模态注意力机制，能够更好地理解上下文和情感细微差别")
        print("  传统方法主要基于词汇统计和简单特征提取")
        
        return True
        
    except Exception as e:
        logger.error(f"对比演示失败: {e}")
        return False

def main():
    """主函数"""
    print("MAVEN多模态情感分析演示")
    print("="*60)
    
    # 检查MAVEN是否可用
    try:
        from contextvibe.core.maven_integration import MAVENVACalculator
        print("✓ MAVEN模块加载成功")
    except ImportError as e:
        print(f"✗ MAVEN模块加载失败: {e}")
        print("请先运行: python install_maven.py")
        return
    
    # 运行各种演示
    demos = [
        ("文本情感分析", demo_text_analysis),
        ("音频情感分析", demo_audio_analysis),
        ("视频情感分析", demo_video_analysis),
        ("多模态情感分析", demo_multimodal_analysis),
        ("方法对比", demo_comparison)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n正在运行{demo_name}...")
        if not demo_func():
            print(f"{demo_name}失败")
        else:
            print(f"{demo_name}完成")
    
    print("\n" + "="*60)
    print("MAVEN演示完成！")
    print("="*60)
    print("\n使用说明:")
    print("1. 在代码中使用MAVENVACalculator()创建计算器")
    print("2. 调用calculate_text_va(), calculate_audio_va(), calculate_video_va()方法")
    print("3. 或者直接使用VAE_CCalculator()，它会自动使用MAVEN")
    print("\n更多信息请参考README.md")

if __name__ == "__main__":
    main()
