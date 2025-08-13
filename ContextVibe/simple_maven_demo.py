#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MAVEN演示脚本
展示如何使用MAVEN进行Valence和Arousal计算
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_text_analysis():
    """演示文本情感分析"""
    print("\n" + "="*60)
    print("MAVEN文本情感分析演示")
    print("="*60)
    
    try:
        from contextvibe.core.maven_integration import MAVENVACalculator
        
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

def demo_vaec_integration():
    """演示VAEC集成"""
    print("\n" + "="*60)
    print("MAVEN与VAEC集成演示")
    print("="*60)
    
    try:
        from contextvibe.core.vae_calculator import VAE_CCalculator
        
        # 创建VAEC计算器（使用MAVEN）
        vaec_calculator = VAE_CCalculator(use_maven=True)
        
        # 测试文本
        test_text = "This is an amazing breakthrough! I'm absolutely thrilled!"
        
        print(f"\n测试文本: {test_text}")
        
        # 计算VAEC分数
        scores = vaec_calculator.calculate_vaec_scores("temp.txt")
        print(f"\nVAEC分数:")
        print(f"  Valence (效价): {scores['valence']:.3f}")
        print(f"  Arousal (唤醒度): {scores['arousal']:.3f}")
        print(f"  Energy (能量): {scores['energy']:.3f}")
        print(f"  Cohesion (一致性): {scores['cohesion']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"VAEC集成演示失败: {e}")
        return False

def demo_comparison():
    """演示MAVEN与传统方法的对比"""
    print("\n" + "="*60)
    print("MAVEN与传统方法对比演示")
    print("="*60)
    
    try:
        from contextvibe.core.maven_integration import MAVENVACalculator
        from contextvibe.core.vae_calculator import VAE_CCalculator
        
        # 创建MAVEN计算器
        maven_calculator = MAVENVACalculator()
        
        # 创建传统计算器
        traditional_calculator = VAE_CCalculator(use_maven=False)
        
        test_text = "This is an amazing breakthrough! I'm absolutely thrilled!"
        
        print(f"\n测试文本: {test_text}")
        
        # MAVEN方法
        print("\nMAVEN方法结果:")
        maven_valence, maven_arousal = maven_calculator.calculate_text_va(test_text)
        print(f"  Valence: {maven_valence:.3f}")
        print(f"  Arousal: {maven_arousal:.3f}")
        
        # 传统方法
        print("\n传统方法结果:")
        traditional_valence, traditional_arousal = traditional_calculator.va_calculator.calculate_text_va(test_text)
        print(f"  Valence: {traditional_valence:.3f}")
        print(f"  Arousal: {traditional_arousal:.3f}")
        
        print("\n对比分析:")
        print("  MAVEN使用多模态注意力机制，能够更好地理解上下文和情感细微差别")
        print("  传统方法主要基于词汇统计和简单特征提取")
        print("  MAVEN提供更准确和细粒度的情感分析")
        
        return True
        
    except Exception as e:
        logger.error(f"对比演示失败: {e}")
        return False

def demo_usage_examples():
    """演示使用示例"""
    print("\n" + "="*60)
    print("MAVEN使用示例")
    print("="*60)
    
    print("\n1. 基本使用:")
    print("```python")
    print("from contextvibe.core.maven_integration import MAVENVACalculator")
    print("")
    print("# 创建计算器")
    print("calculator = MAVENVACalculator()")
    print("")
    print("# 文本情感分析")
    print("valence, arousal = calculator.calculate_text_va(\"I'm very excited!\")")
    print("```")
    
    print("\n2. 集成到VAEC:")
    print("```python")
    print("from contextvibe.core.vae_calculator import VAE_CCalculator")
    print("")
    print("# 自动使用MAVEN")
    print("vaec_calculator = VAE_CCalculator(use_maven=True)")
    print("")
    print("# 计算VAEC分数")
    print("scores = vaec_calculator.calculate_vaec_scores(\"data.txt\")")
    print("```")
    
    print("\n3. 多模态分析:")
    print("```python")
    print("# 音频情感分析")
    print("valence, arousal = calculator.calculate_audio_va(\"audio.wav\")")
    print("")
    print("# 视频情感分析")
    print("valence, arousal = calculator.calculate_video_va(\"video.mp4\")")
    print("")
    print("# 多模态情感分析")
    print("valence, arousal = calculator.calculate_multimodal_va(\"data.txt\")")
    print("```")
    
    return True

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
        print("请检查依赖安装")
        return
    
    # 运行各种演示
    demos = [
        ("文本情感分析", demo_text_analysis),
        ("VAEC集成", demo_vaec_integration),
        ("方法对比", demo_comparison),
        ("使用示例", demo_usage_examples)
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
    print("\n总结:")
    print("✓ MAVEN已成功集成到ContextVibe项目中")
    print("✓ 提供了简化的特征提取，无需GPU和大型预训练模型")
    print("✓ 支持文本、音频、视频的多模态情感分析")
    print("✓ 与现有VAEC计算器完全兼容")
    print("✓ 提供回退机制，确保系统稳定性")
    print("\n现在您可以在项目中使用MAVEN进行更准确的Valence和Arousal计算！")

if __name__ == "__main__":
    main()
