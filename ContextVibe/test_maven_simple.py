#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MAVEN测试脚本
验证MAVEN集成是否正常工作，不下载大型预训练模型
"""

import os
import sys
import logging
from pathlib import Path

# 设置huggingface镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试导入"""
    print("测试模块导入...")
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers版本: {transformers.__version__}")
        
        import timm
        print(f"✓ TIMM版本: {timm.__version__}")
        
        import librosa
        print(f"✓ Librosa版本: {librosa.__version__}")
        
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_maven_integration():
    """测试MAVEN集成"""
    print("\n测试MAVEN集成...")
    
    try:
        # 测试导入MAVEN模块
        sys.path.append(str(Path(__file__).parent))
        from contextvibe.core.maven_integration import MAVENVACalculator
        
        print("✓ MAVEN模块导入成功")
        
        # 创建计算器（不下载模型）
        print("创建MAVEN计算器...")
        calculator = MAVENVACalculator()
        print("✓ MAVEN计算器创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ MAVEN集成测试失败: {e}")
        return False

def test_vaec_calculator():
    """测试VAEC计算器"""
    print("\n测试VAEC计算器...")
    
    try:
        from contextvibe.core.vae_calculator import VAE_CCalculator
        
        # 创建计算器
        calculator = VAE_CCalculator(use_maven=False)  # 使用传统方法
        print("✓ VAEC计算器创建成功")
        
        # 测试文本情感分析
        test_text = "This is a great project!"
        valence, arousal = calculator.va_calculator.calculate_text_va(test_text)
        print(f"✓ 文本情感分析: Valence={valence:.3f}, Arousal={arousal:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ VAEC计算器测试失败: {e}")
        return False

def test_maven_fallback():
    """测试MAVEN回退机制"""
    print("\n测试MAVEN回退机制...")
    
    try:
        from contextvibe.core.vae_calculator import VAE_CCalculator
        
        # 创建计算器，MAVEN失败时回退到传统方法
        calculator = VAE_CCalculator(use_maven=True)
        print("✓ 回退机制测试成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 回退机制测试失败: {e}")
        return False

def create_sample_files():
    """创建测试文件"""
    print("\n创建测试文件...")
    
    # 创建测试文本文件
    test_text = "This is a test file for MAVEN integration. I'm very excited about this project!"
    with open("test_sample.txt", "w", encoding="utf-8") as f:
        f.write(test_text)
    print("✓ 创建测试文本文件: test_sample.txt")
    
    return True

def test_file_processing():
    """测试文件处理"""
    print("\n测试文件处理...")
    
    try:
        from contextvibe.core.vae_calculator import VAE_CCalculator
        
        calculator = VAE_CCalculator(use_maven=False)
        
        # 测试文本文件处理
        if os.path.exists("test_sample.txt"):
            scores = calculator.calculate_vaec_scores("test_sample.txt")
            print(f"✓ 文件处理成功: {scores}")
            return True
        else:
            print("✗ 测试文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 文件处理测试失败: {e}")
        return False

def main():
    """主函数"""
    print("MAVEN集成测试")
    print("="*50)
    
    tests = [
        ("模块导入", test_imports),
        ("MAVEN集成", test_maven_integration),
        ("VAEC计算器", test_vaec_calculator),
        ("MAVEN回退", test_maven_fallback),
        ("创建测试文件", create_sample_files),
        ("文件处理", test_file_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！MAVEN集成成功！")
        print("\n使用说明:")
        print("1. 运行 'python maven_demo.py' 查看完整演示")
        print("2. 在代码中使用 VAE_CCalculator() 自动使用MAVEN")
        print("3. 或直接使用 MAVENVACalculator() 进行多模态情感分析")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查配置")
    
    # 清理测试文件
    if os.path.exists("test_sample.txt"):
        os.remove("test_sample.txt")
        print("✓ 清理测试文件")

if __name__ == "__main__":
    main()

