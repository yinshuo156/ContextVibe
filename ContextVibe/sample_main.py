#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextVibe 示例主程序
展示如何使用ContextVibe包进行多模态会议情感分析
"""

import sys
import os
from pathlib import Path

# 添加包路径
sys.path.insert(0, str(Path(__file__).parent))

from contextvibe import VAE_CCalculator, MultimodalCohesionAnalyzer, AMIDataOrganizer, AMIProcessor

def demo_basic_usage():
    """Demonstrate basic usage"""
    print("=" * 60)
    print("ContextVibe Basic Usage Demo")
    print("=" * 60)
    
    # 1. Create VAEC calculator
    print("\n1. Creating VAEC calculator...")
    calculator = VAE_CCalculator()
    print("✅ VAE_CCalculator initialized successfully")
    
    # 2. Create multimodal cohesion analyzer
    print("\n2. Creating multimodal cohesion analyzer...")
    analyzer = MultimodalCohesionAnalyzer()
    print("✅ MultimodalCohesionAnalyzer initialized successfully")
    
    # 3. Create data organizer
    print("\n3. Creating data organizer...")
    organizer = AMIDataOrganizer()
    print("✅ AMIDataOrganizer initialized successfully")
    
    # 4. Create main processor
    print("\n4. Creating main processor...")
    processor = AMIProcessor()
    print("✅ AMIProcessor initialized successfully")
    
    print("\n🎉 All components initialized successfully!")

def demo_vaec_calculation():
    """Demonstrate VAEC calculation"""
    print("\n" + "=" * 60)
    print("VAEC Calculation Demo")
    print("=" * 60)
    
    calculator = VAE_CCalculator()
    
    # Sample texts
    sample_texts = [
        "This project has great prospects, we are all excited!",
        "This plan is completely unfeasible, I strongly oppose it.",
        "We need to discuss this issue further."
    ]
    
    print("\nText sentiment analysis results:")
    for i, text in enumerate(sample_texts, 1):
        scores = calculator.va_calculator.calculate_text_va(text)
        print(f"Text {i}: {text}")
        print(f"  Valence: {scores[0]:.3f}")
        print(f"  Arousal: {scores[1]:.3f}")
        print()

def demo_multi_speaker_analysis():
    """Demonstrate multi-speaker analysis"""
    print("\n" + "=" * 60)
    print("Multi-Speaker Analysis Demo")
    print("=" * 60)
    
    analyzer = MultimodalCohesionAnalyzer()
    
    # Sample session data
    session_data = {
        "session_id": "demo_meeting",
        "description": "Demo meeting",
        "participants": ["Alice", "Bob", "Charlie"],
        "items": [
            {"speaker": "Alice", "text": "This plan is great, we can implement it immediately."},
            {"speaker": "Bob", "text": "Agree, the timeline is also reasonable."},
            {"speaker": "Charlie", "text": "I'll be responsible for the technical implementation."},
            {"speaker": "Alice", "text": "Great, we'll start next week."},
            {"speaker": "Bob", "text": "I'll prepare the relevant documentation."}
        ]
    }
    
    print("Analyzing session data...")
    report = analyzer.generate_cohesion_report(session_data)
    
    if report:
        summary = report['summary']
        print(f"✅ Analysis completed!")
        print(f"  Text cohesion: {summary.get('text_cohesion', 0):.3f}")
        print(f"  Audio cohesion: {summary.get('audio_cohesion', 0):.3f}")
        print(f"  Video cohesion: {summary.get('video_cohesion', 0):.3f}")
        print(f"  Multimodal cohesion: {summary.get('multimodal_cohesion', 0):.3f}")

def demo_full_pipeline():
    """Demonstrate complete processing pipeline"""
    print("\n" + "=" * 60)
    print("Complete Processing Pipeline Demo")
    print("=" * 60)
    
    processor = AMIProcessor()
    
    print("Starting complete processing pipeline...")
    processor.run_full_pipeline()
    
    print("\n✅ Complete processing pipeline finished!")
    print("Check result files in the results/ directory")

def main():
    """Main function"""
    print("🎯 ContextVibe Multimodal Meeting Emotion Analysis System")
    print("Version: 1.0.0")
    print("=" * 60)
    
    try:
        # Demonstrate basic usage
        demo_basic_usage()
        
        # Demonstrate VAEC calculation
        demo_vaec_calculation()
        
        # Demonstrate multi-speaker analysis
        demo_multi_speaker_analysis()
        
        # Demonstrate complete pipeline
        demo_full_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed!")
        print("=" * 60)
        print("\n📚 For more information, see:")
        print("  - README.md: Detailed documentation")
        print("  - contextvibe --help: Command line help")
        print("  - Example code: sample_main.py")
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
