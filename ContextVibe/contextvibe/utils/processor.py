#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextVibe: AMI Meeting Corpus 主处理程序
整合数据下载、归类、VAEC计算和报告生成功能
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入自定义模块
from ..core.vae_calculator import VAE_CCalculator
from ..analysis.multimodal_cohesion_analyzer import MultimodalCohesionAnalyzer
from ..data.organize_ami_data import AMIDataOrganizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AMIProcessor:
    """AMI Data Processing Main Program"""
    
    def __init__(self, ami_dir: str = "/root/lanyun-tmp/amicorpus"):
        """Initialize processor"""
        self.ami_dir = Path(ami_dir)
        self.results_dir = self.ami_dir / "results"
        self.sample_data_dir = self.ami_dir / "sample_data"
        self.multi_speaker_dir = self.ami_dir / "multi_speaker"
        
        # Create necessary directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.sample_data_dir.mkdir(parents=True, exist_ok=True)
        self.multi_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize calculators
        self.vaec_calculator = VAE_CCalculator()
        self.cohesion_analyzer = MultimodalCohesionAnalyzer()
        self.data_organizer = AMIDataOrganizer(str(self.ami_dir))
        
        logger.info(f"AMI processor initialized successfully, working directory: {self.ami_dir}")
    
    def download_sample_data(self):
        """Download sample data for testing"""
        logger.info("Starting sample data download...")
        
        # Create sample audio files
        sample_audio_files = [
            "sample_happy.wav",
            "sample_sad.wav", 
            "sample_excited.wav",
            "sample_calm.wav",
            "sample_angry.wav",
            "sample_neutral.wav"
        ]
        
        # Create sample text files
        sample_text_files = [
            ("sample_positive.txt", "This project has great prospects, we are all excited!"),
            ("sample_negative.txt", "This plan is completely unfeasible, I strongly oppose it."),
            ("sample_neutral.txt", "We need to discuss this issue further."),
            ("sample_excited.txt", "Excellent! This idea is so creative!"),
            ("sample_calm.txt", "Let's analyze the situation calmly."),
            ("sample_confused.txt", "I don't quite understand this part, can you explain?")
        ]
        
        # Create audio files (simulation)
        for audio_file in sample_audio_files:
            audio_path = self.sample_data_dir / audio_file
            if not audio_path.exists():
                # Create a simple audio file (placeholder here)
                with open(audio_path, 'w') as f:
                    f.write("Sample audio data")
                logger.info(f"Created sample audio file: {audio_file}")
        
        # Create text files
        for filename, content in sample_text_files:
            text_path = self.sample_data_dir / filename
            if not text_path.exists():
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created sample text file: {filename}")
        
        # Create multi-speaker sample data
        self._create_multi_speaker_samples()
        
        logger.info("Sample data download completed")
    
    def _create_multi_speaker_samples(self):
        """Create multi-speaker sample data"""
        multi_speaker_sessions = [
            {
                "session_id": "meeting_productive",
                "description": "Efficient team discussion, stable state, high integration",
                "participants": ["Alice", "Bob", "Charlie"],
                "items": [
                    {"speaker": "Alice", "text": "This plan is great, we can implement it immediately."},
                    {"speaker": "Bob", "text": "Agree, the timeline is also reasonable."},
                    {"speaker": "Charlie", "text": "I'll be responsible for the technical implementation."},
                    {"speaker": "Alice", "text": "Great, we'll start next week."},
                    {"speaker": "Bob", "text": "I'll prepare the relevant documentation."}
                ]
            },
            {
                "session_id": "meeting_stress",
                "description": "Alice's state gradually declines, others remain stable",
                "participants": ["Alice", "Bob", "Charlie"],
                "items": [
                    {"speaker": "Alice", "text": "This idea is very creative."},
                    {"speaker": "Bob", "text": "Yes, we can try it."},
                    {"speaker": "Alice", "text": "I think... there might be problems."},
                    {"speaker": "Charlie", "text": "We can solve these problems."},
                    {"speaker": "Alice", "text": "I'm not sure if I can complete..."}
                ]
            },
            {
                "session_id": "meeting_conflict",
                "description": "David doesn't fit with team atmosphere",
                "participants": ["Alice", "Bob", "Charlie", "David"],
                "items": [
                    {"speaker": "Alice", "text": "Let's discuss this new plan."},
                    {"speaker": "David", "text": "This plan is completely unfeasible."},
                    {"speaker": "Bob", "text": "We can discuss improvement plans."},
                    {"speaker": "David", "text": "Waste of time, I don't agree."},
                    {"speaker": "Charlie", "text": "Let's hear David's concerns."}
                ]
            },
            {
                "session_id": "meeting_complex",
                "description": "Complex scenario, multiple people in unstable state",
                "participants": ["Alice", "Bob", "Charlie", "Eva", "Frank"],
                "items": [
                    {"speaker": "Alice", "text": "This project is important."},
                    {"speaker": "Eva", "text": "I think it's a bit difficult."},
                    {"speaker": "Bob", "text": "We can proceed step by step."},
                    {"speaker": "Frank", "text": "I won't participate in this project."},
                    {"speaker": "Charlie", "text": "Let's replan this."}
                ]
            }
        ]
        
        # Save multi-speaker data
        sessions_file = self.multi_speaker_dir / "multi_speaker_sessions.json"
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(multi_speaker_sessions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created multi-speaker sample data: {sessions_file}")
    
    def process_sample_data(self):
        """Process sample data"""
        logger.info("Starting sample data processing...")
        
        # Collect all sample files
        sample_files = []
        for file_path in self.sample_data_dir.glob("*"):
            if file_path.is_file():
                sample_files.append(str(file_path))
        
        # Calculate VAEC scores
        vaec_scores = []
        for file_path in sample_files:
            try:
                scores = self.vaec_calculator.calculate_vaec_scores(file_path)
                file_info = {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'file_type': Path(file_path).suffix,
                    'file_size': os.path.getsize(file_path),
                    'valence': scores['valence'],
                    'arousal': scores['arousal'],
                    'energy': scores['energy'],
                    'cohesion': scores['cohesion']
                }
                vaec_scores.append(file_info)
                logger.info(f"Processing file: {Path(file_path).name}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
        
        # Save file-level scores
        vaec_df = pd.DataFrame(vaec_scores)
        vaec_file = self.results_dir / "vaec_scores.csv"
        vaec_df.to_csv(vaec_file, index=False, encoding='utf-8')
        logger.info(f"Saved file-level VAEC scores: {vaec_file}")
        
        # Calculate session-level scores
        self._calculate_session_scores(vaec_scores)
        
        # Generate statistics report
        self._generate_statistics_report(vaec_scores)
        
        # Generate summary report
        self._generate_summary_report(vaec_scores)
        
        # Generate visualization charts
        self._generate_visualization(vaec_df)
        
        logger.info("Sample data processing completed")
    
    def _calculate_session_scores(self, vaec_scores: List[Dict]):
        """Calculate session-level scores"""
        # Group by filename prefix
        session_groups = {}
        for score in vaec_scores:
            file_name = score['file_name']
            session_id = file_name.split('_')[0] if '_' in file_name else 'unknown'
            
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(score)
        
        # Calculate session-level scores
        session_scores = []
        for session_id, scores in session_groups.items():
            session_score = {
                'session_id': session_id,
                'file_count': len(scores),
                'valence_mean': sum(s['valence'] for s in scores) / len(scores),
                'arousal_mean': sum(s['arousal'] for s in scores) / len(scores),
                'energy_mean': sum(s['energy'] for s in scores) / len(scores),
                'cohesion_mean': sum(s['cohesion'] for s in scores) / len(scores)
            }
            session_scores.append(session_score)
        
        # Save session-level scores
        session_df = pd.DataFrame(session_scores)
        session_file = self.results_dir / "session_scores.csv"
        session_df.to_csv(session_file, index=False, encoding='utf-8')
        logger.info(f"Saved session-level scores: {session_file}")
    
    def _generate_statistics_report(self, vaec_scores: List[Dict]):
        """Generate statistics report"""
        if not vaec_scores:
            return
        
        # Calculate statistics
        df = pd.DataFrame(vaec_scores)
        stats = {
            'total_files': len(vaec_scores),
            'file_types': df['file_type'].value_counts().to_dict(),
            'valence_stats': {
                'mean': df['valence'].mean(),
                'std': df['valence'].std(),
                'min': df['valence'].min(),
                'max': df['valence'].max()
            },
            'arousal_stats': {
                'mean': df['arousal'].mean(),
                'std': df['arousal'].std(),
                'min': df['arousal'].min(),
                'max': df['arousal'].max()
            },
            'energy_stats': {
                'mean': df['energy'].mean(),
                'std': df['energy'].std(),
                'min': df['energy'].min(),
                'max': df['energy'].max()
            },
            'cohesion_stats': {
                'mean': df['cohesion'].mean(),
                'std': df['cohesion'].std(),
                'min': df['cohesion'].min(),
                'max': df['cohesion'].max()
            }
        }
        
        # Save statistics report
        stats_file = self.results_dir / "statistics_report.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Generated statistics report: {stats_file}")
    
    def _generate_summary_report(self, vaec_scores: List[Dict]):
        """Generate summary report"""
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_files_processed': len(vaec_scores),
            'processing_status': 'completed',
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            },
            'file_summary': {
                'audio_files': len([s for s in vaec_scores if s['file_type'] in ['.wav', '.mp3', '.flac']]),
                'text_files': len([s for s in vaec_scores if s['file_type'] in ['.txt', '.json']]),
                'video_files': len([s for s in vaec_scores if s['file_type'] in ['.mp4', '.avi', '.mov']])
            }
        }
        
        # Save summary report
        summary_file = self.results_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Generated summary report: {summary_file}")
    
    def _generate_visualization(self, vaec_df: pd.DataFrame):
        """Generate visualization charts"""
        try:
            # Set font
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Create charts
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ContextVibe VAEC Analysis Results', fontsize=16, fontweight='bold')
            
            # 1. Valence distribution
            axes[0, 0].hist(vaec_df['valence'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Valence Distribution')
            axes[0, 0].set_xlabel('Valence Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Arousal distribution
            axes[0, 1].hist(vaec_df['arousal'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Arousal Distribution')
            axes[0, 1].set_xlabel('Arousal Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Energy distribution
            axes[1, 0].hist(vaec_df['energy'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('Energy Distribution')
            axes[1, 0].set_xlabel('Energy Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Cohesion distribution
            axes[1, 1].hist(vaec_df['cohesion'], bins=10, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].set_title('Cohesion Distribution')
            axes[1, 1].set_xlabel('Cohesion Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_file = self.results_dir / "vaec_analysis.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated visualization charts: {chart_file}")
        except Exception as e:
            logger.error(f"Failed to generate visualization charts: {e}")
    
    def analyze_multi_speaker_sessions(self):
        """Analyze multi-speaker sessions"""
        logger.info("Starting multi-speaker session analysis...")
        
        sessions_file = self.multi_speaker_dir / "multi_speaker_sessions.json"
        if not sessions_file.exists():
            logger.warning("Multi-speaker data file does not exist, skipping analysis")
            return
        
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        # Analyze each session
        for session in sessions:
            logger.info(f"Analyzing session: {session['session_id']}")
            report = self.cohesion_analyzer.generate_cohesion_report(session)
            
            if report:
                summary = report['summary']
                logger.info(f"  Text cohesion: {summary.get('text_cohesion', 0):.3f}")
                logger.info(f"  Audio cohesion: {summary.get('audio_cohesion', 0):.3f}")
                logger.info(f"  Video cohesion: {summary.get('video_cohesion', 0):.3f}")
                logger.info(f"  Multimodal cohesion: {summary.get('multimodal_cohesion', 0):.3f}")
        
        logger.info("Multi-speaker session analysis completed")
    
    def organize_ami_data(self):
        """Organize AMI data"""
        logger.info("Starting AMI data organization...")
        self.data_organizer.run_organization()
        logger.info("AMI data organization completed")
    
    def run_full_pipeline(self):
        """Run complete processing pipeline"""
        logger.info("Starting complete processing pipeline...")
        
        # 1. Download sample data
        self.download_sample_data()
        
        # 2. Process sample data
        self.process_sample_data()
        
        # 3. Analyze multi-speaker sessions
        self.analyze_multi_speaker_sessions()
        
        # 4. Organize AMI data (if exists)
        if any(self.ami_dir.glob("*")):
            self.organize_ami_data()
        
        logger.info("Complete processing pipeline finished")
        self._print_results_summary()
    
    def _print_results_summary(self):
        """Print results summary"""
        print("\n" + "="*60)
        print("ContextVibe Processing Results Summary")
        print("="*60)
        
        # Check result files
        results_files = list(self.results_dir.glob("*"))
        if results_files:
            print(f"📊 Generated result files: {len(results_files)} files")
            for file_path in results_files:
                file_size = file_path.stat().st_size
                print(f"   📄 {file_path.name} ({file_size} bytes)")
        else:
            print("❌ No result files generated")
        
        # Check sample data
        sample_files = list(self.sample_data_dir.glob("*"))
        if sample_files:
            print(f"📁 Sample data files: {len(sample_files)} files")
        
        # Check multi-speaker data
        multi_speaker_files = list(self.multi_speaker_dir.glob("*"))
        if multi_speaker_files:
            print(f"👥 Multi-speaker data: {len(multi_speaker_files)} files")
        
        print("="*60)
        print("🎉 Processing completed! Check result files for detailed information.")
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ContextVibe: AMI Meeting Corpus 处理程序')
    parser.add_argument('--ami_dir', type=str, default='/root/lanyun-tmp/amicorpus',
                       help='AMI数据目录路径')
    parser.add_argument('--download_sample', action='store_true',
                       help='下载样本数据')
    parser.add_argument('--process_sample', action='store_true',
                       help='处理样本数据')
    parser.add_argument('--analyze_multi_speaker', action='store_true',
                       help='分析多人对话')
    parser.add_argument('--organize_data', action='store_true',
                       help='归类AMI数据')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='运行完整处理流程')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = AMIProcessor(args.ami_dir)
    
    # 根据参数执行相应功能
    if args.download_sample:
        processor.download_sample_data()
    elif args.process_sample:
        processor.process_sample_data()
    elif args.analyze_multi_speaker:
        processor.analyze_multi_speaker_sessions()
    elif args.organize_data:
        processor.organize_ami_data()
    elif args.full_pipeline:
        processor.run_full_pipeline()
    else:
        # 默认运行完整流程
        processor.run_full_pipeline()

if __name__ == "__main__":
    main()
