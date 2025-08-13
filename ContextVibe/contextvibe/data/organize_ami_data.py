#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMI Meeting Corpus 数据归类脚本
基于 https://www.idiap.ch/software/bob/docs/bob/bob.db.ami/master/
"""

import os
import shutil
import glob
from pathlib import Path

class AMIDataOrganizer:
    def __init__(self, ami_dir="/root/lanyun-tmp/amicorpus"):
        self.ami_dir = Path(ami_dir)
        self.setup_directories()
    
    def setup_directories(self):
        """创建AMI数据目录结构"""
        directories = [
            "audio/close_talking",      # 近距离麦克风音频
            "audio/far_field",          # 远场麦克风音频
            "video/individual",         # 个人视角视频
            "video/room_view",          # 房间视角视频
            "annotations/transcripts",  # 转录文本
            "annotations/dialogue_acts", # 对话行为
            "annotations/emotions",     # 情感状态
            "annotations/gestures",     # 手势标注
            "slides",                   # 幻灯片数据
            "whiteboard",               # 白板数据
            "metadata"                  # 元数据
        ]
        
        for dir_path in directories:
            full_path = self.ami_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {full_path}")
    
    def organize_audio_files(self):
        """归类音频文件"""
        print("\n=== 归类音频文件 ===")
        
        # 查找音频文件
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(str(self.ami_dir / "**" / ext), recursive=True))
        
        for audio_file in audio_files:
            file_path = Path(audio_file)
            filename = file_path.name.lower()
            
            # 根据文件名判断是近距离还是远场麦克风
            if any(keyword in filename for keyword in ['close', 'head', 'individual']):
                dest_dir = self.ami_dir / "audio" / "close_talking"
            elif any(keyword in filename for keyword in ['far', 'room', 'array']):
                dest_dir = self.ami_dir / "audio" / "far_field"
            else:
                # 默认归类到近距离
                dest_dir = self.ami_dir / "audio" / "close_talking"
            
            dest_path = dest_dir / file_path.name
            if not dest_path.exists():
                shutil.move(str(file_path), str(dest_path))
                print(f"移动音频文件: {file_path.name} -> {dest_dir}")
    
    def organize_video_files(self):
        """归类视频文件"""
        print("\n=== 归类视频文件 ===")
        
        # 查找视频文件
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(str(self.ami_dir / "**" / ext), recursive=True))
        
        for video_file in video_files:
            file_path = Path(video_file)
            filename = file_path.name.lower()
            
            # 根据文件名判断是个人视角还是房间视角
            if any(keyword in filename for keyword in ['individual', 'person', 'speaker']):
                dest_dir = self.ami_dir / "video" / "individual"
            elif any(keyword in filename for keyword in ['room', 'overview', 'wide']):
                dest_dir = self.ami_dir / "video" / "room_view"
            else:
                # 默认归类到房间视角
                dest_dir = self.ami_dir / "video" / "room_view"
            
            dest_path = dest_dir / file_path.name
            if not dest_path.exists():
                shutil.move(str(file_path), str(dest_path))
                print(f"移动视频文件: {file_path.name} -> {dest_dir}")
    
    def organize_annotation_files(self):
        """归类标注文件"""
        print("\n=== 归类标注文件 ===")
        
        # 查找标注文件
        annotation_extensions = ['*.txt', '*.xml', '*.json', '*.csv']
        annotation_files = []
        
        for ext in annotation_extensions:
            annotation_files.extend(glob.glob(str(self.ami_dir / "**" / ext), recursive=True))
        
        for annotation_file in annotation_files:
            file_path = Path(annotation_file)
            filename = file_path.name.lower()
            
            # 根据文件名和内容判断标注类型
            if any(keyword in filename for keyword in ['transcript', 'ortho', 'words']):
                dest_dir = self.ami_dir / "annotations" / "transcripts"
            elif any(keyword in filename for keyword in ['dialogue', 'da', 'acts']):
                dest_dir = self.ami_dir / "annotations" / "dialogue_acts"
            elif any(keyword in filename for keyword in ['emotion', 'affect', 'sentiment']):
                dest_dir = self.ami_dir / "annotations" / "emotions"
            elif any(keyword in filename for keyword in ['gesture', 'head', 'hand', 'gaze']):
                dest_dir = self.ami_dir / "annotations" / "gestures"
            else:
                # 默认归类到转录
                dest_dir = self.ami_dir / "annotations" / "transcripts"
            
            dest_path = dest_dir / file_path.name
            if not dest_path.exists():
                shutil.move(str(file_path), str(dest_path))
                print(f"移动标注文件: {file_path.name} -> {dest_dir}")
    
    def organize_slides_and_whiteboard(self):
        """归类幻灯片和白板数据"""
        print("\n=== 归类幻灯片和白板数据 ===")
        
        # 查找幻灯片和白板文件
        slide_extensions = ['*.ppt', '*.pptx', '*.pdf']
        whiteboard_extensions = ['*.png', '*.jpg', '*.jpeg']
        
        # 处理幻灯片文件
        for ext in slide_extensions:
            slide_files = glob.glob(str(self.ami_dir / "**" / ext), recursive=True)
            for slide_file in slide_files:
                file_path = Path(slide_file)
                dest_path = self.ami_dir / "slides" / file_path.name
                if not dest_path.exists():
                    shutil.move(str(file_path), str(dest_path))
                    print(f"移动幻灯片文件: {file_path.name} -> slides/")
        
        # 处理白板文件
        for ext in whiteboard_extensions:
            whiteboard_files = glob.glob(str(self.ami_dir / "**" / ext), recursive=True)
            for whiteboard_file in whiteboard_files:
                file_path = Path(whiteboard_file)
                filename = file_path.name.lower()
                
                # 判断是否为白板图像
                if any(keyword in filename for keyword in ['whiteboard', 'board', 'drawing']):
                    dest_path = self.ami_dir / "whiteboard" / file_path.name
                    if not dest_path.exists():
                        shutil.move(str(file_path), str(dest_path))
                        print(f"移动白板文件: {file_path.name} -> whiteboard/")
    
    def generate_summary(self):
        """生成数据摘要"""
        print("\n=== AMI数据集摘要 ===")
        
        summary = {}
        for root, dirs, files in os.walk(self.ami_dir):
            rel_path = Path(root).relative_to(self.ami_dir)
            if rel_path != Path('.'):
                summary[str(rel_path)] = len(files)
        
        print("各目录文件数量:")
        for dir_path, file_count in sorted(summary.items()):
            print(f"  {dir_path}: {file_count} 个文件")
        
        total_files = sum(summary.values())
        print(f"\n总文件数: {total_files}")
    
    def run_organization(self):
        """运行完整的数据归类流程"""
        print("开始AMI数据归类...")
        
        self.organize_audio_files()
        self.organize_video_files()
        self.organize_annotation_files()
        self.organize_slides_and_whiteboard()
        self.generate_summary()
        
        print("\nAMI数据归类完成！")
        print(f"数据已整理到: {self.ami_dir}")

if __name__ == "__main__":
    organizer = AMIDataOrganizer()
    organizer.run_organization()
