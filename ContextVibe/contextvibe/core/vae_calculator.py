#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V、A、E、C得分计算模块
- V (Valence): 情感效价
- A (Arousal): 情感唤醒度
- E (Energy): 交互同步性
- C (Cohesion): 上下文一致性
"""

import numpy as np
import pandas as pd
import librosa
import cv2
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.signal import correlate
import torch
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VACalculator:
    """Valence and Arousal Calculator - 使用MAVEN多模态情感分析"""
    
    def __init__(self, use_maven: bool = True, model_path: Optional[str] = None):
        """Initialize sentiment analysis model"""
        self.use_maven = use_maven
        
        if use_maven:
            try:
                # 导入MAVEN模块
                from .maven_integration import MAVENVACalculator
                self.maven_calculator = MAVENVACalculator(model_path=model_path)
                logger.info("MAVEN VACalculator initialized successfully")
            except Exception as e:
                logger.warning(f"MAVEN初始化失败，回退到传统方法: {e}")
                self.use_maven = False
        
        if not self.use_maven:
            # 回退到传统方法
            try:
                # Download necessary NLTK data
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                
                # Initialize transformers sentiment analysis model (use local model or simplified version)
                try:
                    self.text_sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=0 if torch.cuda.is_available() else -1,
                        mirror="https://hf-mirror.com"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load online model, using simplified version: {e}")
                    self.text_sentiment_pipeline = None
                
                logger.info("Traditional VACalculator initialized successfully")
            except Exception as e:
                logger.error(f"VACalculator initialization failed: {e}")
    
    def calculate_text_va(self, text: str) -> Tuple[float, float]:
        """Calculate Valence and Arousal for text"""
        if self.use_maven:
            try:
                return self.maven_calculator.calculate_text_va(text)
            except Exception as e:
                logger.error(f"MAVEN text VA calculation failed: {e}")
                # 回退到传统方法
                pass
        
        # 传统方法
        try:
            # Use VADER for sentiment analysis
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # VADER returns compound score as valence
            valence = scores['compound']
            
            # Use difference between positive and negative scores as arousal
            arousal = scores['pos'] - scores['neg']
            
            return valence, arousal
        except Exception as e:
            logger.error(f"Text VA calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_audio_va(self, audio_path: str) -> Tuple[float, float]:
        """Calculate Valence and Arousal for audio"""
        if self.use_maven:
            try:
                return self.maven_calculator.calculate_audio_va(audio_path)
            except Exception as e:
                logger.error(f"MAVEN audio VA calculation failed: {e}")
                # 回退到传统方法
                pass
        
        # 传统方法
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract audio features
            # Pitch features (valence related)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[magnitudes > 0.1])
            
            # Energy features (arousal related)
            energy = np.mean(librosa.feature.rms(y=y))
            
            # Spectral centroid (valence related)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Normalize features
            valence = (pitch_mean / 1000.0 + spectral_centroid / 5000.0) / 2
            arousal = energy / 0.1
            
            # Limit to [-1, 1] range
            valence = np.clip(valence, -1, 1)
            arousal = np.clip(arousal, -1, 1)
            
            return valence, arousal
        except Exception as e:
            logger.error(f"Audio VA calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_video_va(self, video_path: str) -> Tuple[float, float]:
        """Calculate Valence and Arousal for video"""
        if self.use_maven:
            try:
                return self.maven_calculator.calculate_video_va(video_path)
            except Exception as e:
                logger.error(f"MAVEN video VA calculation failed: {e}")
                # 回退到传统方法
                pass
        
        # 传统方法 - 简单实现
        try:
            # 对于视频，我们主要分析音频部分
            # 这里可以扩展为更复杂的视频分析
            logger.warning("Traditional video VA calculation not implemented, returning default values")
            return 0.0, 0.0
        except Exception as e:
            logger.error(f"Video VA calculation failed: {e}")
            return 0.0, 0.0

class EnergyCalculator:
    """Energy (Interaction Synchrony) Calculator"""
    
    def __init__(self):
        """Initialize synchrony calculator"""
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        logger.info("EnergyCalculator initialized successfully")
    
    def calculate_audio_energy(self, audio_path: str) -> float:
        """Calculate audio energy features"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate volume intensity
            rms = librosa.feature.rms(y=y)
            volume_intensity = np.mean(rms)
            
            # Calculate speech rhythm
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            speech_rhythm = tempo / 200.0  # Normalize
            
            # Calculate energy score
            energy = (volume_intensity + speech_rhythm) / 2
            return float(np.clip(energy, 0, 1))
        except Exception as e:
            logger.error(f"Audio energy calculation failed: {e}")
            return 0.0
    
    def calculate_video_energy(self, video_path: str) -> float:
        """Calculate video energy features"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            motion_scores = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10 frames
                if frame_count % 10 == 0:
                    # Convert to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Face detection
                    results = self.mp_face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        # Calculate facial motion
                        landmarks = results.multi_face_landmarks[0]
                        motion_score = self._calculate_facial_motion(landmarks)
                        motion_scores.append(motion_score)
                
                frame_count += 1
            
            cap.release()
            
            if motion_scores:
                return float(np.mean(motion_scores))
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Video energy calculation failed: {e}")
            return 0.0
    
    def _calculate_facial_motion(self, landmarks) -> float:
        """Calculate facial motion score"""
        try:
            # Extract keypoints
            points = []
            for landmark in landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
            
            points = np.array(points)
            
            # Calculate facial motion (simplified version)
            # This can be extended to more complex AU detection
            motion_score = np.std(points[:, 2])  # Z-axis variation
            return np.clip(motion_score * 10, 0, 1)
        except Exception as e:
            logger.error(f"Facial motion calculation failed: {e}")
            return 0.0
    
    def calculate_synchrony(self, audio_energy: float, video_energy: float) -> float:
        """Calculate audio-video synchrony"""
        try:
            # Simple synchrony calculation
            synchrony = 1 - abs(audio_energy - video_energy)
            return np.clip(synchrony, 0, 1)
        except Exception as e:
            logger.error(f"Synchrony calculation failed: {e}")
            return 0.0

class CohesionCalculator:
    """Cohesion (Contextual Consistency) Calculator
    
    Contains two dimensions:
    1. Personal consistency - reflects state stability
    2. Environmental consistency - reflects integration
    """
    
    def __init__(self):
        """Initialize consistency calculator"""
        try:
            # Initialize text embedding model
            self.text_embedding_pipeline = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-MiniLM-L6-v2",
                device=0 if torch.cuda.is_available() else -1,
                mirror="https://hf-mirror.com"
            )
            logger.info("CohesionCalculator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load online model, using simplified version: {e}")
            self.text_embedding_pipeline = None
        
        # Initialize sentiment analyzer for state detection
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def calculate_text_cohesion(self, texts: List[str]) -> float:
        """Calculate text cohesion"""
        try:
            if len(texts) < 2:
                return 1.0
            
            # If no embedding model, use simple vocabulary overlap method
            if self.text_embedding_pipeline is None:
                return self._calculate_simple_cohesion(texts)
            
            # Get text embeddings
            embeddings = []
            for text in texts:
                embedding = self.text_embedding_pipeline(text)
                embeddings.append(embedding[0][0])  # Take first token embedding
            
            embeddings = np.array(embeddings)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Calculate average cohesion
            cohesion = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            return np.clip(cohesion, 0, 1)
        except Exception as e:
            logger.error(f"Text cohesion calculation failed: {e}")
            return self._calculate_simple_cohesion(texts)
    
    def _calculate_simple_cohesion(self, texts: List[str]) -> float:
        """Calculate text cohesion using simple method"""
        try:
            if len(texts) < 2:
                return 1.0
            
            # Use vocabulary overlap to calculate cohesion
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    words1 = set(texts[i].lower().split())
                    words2 = set(texts[j].lower().split())
                    
                    if len(words1) == 0 or len(words2) == 0:
                        similarity = 0.0
                    else:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
        except Exception as e:
            logger.error(f"Simple cohesion calculation failed: {e}")
            return 0.0
    
    def calculate_comprehensive_cohesion(self, session_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive cohesion score
        
        Returns:
            Dict containing:
            - personal_cohesion: Personal consistency
            - environmental_cohesion: Environmental consistency
            - overall_cohesion: Overall cohesion score
        """
        try:
            personal_cohesion = self._calculate_personal_cohesion(session_data)
            environmental_cohesion = self._calculate_environmental_cohesion(session_data)
            
            # Overall score (adjustable weights)
            overall_cohesion = 0.4 * personal_cohesion + 0.6 * environmental_cohesion
            
            return {
                'personal_cohesion': personal_cohesion,
                'environmental_cohesion': environmental_cohesion,
                'overall_cohesion': overall_cohesion
            }
        except Exception as e:
            logger.error(f"Comprehensive cohesion calculation failed: {e}")
            return {
                'personal_cohesion': 0.0,
                'environmental_cohesion': 0.0,
                'overall_cohesion': 0.0
            }
    
    def _calculate_personal_cohesion(self, session_data: Dict) -> float:
        """Calculate personal consistency (reflects state stability)"""
        try:
            speaker_texts = {}
            
            # Organize texts by speaker
            for item in session_data.get('items', []):
                if 'text' in item:
                    speaker = item.get('speaker', 'unknown')
                    if speaker not in speaker_texts:
                        speaker_texts[speaker] = []
                    speaker_texts[speaker].append(item['text'])
            
            if not speaker_texts:
                return 0.0
            
            personal_cohesions = []
            
            for speaker, texts in speaker_texts.items():
                if len(texts) < 2:
                    continue
                
                # Calculate consistency for this speaker
                if self.sentiment_analyzer:
                    # Use sentiment analysis to calculate state stability
                    sentiments = []
                    for text in texts:
                        scores = self.sentiment_analyzer.polarity_scores(text)
                        sentiments.append(scores['compound'])
                    
                    # Calculate standard deviation of sentiment changes (smaller = more stable)
                    sentiment_std = np.std(sentiments)
                    stability = 1.0 / (1.0 + sentiment_std)  # Normalize
                    personal_cohesions.append(stability)
                else:
                    # Use vocabulary consistency
                    cohesion = self._calculate_simple_cohesion(texts)
                    personal_cohesions.append(cohesion)
            
            return np.mean(personal_cohesions) if personal_cohesions else 0.0
            
        except Exception as e:
            logger.error(f"Personal cohesion calculation failed: {e}")
            return 0.0
    
    def _calculate_environmental_cohesion(self, session_data: Dict) -> float:
        """Calculate environmental consistency (reflects integration)"""
        try:
            items = session_data.get('items', [])
            if len(items) < 2:
                return 0.0
            
            # Collect all texts
            all_texts = []
            speaker_texts = {}
            
            for item in items:
                if 'text' in item:
                    text = item['text']
                    all_texts.append(text)
                    
                    speaker = item.get('speaker', 'unknown')
                    if speaker not in speaker_texts:
                        speaker_texts[speaker] = []
                    speaker_texts[speaker].append(text)
            
            if len(speaker_texts) < 2:
                return 0.0
            
            # Calculate overall environmental features
            if self.sentiment_analyzer:
                # Use sentiment analysis
                all_sentiments = []
                for text in all_texts:
                    scores = self.sentiment_analyzer.polarity_scores(text)
                    all_sentiments.append(scores['compound'])
                
                env_sentiment = np.mean(all_sentiments)
                
                # Calculate deviation of each speaker from environment
                speaker_deviations = []
                for speaker, texts in speaker_texts.items():
                    if len(texts) < 1:
                        continue
                    
                    speaker_sentiments = []
                    for text in texts:
                        scores = self.sentiment_analyzer.polarity_scores(text)
                        speaker_sentiments.append(scores['compound'])
                    
                    speaker_avg = np.mean(speaker_sentiments)
                    deviation = 1.0 - abs(speaker_avg - env_sentiment)  # Smaller difference = higher integration
                    speaker_deviations.append(deviation)
                
                return np.mean(speaker_deviations) if speaker_deviations else 0.0
            else:
                # Use vocabulary overlap
                return self._calculate_simple_cohesion(all_texts)
                
        except Exception as e:
            logger.error(f"Environmental cohesion calculation failed: {e}")
            return 0.0
    
    def calculate_speaker_cohesion(self, speaker_texts: Dict[str, List[str]]) -> float:
        """Calculate speaker cohesion (maintain backward compatibility)"""
        try:
            if len(speaker_texts) < 2:
                return 1.0
            
            # If no embedding model, use simple method
            if self.text_embedding_pipeline is None:
                return self._calculate_simple_speaker_cohesion(speaker_texts)
            
            speaker_embeddings = {}
            
            # Calculate average embedding for each speaker
            for speaker, texts in speaker_texts.items():
                if not texts:
                    continue
                
                embeddings = []
                for text in texts:
                    embedding = self.text_embedding_pipeline(text)
                    embeddings.append(embedding[0][0])
                
                speaker_embeddings[speaker] = np.mean(embeddings, axis=0)
            
            if len(speaker_embeddings) < 2:
                return 1.0
            
            # Calculate similarity between speakers
            embeddings_list = list(speaker_embeddings.values())
            similarity_matrix = cosine_similarity(embeddings_list)
            
            # Calculate average cohesion
            cohesion = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            return np.clip(cohesion, 0, 1)
        except Exception as e:
            logger.error(f"Speaker cohesion calculation failed: {e}")
            return self._calculate_simple_speaker_cohesion(speaker_texts)
    
    def _calculate_simple_speaker_cohesion(self, speaker_texts: Dict[str, List[str]]) -> float:
        """Calculate speaker cohesion using simple method"""
        try:
            if len(speaker_texts) < 2:
                return 1.0
            
            # Calculate average text length and vocabulary diversity for each speaker
            speaker_features = {}
            for speaker, texts in speaker_texts.items():
                if not texts:
                    continue
                
                avg_length = np.mean([len(text.split()) for text in texts])
                all_words = set()
                for text in texts:
                    all_words.update(text.lower().split())
                vocab_size = len(all_words)
                
                speaker_features[speaker] = [avg_length, vocab_size]
            
            if len(speaker_features) < 2:
                return 1.0
            
            # Calculate similarity between speakers
            features_list = list(speaker_features.values())
            similarities = []
            
            for i in range(len(features_list)):
                for j in range(i + 1, len(features_list)):
                    # Simple Euclidean distance similarity
                    distance = np.linalg.norm(np.array(features_list[i]) - np.array(features_list[j]))
                    similarity = 1 / (1 + distance)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
        except Exception as e:
            logger.error(f"Simple speaker cohesion calculation failed: {e}")
            return 0.0

class VAE_CCalculator:
    """V, A, E, C Comprehensive Calculator"""
    
    def __init__(self):
        """Initialize comprehensive calculator"""
        self.va_calculator = VACalculator()
        self.energy_calculator = EnergyCalculator()
        self.cohesion_calculator = CohesionCalculator()
        logger.info("VAE_CCalculator initialized successfully")
    
    def calculate_vaec_scores(self, data_path: str) -> Dict[str, float]:
        """Calculate V, A, E, C scores"""
        try:
            results = {
                'valence': 0.0,
                'arousal': 0.0,
                'energy': 0.0,
                'cohesion': 0.0
            }
            
            # Check data type and calculate corresponding scores
            if data_path.endswith(('.wav', '.mp3', '.flac')):
                # Audio files
                va, aa = self.va_calculator.calculate_audio_va(data_path)
                ea = self.energy_calculator.calculate_audio_energy(data_path)
                
                results['valence'] = va
                results['arousal'] = aa
                results['energy'] = ea
                
            elif data_path.endswith(('.mp4', '.avi', '.mov')):
                # Video files
                va, aa = self.va_calculator.calculate_video_va(data_path)
                ev = self.energy_calculator.calculate_video_energy(data_path)
                
                results['valence'] = va
                results['arousal'] = aa
                results['energy'] = ev
                
            elif data_path.endswith(('.txt', '.json')):
                # Text files
                with open(data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                vt, at = self.va_calculator.calculate_text_va(text)
                results['valence'] = vt
                results['arousal'] = at
            
            return results
        except Exception as e:
            logger.error(f"VAEC score calculation failed: {e}")
            return results
    
    def calculate_session_scores(self, session_data: Dict) -> Dict[str, float]:
        """Calculate V, A, E, C scores for entire session"""
        try:
            session_results = {
                'valence': [],
                'arousal': [],
                'energy': [],
                'cohesion': []
            }
            
            # Collect all texts for cohesion calculation
            all_texts = []
            speaker_texts = {}
            
            # Process each data item in session
            for item in session_data.get('items', []):
                if 'text' in item:
                    all_texts.append(item['text'])
                    
                    speaker = item.get('speaker', 'unknown')
                    if speaker not in speaker_texts:
                        speaker_texts[speaker] = []
                    speaker_texts[speaker].append(item['text'])
                
                # Calculate scores for individual items
                if 'file_path' in item:
                    scores = self.calculate_vaec_scores(item['file_path'])
                    for key in scores:
                        if scores[key] != 0:
                            session_results[key].append(scores[key])
            
            # Calculate final scores
            final_scores = {}
            for key in session_results:
                if session_results[key]:
                    final_scores[key] = np.mean(session_results[key])
                else:
                    final_scores[key] = 0.0
            
            # Calculate cohesion score
            if all_texts:
                final_scores['cohesion'] = self.cohesion_calculator.calculate_text_cohesion(all_texts)
            
            return final_scores
        except Exception as e:
            logger.error(f"Session score calculation failed: {e}")
            return {'valence': 0.0, 'arousal': 0.0, 'energy': 0.0, 'cohesion': 0.0}

if __name__ == "__main__":
    # 测试代码
    calculator = VAE_CCalculator()
    print("VAE_CCalculator测试完成")
