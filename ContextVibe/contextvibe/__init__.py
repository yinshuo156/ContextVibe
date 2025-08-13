#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextVibe: 多模态会议情感分析系统
"""

__version__ = "1.0.0"
__author__ = "ContextVibe Team"
__email__ = "contact@contextvibe.com"

from .core.vae_calculator import VAE_CCalculator
from .analysis.multimodal_cohesion_analyzer import MultimodalCohesionAnalyzer
from .data.organize_ami_data import AMIDataOrganizer
from .utils.processor import AMIProcessor

__all__ = [
    'VAE_CCalculator',
    'MultimodalCohesionAnalyzer', 
    'AMIDataOrganizer',
    'AMIProcessor'
]
