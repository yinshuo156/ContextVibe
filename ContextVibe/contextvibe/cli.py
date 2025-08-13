#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextVibe 命令行接口
"""

import argparse
import sys
import logging
from pathlib import Path

from .utils.processor import AMIProcessor

def setup_logging(verbose=False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='ContextVibe: Multimodal Meeting Emotion Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  contextvibe --full-pipeline                    # Run complete pipeline
  contextvibe --download-sample                  # Download sample data
  contextvibe --process-sample                   # Process sample data
  contextvibe --analyze-multi-speaker            # Analyze multi-speaker sessions
  contextvibe --organize-data                    # Organize AMI data
  contextvibe --ami-dir /path/to/ami/data        # Specify AMI data directory
        """
    )
    
    parser.add_argument('--ami-dir', type=str, default='/root/lanyun-tmp/amicorpus',
                       help='AMI data directory path')
    parser.add_argument('--download-sample', action='store_true',
                       help='Download sample data')
    parser.add_argument('--process-sample', action='store_true',
                       help='Process sample data')
    parser.add_argument('--analyze-multi-speaker', action='store_true',
                       help='Analyze multi-speaker sessions')
    parser.add_argument('--organize-data', action='store_true',
                       help='Organize AMI data')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete processing pipeline')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version', version='ContextVibe 1.0.0')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        # Create processor
        processor = AMIProcessor(args.ami_dir)
        
        # Execute corresponding functions based on arguments
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
            # Default: run complete pipeline
            processor.run_full_pipeline()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
