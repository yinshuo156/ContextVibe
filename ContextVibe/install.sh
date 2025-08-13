#!/bin/bash
# ContextVibe Installation Script

echo "🎯 ContextVibe Multimodal Meeting Emotion Analysis System"
echo "Version: 1.0.0"
echo "============================================================"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python version check passed: $python_version"
else
    echo "❌ Python version too low, requires 3.8 or higher, current version: $python_version"
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Install ContextVibe package
echo "📦 Installing ContextVibe package..."
pip3 install -e .

echo "============================================================"
echo "🎉 ContextVibe installation completed!"
echo "============================================================"
echo ""
echo "📚 Usage:"
echo "  1. Command line tool: contextvibe --help"
echo "  2. Example program: python sample_main.py"
echo "  3. Python API: from contextvibe import VAE_CCalculator"
echo ""
echo "🚀 Quick start:"
echo "  contextvibe --full-pipeline"
echo ""
