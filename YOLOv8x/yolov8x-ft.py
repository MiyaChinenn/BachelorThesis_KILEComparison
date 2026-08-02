# YOLOv8x Fine-Tuning Pipeline

import os
import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).parent.absolute()
FT_DIR = SCRIPT_DIR / "yolov8 fine-tuning"
if str(FT_DIR) not in sys.path:
    sys.path.insert(0, str(FT_DIR))

try:
    from run_training import start_training
except ImportError as e:
    print(f"Error importing run_training: {e}")
    start_training = None

if __name__ == "__main__":
    if start_training:
        print("--- Starting YOLOv8x Fine-Tuning ---")
        start_training()
    else:
        print("Could not locate start_training module in yolov8 fine-tuning/run_training.py")
