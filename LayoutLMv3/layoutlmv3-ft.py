# LayoutLMv3 Fine-Tuning Pipeline

import os
import sys
import json
import torch
import numpy as np
from torch.optim import AdamW

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from trainer import ModelModule
from loader import dataSet
from engine import train_fn, eval_fn, plot_metrics
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, LayoutLMv3Processor

def start_training():
    base_dir = SCRIPT_DIR
    model_path_local = os.path.join(base_dir, "inputs", "layoutlmv3Microsoft")
    model_path = model_path_local if os.path.exists(model_path_local) else "microsoft/layoutlmv3-base"

    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_path, ignore_mismatched_sizes=True)
    processor = LayoutLMv3Processor(tokenizer=tokenizer, image_processor=feature_extractor)

    train_json = os.path.join(base_dir, "inputs", "Training_layoutLMV3.json")
    if not os.path.exists(train_json):
        print(f"Training dataset not found at {train_json}. Please place your training JSON in 'inputs/'.")
        return

    ds = dataSet(train_json, processor)
    
    batch_size = 8 
    num_epochs = 50
    accumulation_steps = 2
    
    dataload = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = ModelModule(49) 

    config_path = os.path.join(base_dir, "inputs", "label_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    id2label = {int(v): k for k, v in config['label2id'].items()}

    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_loss = np.inf

    train_losses, val_losses, precisions, recalls, f1_scores = [], [], [], [], []

    print(f"--- Starting LayoutLMv3 Fine-tuning ---")
    print(f"Batch size: {batch_size} (Effective: {batch_size*accumulation_steps}), Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        train_loss = train_fn(dataload, model, optimizer, accumulation_steps=accumulation_steps)
        eval_loss, precision, recall, f1 = eval_fn(dataload, model, id2label)
        
        train_losses.append(train_loss)
        val_losses.append(eval_loss)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {eval_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        plot_metrics(train_losses, precisions)

        if eval_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(base_dir, 'model.bin'))
            best_loss = eval_loss

    print("Training Complete. Final model saved as 'model.bin'.")

if __name__ == "__main__":
    start_training()
