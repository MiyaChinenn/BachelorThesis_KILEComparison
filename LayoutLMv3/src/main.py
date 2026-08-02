# Main

import torch
import os
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from trainer import *
from loader import *
from torch.optim import AdamW
import numpy as np
from engine import *


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path_local = os.path.join(base_dir, "inputs", "layoutlmv3Microsoft")
model_path = model_path_local if os.path.exists(model_path_local) else "microsoft/layoutlmv3-base"

featur_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
tokeniser = LayoutLMv3TokenizerFast.from_pretrained(model_path, ignore_mismatched_sizes=True)

processor = LayoutLMv3Processor(tokenizer=tokeniser, image_processor=featur_extractor)
# model is initialized within ModelModule below


if __name__ == "__main__":
    train_json = os.path.join(base_dir, "inputs", "Training_layoutLMV3.json")
    ds = dataSet(train_json, processor)
    
    # Configuration
    batch_size = 8 
    num_epochs = 50
    accumulation_steps = 2
    
    dataload = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    # creating model instance
    model = ModelModule(49) 

    # Load label mapping for metrics
    config_path = os.path.join(base_dir, "inputs", "label_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    id2label = {int(v): k for k, v in config['label2id'].items()}

    # optimizer and loss
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_loss = np.inf


    # Training metrics storage
    train_losses = []
    val_losses = []
    precisions = []
    recalls = []
    f1_scores = []

    print(f"Starting training on 30GB GPU with batch_size={batch_size} (Effective={batch_size*accumulation_steps})")

    for epoch in range(num_epochs):
        # Training phase
        train_loss = train_fn(dataload, model, optimizer, accumulation_steps=accumulation_steps)
        
        # Evaluation phase
        eval_loss, precision, recall, f1 = eval_fn(dataload, model, id2label)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(eval_loss)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {eval_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Update and save the dashboard plot (Simplified to Train Loss and Precision as requested)
        plot_metrics(train_losses, precisions)

        # Save best model based on val loss
        if eval_loss < best_loss:
            torch.save(model.state_dict(), './model.bin')
            best_loss = eval_loss

        # Save periodic checkpoints
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f'./model_{epoch}.bin')
            print(f"Checkpoint saved: model_{epoch}.bin")

    print(f"Training Complete. Final model saved as 'model.bin'. Metrics plot saved as 'metrics_plot.png'")
