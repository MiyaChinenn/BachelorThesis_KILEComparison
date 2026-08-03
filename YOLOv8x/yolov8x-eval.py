import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# 1. SETUP PATHS
BASE_DIR = Path(__file__).parent.absolute()
PRED_DIR = BASE_DIR / "Evaluate" / "inference_finetuned_model" / "annotations"
# Assuming GT is here
GT_DIR = BASE_DIR / "Evaluate" / "ground-truth annotations"
GEN_DIR = BASE_DIR / "Evaluate"

# Scaling factors for Ground Truth (from 150 DPI to 1280x1664)
SCALE_X = 1280 / 1275
SCALE_Y = 1664 / 1650

# Label Mapping to handle discrepancies
LABEL_MAP = {
    "vendor_phone": "sender_phone",
    "vendor_name": "sender_name",
    "vendor_address": "sender_address",
    "vendor_vat_id": "sender_vat_id",
    # Add others if found
}

def clean_label(label):
    if not label: return ""
    # Strip B- or I- prefixes
    if label.startswith(("B-", "I-")):
        label = label[2:]
    # Map to standard
    return LABEL_MAP.get(label, label)

def calculate_iou(box1, box2):
    ix0, iy0 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix1, iy1 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    b1_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_a = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = b1_a + b2_a - inter
    return inter / union if union > 0 else 0

def normalize_text(text):
    if not text: return ""
    # Strip everything except alphanumeric characters
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def evaluate_files():
    target_files = [
        "invoice_CPI007529644.json",
        "invoice_CPI007529714.json",
        "invoice_CPI007534847.json",
        "invoice_CPI007534930.json",
        "invoice_CPI007529621.json"
    ]
    pred_files = [PRED_DIR / f for f in target_files if (PRED_DIR / f).exists()]
    
    if not pred_files:
        print("No matching prediction files found.")
        return

    all_results = []
    print(f"Evaluating Header/Footer fields for {len(pred_files)} specific files...")
    
    for p_file in tqdm(pred_files):
        clean_name = p_file.stem.split("_page")[0] + ".json"
        gt_file = GT_DIR / clean_name
        if not gt_file.exists(): continue
            
        with open(p_file, 'r') as f:
            p_raw = json.load(f)
            preds = [p for p in p_raw if not p["label"].startswith("item_")] 
        with open(gt_file, 'r') as f:
            gt_data = [g for g in json.load(f)["field_extractions"] if not g["fieldtype"].startswith(("item_", "line_item_"))]
            
        gt_processed = []
        for item in gt_data:
            if item.get("bbox"):
                b = item["bbox"]
                item["bbox_scaled"] = [b[0]*SCALE_X, b[1]*SCALE_Y, b[2]*SCALE_X, b[3]*SCALE_Y]
                item["field_clean"] = clean_label(item["fieldtype"])
                gt_processed.append(item)

        metrics = {"file": p_file.name, "text_tp": 0, "text_fp": 0, "text_fn": 0, "bbox_tp": 0, "bbox_fp": 0, "bbox_fn": 0}
        matched_gt_indices = set()
        
        for p in preds:
            p_label = clean_label(p["label"])
            p_text_norm = normalize_text(p["text"])
            p_bbox = p["bbox"]
            
            best_iou, best_gt_idx = 0, -1
            for i, gt in enumerate(gt_processed):
                if i in matched_gt_indices: continue
                if gt["field_clean"] != p_label: continue
                iou = calculate_iou(p_bbox, gt["bbox_scaled"])
                if iou > best_iou: best_iou, best_gt_idx = iou, i
            
            if best_gt_idx != -1 and best_iou > 0.9:
                metrics["bbox_tp"] += 1
                gt_text_norm = normalize_text(gt_processed[best_gt_idx]["text"])
                if (gt_text_norm in p_text_norm) or (p_text_norm in gt_text_norm):
                    metrics["text_tp"] += 1
                else: 
                    metrics["text_fp"] += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                metrics["bbox_fp"] += 1
                metrics["text_fp"] += 1
        
        metrics["bbox_fn"] = len(gt_processed) - metrics["bbox_tp"]
        metrics["text_fn"] = len(gt_processed) - metrics["text_tp"]
        
        def get_scores(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
            return p, r, f1

        tp, tr, tf1 = get_scores(metrics["text_tp"], metrics["text_fp"], metrics["text_fn"])
        bp, br, bf1 = get_scores(metrics["bbox_tp"], metrics["bbox_fp"], metrics["bbox_fn"])
        
        all_results.append({
            "file": p_file.name,
            "text_precision": tp, "text_recall": tr, "text_f1": tf1, "text_accuracy": tf1,
            "bbox_precision": bp, "bbox_recall": br, "bbox_f1": bf1, "bbox_accuracy": bf1,
            "field_count": len(preds),
            "char_count": sum(len(str(p.get("text", ""))) for p in p_raw)
        })

    df = pd.DataFrame(all_results)
    
    # 1. Dashboard Layout
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    
    # Panel 0,0: Overall Summary
    overall = pd.DataFrame([
        {"metric": "accuracy", "text": df["text_accuracy"].mean(), "bbox": df["bbox_accuracy"].mean()},
        {"metric": "precision", "text": df["text_precision"].mean(), "bbox": df["bbox_precision"].mean()},
        {"metric": "recall", "text": df["text_recall"].mean(), "bbox": df["bbox_recall"].mean()},
        {"metric": "f1", "text": df["text_f1"].mean(), "bbox": df["bbox_f1"].mean()},
    ])
    sns.heatmap(overall.set_index("metric").T, annot=True, fmt=".3f", cmap="Blues", ax=axes[0, 0], vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Overall Text vs BBox Metrics")
    
    # Panel 0,1: Per-File Heatmap (Top 15 files for readability)
    heatmap_cols = ["text_accuracy", "text_precision", "text_recall", "text_f1", "bbox_accuracy", "bbox_precision", "bbox_recall", "bbox_f1"]
    df_heat = df.sort_values("text_f1", ascending=False).head(15)
    sns.heatmap(df_heat.set_index("file")[heatmap_cols], annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0, 1], cbar=False, vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Per-File Metrics Heatmap (Top 15)")
    
    # Panel 1,0: Per-File F1 Comparison (Top 15)
    x_labels = df_heat["file"].tolist()
    axes[1, 0].bar([i-0.2 for i in range(len(x_labels))], df_heat["text_f1"], width=0.4, label="Text F1", color="#2ecc71")
    axes[1, 0].bar([i+0.2 for i in range(len(x_labels))], df_heat["bbox_f1"], width=0.4, label="BBox F1", color="#3498db")
    axes[1, 0].set_xticks(range(len(x_labels)))
    axes[1, 0].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    axes[1, 0].set_title("Per-File F1 Comparison")
    axes[1, 0].legend()
    
    # Remove the 4th subplot (axes[1, 1]) to leave it blank
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(GEN_DIR / "final_evaluation_dashboard.png", dpi=200)
    print(f"Saved: {GEN_DIR / 'final_evaluation_dashboard.png'}")

if __name__ == "__main__":
    evaluate_files()


