import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher

# 1. SETUP PATHS
BASE_DIR = Path(__file__).parent.absolute()
PRED_DIR = BASE_DIR / "Evaluate" / "inference_finetuned_model" / "annotations"
# Assuming GT is here
GT_DIR = BASE_DIR / "Evaluate" / "ground-truth annotations"
GEN_DIR = BASE_DIR / "Evaluate"

# Scaling factors for Ground Truth
if "LayoutLMv3" in str(BASE_DIR):
    # Scale Ground Truth (1275x1650) to LayoutLMv3 normalized 1000x1000 coordinate system
    SCALE_X = 1000 / 1275
    SCALE_Y = 1000 / 1650
else:
    # Scaling factors for Ground Truth (from 150 DPI to 1280x1664 for YOLOv8x)
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

def is_fuzzy_match(s1, s2, threshold=0.85):
    s1_norm = normalize_text(s1)
    s2_norm = normalize_text(s2)
    if not s1_norm or not s2_norm:
        return False
    if s1_norm in s2_norm or s2_norm in s1_norm:
        return True
    matcher = SequenceMatcher(None, s1_norm, s2_norm)
    return matcher.ratio() >= threshold

def should_merge(item1, item2, label):
    b1 = item1["box"]
    b2 = item2["box"]
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    
    # Overlaps
    overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    overlap_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    
    # 1. Horizontally adjacent (same line, e.g. words on the same line)
    same_line = (overlap_y > 0.3 * min(h1, h2))
    horiz_close = (max(b1[0], b2[0]) - min(b1[2], b2[2]) <= 60)
    
    if same_line and horiz_close:
        return True
        
    # Vertical merging is only allowed for address fields to prevent over-merging on multiple taxes/fields
    if not label.endswith("address"):
        return False
        
    # 2. Vertically stacked (lines in the same block, e.g. address block)
    vert_stack = (overlap_x > 0.2 * min(w1, w2))
    vert_close = (max(b1[1], b2[1]) - min(b1[3], b2[3]) <= 40)
    
    # 3. Extreme proximity (left-aligned lines)
    left_aligned = (abs(b1[0] - b2[0]) <= 50)
    vert_close_any = (max(b1[1], b2[1]) - min(b1[3], b2[3]) <= 35)
    
    return (vert_stack and vert_close) or (left_aligned and vert_close_any)

def merge_spatial_tokens(preds, conf_threshold=0.25):
    from collections import defaultdict
    # Filter by confidence
    preds_filtered = [p for p in preds if p.get("confidence", 1.0) > conf_threshold]
    
    # Group by clean label
    label_groups = defaultdict(list)
    for p in preds_filtered:
        label = clean_label(p.get("label", ""))
        box = p.get("box_2d") or p.get("bbox")
        if not box or label == "O":
            continue
        label_groups[label].append({
            "text": p.get("text", ""),
            "box": list(box),
            "confidence": p.get("confidence", 1.0)
        })
        
    merged_preds = []
    
    for label, items in label_groups.items():
        # Find connected components of items that should merge
        clusters = [[item] for item in items]
        
        merged_any = True
        while merged_any:
            merged_any = False
            new_clusters = []
            used = set()
            for i in range(len(clusters)):
                if i in used:
                    continue
                current_cluster = clusters[i]
                for j in range(i + 1, len(clusters)):
                    if j in used:
                        continue
                    # Check if any item in current_cluster should merge with any item in clusters[j]
                    can_merge = False
                    for item1 in current_cluster:
                        for item2 in clusters[j]:
                            if should_merge(item1, item2, label):
                                can_merge = True
                                break
                        if can_merge:
                            break
                    if can_merge:
                        current_cluster.extend(clusters[j])
                        used.add(j)
                        merged_any = True
                new_clusters.append(current_cluster)
            clusters = new_clusters
            
        # Reconstruct each cluster
        for cluster in clusters:
            # Sort by top-to-bottom, then left-to-right
            sorted_items = sorted(cluster, key=lambda x: (x["box"][1], x["box"][0]))
            
            # Combine text
            text = " ".join(item["text"] for item in sorted_items)
            
            # Combine bounding box
            min_x = min(item["box"][0] for item in sorted_items)
            min_y = min(item["box"][1] for item in sorted_items)
            max_x = max(item["box"][2] for item in sorted_items)
            max_y = max(item["box"][3] for item in sorted_items)
            box = [min_x, min_y, max_x, max_y]
            
            # Confidence is the mean of cluster confidences
            confidence = sum(item["confidence"] for item in sorted_items) / len(sorted_items)
            
            merged_preds.append({
                "label": label,
                "text": text,
                "box": box,
                "confidence": confidence
            })
            
    return merged_preds

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

def evaluate_files(num_samples=5, random_seed=42):
    SET_EVAL_DIR = BASE_DIR / "set_eval"
    pdf_files = sorted(list(SET_EVAL_DIR.glob("*.pdf")))
    
    if not pdf_files:
        print(f"[WARN] No PDF files found in {SET_EVAL_DIR}")
        return

    # Select num_samples out of available set_eval PDFs
    if num_samples >= len(pdf_files):
        sampled_pdfs = pdf_files
        print(f"Evaluating ALL {len(sampled_pdfs)} files from 'set_eval':")
    else:
        random.seed(random_seed)
        sampled_pdfs = random.sample(pdf_files, num_samples)
        print(f"Randomly selected {len(sampled_pdfs)} files out of {len(pdf_files)} from 'set_eval':")
    target_stems = [p.stem.replace("_page0", "") for p in sampled_pdfs]

    for s in target_stems:
        print(f"  - {s}.pdf")

    # Build list of prediction files
    pred_files = []
    for stem in target_stems:
        p_json = PRED_DIR / f"{stem}.json"
        p_page = PRED_DIR / f"{stem}_page0.json"
        if p_page.exists():
            pred_files.append(p_page)
        elif p_json.exists():
            pred_files.append(p_json)
    
    if not pred_files:
        print("No matching prediction files found for the sampled PDFs.")
        return

    all_results = []
    print(f"\nEvaluating Header/Footer fields for {len(pred_files)} sampled files (IoU > 0.60)...")
    
    for p_file in tqdm(pred_files):
        clean_name = p_file.stem.split("_page")[0] + ".json"
        gt_file = GT_DIR / clean_name
        if not gt_file.exists(): continue
            
        with open(p_file, 'r') as f:
            p_raw = json.load(f)
            
        # Support both dictionary format with "predictions" key and direct list format
        if isinstance(p_raw, dict) and "predictions" in p_raw:
            p_list = p_raw["predictions"]
        else:
            p_list = p_raw
            
        # Merge spatially and filter by confidence
        preds_merged = merge_spatial_tokens(p_list, conf_threshold=0.25)
            
        # Filter out item fields
        preds = [p for p in preds_merged if not p["label"].startswith("item_")]
        
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
            p_label = p["label"]
            p_text = p["text"]
            p_bbox = p["box"]
            
            best_iou, best_gt_idx = 0, -1
            for i, gt in enumerate(gt_processed):
                if i in matched_gt_indices: continue
                if gt["field_clean"] != p_label: continue
                iou = calculate_iou(p_bbox, gt["bbox_scaled"])
                if iou > best_iou: best_iou, best_gt_idx = iou, i
            
            if best_gt_idx != -1 and best_iou > 0.6:
                metrics["bbox_tp"] += 1
                gt_text = gt_processed[best_gt_idx]["text"]
                if is_fuzzy_match(gt_text, p_text, threshold=0.85):
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
            "char_count": sum(len(str(p.get("text", ""))) for p in p_list)
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
    
    # Panel 1,1: Field Count vs Character Count (Proxy for Complexity)
    sns.scatterplot(data=df, x="char_count", y="field_count", hue="text_f1", size="bbox_f1", palette="magma", ax=axes[1, 1])
    axes[1, 1].set_title("Extraction Complexity vs Accuracy")
    axes[1, 1].set_xlabel("Total Characters")
    axes[1, 1].set_ylabel("Total Fields Detected")
    
    plt.tight_layout()
    plt.savefig(GEN_DIR / "final_evaluation_dashboard.png", dpi=200)
    print(f"Saved: {GEN_DIR / 'final_evaluation_dashboard.png'}")
    
    print("\n=== EVALUATION RESULTS (IoU > 0.6) ===")
    print(df.to_string(index=False))
    print("\n--- Mean Scores ---")
    print(f"Text Precision: {df['text_precision'].mean():.4f}")
    print(f"Text Recall:    {df['text_recall'].mean():.4f}")
    print(f"Text F1:        {df['text_f1'].mean():.4f}")
    print(f"BBox Precision: {df['bbox_precision'].mean():.4f}")
    print(f"BBox Recall:    {df['bbox_recall'].mean():.4f}")
    print(f"BBox F1:        {df['bbox_f1'].mean():.4f}")

if __name__ == "__main__":
    evaluate_files()


