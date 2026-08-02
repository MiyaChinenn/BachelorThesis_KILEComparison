# Comparison script to evaluate Fine-tuned vs Base LayoutLMv3 models

import os
import json
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINE_TUNED_DIR = os.path.join(PROJECT_ROOT, "inference_annotations", "annotations")
BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, "inference_base_model", "annotations_base")

def analyze_folder(folder_path):
    stats = {
        "total_files": 0,
        "total_entities": 0,
        "label_counts": {},
        "confidences": []
    }
    
    if not os.path.exists(folder_path):
        return None
        
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue
            
        stats["total_files"] += 1
        with open(os.path.join(folder_path, file_name), "r") as f:
            data = json.load(f)
            preds = data.get("predictions", [])
            stats["total_entities"] += len(preds)
            
            for p in preds:
                label = p["label"]
                conf = p["confidence"]
                stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1
                stats["confidences"].append(conf)
                
    if stats["confidences"]:
        stats["avg_confidence"] = sum(stats["confidences"]) / len(stats["confidences"])
    else:
        stats["avg_confidence"] = 0
        
    return stats

def main():
    print("--- Model Comparison Study ---\n")
    
    ft_stats = analyze_folder(FINE_TUNED_DIR)
    bs_stats = analyze_folder(BASE_MODEL_DIR)
    
    if not ft_stats or not bs_stats:
        print("[Error] Results folders not found. Please run the inference scripts first.")
        return

    # 1. High Level Summary
    summary = {
        "Metric": ["Total Images", "Total Entities Found", "Avg Confidence"],
        "Fine-Tuned (model.bin)": [
            ft_stats["total_files"],
            ft_stats["total_entities"],
            f"{ft_stats['avg_confidence']:.2%}"
        ],
        "Base Model (microsoft/layoutlmv3-base)": [
            bs_stats["total_files"],
            bs_stats["total_entities"],
            f"{bs_stats['avg_confidence']:.2%}"
        ]
    }
    
    df_summary = pd.DataFrame(summary)
    print("SUMMARY STATS:")
    print(df_summary.to_string(index=False))
    print("\n" + "="*50 + "\n")

    # 2. Per-Label Breakdown
    all_labels = sorted(list(set(ft_stats["label_counts"].keys()) | set(bs_stats["label_counts"].keys())))
    
    label_breakdown = []
    for label in all_labels:
        label_breakdown.append({
            "Label": label,
            "Fine-Tuned Count": ft_stats["label_counts"].get(label, 0),
            "Base Model Count": bs_stats["label_counts"].get(label, 0),
            "Diff": ft_stats["label_counts"].get(label, 0) - bs_stats["label_counts"].get(label, 0)
        })
    
    df_labels = pd.DataFrame(label_breakdown)
    print("PER-LABEL DETECTION COUNT:")
    print(df_labels.to_string(index=False))
    
    # 3. Insight for the USER
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    if ft_stats["total_entities"] > bs_stats["total_entities"]:
        print(f"-> Fine-tuned model found {ft_stats['total_entities'] - bs_stats['total_entities']} MORE invoice-specific entities.")
    
    if ft_stats["avg_confidence"] > bs_stats["avg_confidence"]:
        print(f"-> Fine-tuned model is {ft_stats['avg_confidence']*100 - bs_stats['avg_confidence']*100:.1f}% more confident on average.")
    
    missing_in_ft = [l for l in all_labels if ft_stats["label_counts"].get(l, 0) == 0]
    if missing_in_ft:
        print(f"-> Labels NEVER found by Fine-tuned: {missing_in_ft}")
    else:
        print("-> The Fine-tuned model is effectively detecting all defined label types across the dataset.")

if __name__ == "__main__":
    main()
