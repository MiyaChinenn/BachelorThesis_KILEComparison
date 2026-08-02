# Fine-tuning Summary

## Model to Finetune
- `yolov8x.pt` (Ultralytics YOLOv8x Object Detection Model)

## Input
- **Images**: invoice images / document page renders
- **Labels**: DocILE-style bounding box annotations (52 field classes: 41 KILE header fields + 11 line-item fields)

## Data Format
- **Input format**: document image (RGB or 6-channel warm-start context)
- **Target format**: YOLO bounding box format `[class_id, x_center, y_center, width, height]` mapped across 52 DocILE classes

## Training Parameters
- **Epochs**: `100`
- **Batch size**: `8`
- **Image resolution (`imgsz`)**: `(1664, 1280)`
- **Optimizer**: `AdamW`
- **Learning rate (`lr0`)**: `0.001`
- **Weight decay**: `0.0005`
- **Workers**: `4`
- **Evaluation IoU threshold**: `0.6`

## Step
1. Load dataset configuration (`docile.yaml` / `single_template.yaml`).
2. Load pretrained `yolov8x.pt` model weights.
3. Migrate first convolution layer to 6-channel warm start when utilizing extra document context channels.
4. Train model using Ultralytics `YOLO` trainer with 1664x1280 image resolution.
5. Validate detection performance using NMS with IoU threshold = `0.6`.
6. Save trained model weights (`best.pt`).

## Training Method
- **Framework**: `Ultralytics YOLOv8`
- **Trainer**: `ultralytics.YOLO` Object Detection Trainer
- **Task**: Bounding box object detection (`task: detect`)

## Expected Result
- A fine-tuned object detection model specialized for **invoice document layout region localization**.
- Bounding box predictions for:
  - invoice header & identity fields
  - amount summary fields
  - tax fields
  - line item rows and cells

## Resource Consumption
- **GPU**: NVIDIA GPU (recommended 16GB+ VRAM for 1664x1280 batch 8)
- **Memory optimization**:
  - `Automatic Mixed Precision (AMP)`
  - `Rectangular training (rect=True)`
- **Expected usage**:
  - high GPU memory utilization due to high-resolution input (1664x1280)
  - fast detection inference speed
