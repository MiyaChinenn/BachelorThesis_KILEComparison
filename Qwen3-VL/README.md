# Fine-tuning Summary

## Model to Finetune
- `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`

## Input
- **Images**: invoice images from `IMAGE_DIR`
- **Labels**: DocILE-style ground-truth JSON files from `JSON_DIR`

## Data Format
- **Input format**: image + instruction prompt in chat-message structure
- **Target format**: compact multi-pass JSON targets (4 supervised passes per document):
  - pass 1 (identity): `fields` + `metadata`
  - pass 2 (amounts): `fields` + `metadata`
  - pass 3 (tax): `tax_rows` + `metadata`
  - pass 4 (line items): `line_items` + `line_item_headers`

## Training Parameters
- **Epochs**: default `12` (configurable via env)
- **Per-device batch size**: default `1` (configurable via env)
- **Gradient accumulation**: default `4`
- **Effective batch size**: default `4`
- **Learning rate**: default `1e-4`
- **Validation split**: default `0.15` (clamped to max `0.4`)
- **Repeat factor (train split only)**: default `1`
- **Max sequence length**: default `4096`
- **Optimizer**: `adamw_8bit`
- **Scheduler**: `cosine`
- **Warmup ratio**: `0.1`
- **Weight decay**: `0.01`
- **Precision**: `bf16` if supported, otherwise `fp16`
- **Quantization**: `4-bit`

## Step
1. Load environment variables and dataset paths.
2. Load `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` with `Unsloth` in 4-bit mode.
3. Apply `LoRA/PEFT` to both vision and language layers.
4. Build the dataset from paired invoice image and JSON label files and expand each document into 4 compact supervision passes.
5. Format the dataset into chat-template text.
6. Train using `SFTTrainer`.
7. Track step loss and generation-based validation metrics: exact match, field precision, field recall, and field F1.
8. Save the trained adapter and processor.
9. Save plots and CSV metrics (`training_metrics.png`, `step_losses.csv`, `epoch_metrics.csv`).

## Training Method
- **Framework**: `Unsloth`
- **Trainer**: `TRL SFTTrainer` with Hugging Face-style config
- **PEFT method**: `LoRA`
- **Data collator**: `UnslothVisionDataCollator`

## Expected Result
- A fine-tuned vision-language adapter specialized for **compact multi-pass invoice extraction**.
- Improved extraction quality for:
  - invoice header fields
  - amount summary fields
  - tax rows
  - line items
  - metadata
  - bbox-aware outputs

## Resource Consumption
- **GPU**: `GB10`
- **Memory optimization**:
  - `4-bit loading`
  - `gradient checkpointing`
  - `8-bit optimizer`
- **Expected usage**:
  - moderate-to-high VRAM usage
  - training time depends on dataset size and number of epochs
  - more efficient than full fine-tuning
