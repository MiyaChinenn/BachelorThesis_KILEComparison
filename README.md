# Intelligent Document Processing Benchmarking: Performance Comparison on fire-tuned VLMs and Vision models

**Supervisors:**
- **Supervisor 1**: Dr. Nguyen Tuan Cuong
- **Supervisor 2**: Dr. Truong Dinh Huy

**Presented by:**
- Luu Danh Thanh Khoi
- Nguyen Thien Nguyen

---

## 📝 Abstract

Extracting structured key-value fields and tabular line items from invoices is a core bottleneck in document automation. Rule parsers fail on layout shifts, and raw OCR cascades discard spatial context encoding field meaning in semi-structured documents. Document intelligence splits into two paradigms: layout-aware vision models predicting bounding coordinates directly from spatial feature maps, and generative vision-language models bypassing OCR by mapping document canvases into language embedding space. This research benchmarks four architectures: YOLOv8x with a 6-channel Chargrid stack, LayoutLMv3, Qwen3-VL-8B-Instruct, and Gemma-4-E4B-it, on a single-template invoice extraction task under the DocILE KILE and LIR evaluation protocol. Annotated single-template corporate invoices from an enterprise environment are partitioned into training and evaluation splits. All four models follow a three-phase pipeline covering DocILE annotation, parameter-efficient fine-tuning, and benchmark evaluation under unified spatial and text-matching criteria.

Layout-aware vision models offer high spatial accuracy and throughput on constrained hardware. YOLOv8x achieves the best spatial localization, the smallest memory footprint and the fastest inference speed among all architectures. LayoutLMv3 achieves high text accuracy, but suffers from lower initial recall due to misalignments between the OCR tokens and the ground-truth annotations during pre-processing. We present a targeted post-processing correction to fill these gaps and achieve measurable F1 improvements without retraining the model. These visual architectures are well suited for high-throughput enterprise pipelines, but still rely on external OCR quality and fixed label taxonomies.

Generative vision-language models do not require upstream OCR dependencies and enable quick schema adaption by changing prompts, but at a greater computational cost. Without an OCR stage, Gemma-4-E4B-it outperforms all benchmarked networks in terms of exact-match and field-level F1 scores, but it uses more VRAM and has higher per-page latency. We provide a four-pass instruction-partitioning strategy to solve the output sequence saturation problem. Our method eliminates the requirement for retraining and enables tiny VLM architectures to produce full schemas on token-limited hardware. Whether the enterprise prioritizes hardware efficiency and execution throughput over prompt-driven schema flexibility will ultimately determine the paradigm.

---

This repository contains the empirical benchmark evaluation and comparison across four core model architectures for **Invoice Key Information Extraction (KILE)** and **Bounding Box (BBox) Localization**:

1. **Gemma4-E4B** (`google/gemma-4-E4B-it`) – Fine-tuned Vision-Language Model via Unsloth 4-bit LoRA
2. **Qwen3-VL** (`Qwen/Qwen3-VL-8B-Instruct`) – Fine-tuned Vision-Language Model via Unsloth 4-bit LoRA
3. **LayoutLMv3** (`microsoft/layoutlmv3-base`) – Multimodal Document Transformer for Token Classification
4. **YOLOv8x** (`yolov8x.pt`) – High-resolution Object Detection model for region localization

---

## 📊 Benchmark Results

<p float="left">
  <img src="./ComparisonBenchmark/accuracy-compare.png" width="49%" alt="Accuracy Comparison: Text (OCR) vs. BBox (Localization)" />
  <img src="./ComparisonBenchmark/f1-compare.png" width="49%" alt="F1-Score Comparison: Text (OCR) vs. BBox (Localization)" />
</p>

---

## 📈 Comparison Table

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric / Model</th>
      <th colspan="2" align="center">Accuracy</th>
      <th colspan="2" align="center">F1-Score</th>
    </tr>
    <tr>
      <th align="center">Text</th>
      <th align="center">Bounding Box</th>
      <th align="center">Text</th>
      <th align="center">Bounding Box</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>YOLOv8x</strong></td>
      <td align="center">0.927</td>
      <td align="center">0.979</td>
      <td align="center">0.927</td>
      <td align="center">0.979</td>
    </tr>
    <tr>
      <td><strong>LayoutLMv3</strong></td>
      <td align="center">0.850</td>
      <td align="center">0.923</td>
      <td align="center">0.850</td>
      <td align="center">0.923</td>
    </tr>
    <tr>
      <td><strong>Qwen3-VL</strong></td>
      <td align="center">0.800</td>
      <td align="center">0.781</td>
      <td align="center">0.783</td>
      <td align="center">0.764</td>
    </tr>
    <tr>
      <td><strong>Gemma4-E4B</strong></td>
      <td align="center">0.914</td>
      <td align="center">0.860</td>
      <td align="center">0.926</td>
      <td align="center">0.873</td>
    </tr>
  </tbody>
</table>

---

## 📂 Repository Structure

```text
BachelorThesis_KILEComparison/
├── ComparisonBenchmark/            # Comparative visualization charts
│   ├── accuracy-compare.png
│   └── f1-compare.png
├── Gemma4-E4B/                      # Fine-tuning & inference pipeline for Gemma4-E4B
│   ├── README.md
│   ├── gemma4-ft.py
│   └── gemma4-inference.ipynb
├── Qwen3-VL/                        # Fine-tuning & inference pipeline for Qwen3-VL
│   ├── README.md
│   ├── qwen3-ft.py
│   └── qwen3-inference.ipynb
├── LayoutLMv3/                      # Multimodal token classification pipeline
│   ├── README.md
│   ├── layoutlmv3-ft.py
│   ├── layoutlmv3-ft.ipynb
│   ├── layoutlmv3-inference.ipynb
│   └── layoutlmv3-eval.py
└── YOLOv8x/                         # High-resolution object detection pipeline
    ├── README.md
    ├── yolov8x-ft.py
    ├── yolov8x-inference.ipynb
    └── yolov8x-eval.py
```
