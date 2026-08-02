import os
import json
import math
import ssl
from collections import defaultdict

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

import torch
import requests
import pandas as pd
from datasets import Dataset
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
from transformers import TrainerCallback
from requests.adapters import HTTPAdapter
from trl import SFTTrainer, SFTConfig
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import urllib3


def maybe_disable_ssl_verification():
    """Optional fallback for restricted environments that break HF downloads."""
    if os.getenv("DISABLE_SSL_VERIFY", "0") != "1":
        return

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    class SSLIgnoreAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = ssl._create_unverified_context()
            return super().init_poolmanager(*args, **kwargs)

    _old_session = requests.Session

    class PatchedSession(_old_session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.verify = False
            self.mount("https://", SSLIgnoreAdapter())

    requests.Session = PatchedSession
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    print("⚠ SSL verification disabled via DISABLE_SSL_VERIFY=1")


# 1. Load configuration from .env file
load_dotenv()
maybe_disable_ssl_verification()

JSON_DIR = os.getenv("JSON_DIR", "./ground_truth")
IMAGE_DIR = os.getenv("IMAGE_DIR", "./document_images")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MODEL_ID = os.getenv("MODEL_ID", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")

# --- BALANCED DEFAULTS FOR SMALL INVOICE DATASETS ---
SEED = int(os.getenv("SEED", "42"))
EPOCHS = int(os.getenv("EPOCHS", "12"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
LR = float(os.getenv("LEARNING_RATE", "1e-4"))
VAL_SPLIT = max(0.0, min(float(os.getenv("VAL_SPLIT", "0.15")), 0.4))
REPEAT_FACTOR = max(1, int(os.getenv("REPEAT_FACTOR", "1")))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("--- Starting Training Pipeline (Unsloth) ---")
print(f"Model: {MODEL_ID}")
print(f"Batch Size: {BATCH_SIZE}, Accumulation: {GRAD_ACCUM}, Effective Batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Epochs: {EPOCHS}, LR: {LR}, Val Split: {VAL_SPLIT}, Max Seq Length: {MAX_SEQ_LENGTH}\n")
 
# 2. Model & LoRA Initialization
print("Loading Model & Processor with Unsloth...")
model, processor = FastVisionModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
 
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=SEED,
    use_rslora=False,
)
model.print_trainable_parameters()
 
# 3. Data Preparation
print("\nPreparing dataset...")
 
# --- TRAIN ON THE SAME COMPACT 4-PASS FORMAT USED AT INFERENCE ---
IDENTITY_METADATA_FIELDS = ("document_type", "currency", "language", "original_filename", "page_count")
SHARED_METADATA_FIELDS = ("currency", "document_type")

IDENTITY_FIELDS = (
    "document_id", "date_issue", "date_due", "purchase_order_id", "terms",
    "sender_name", "sender_address", "recipient_name", "recipient_address",
    "recipient_delivery_name", "recipient_delivery_address", "vendor_phone",
)
AMOUNT_FIELDS = ("amount_due", "amount_total_tax", "amount_total_base")
TAX_FIELDS = ("tax_name", "tax_amount", "sender_vat_id")
LINE_ITEM_FIELDS = (
    "item_code", "item_description", "item_uom", "item_quantity",
    "item_amount", "item_amount_total",
)

COMPACT_RULES = """Return exactly one compact JSON object and nothing else. Use only the allowed visible keys. Recover every allowed field or row cell that is visibly present in the invoice with best effort. Do not leave a visibly present allowed field blank and do not omit it unless it is truly absent or unreadable. No markdown, no explanation, no null placeholders, and no duplicated rows. bbox must be [x1, y1, x2, y2] or null, and page is 1-based. Keep metadata-only values such as document_type, currency, and language under metadata."""

IDENTITY_PROMPT = f"""{COMPACT_RULES}

Task: extract only header / identity / contact fields in compact DocILE-style JSON.
Allowed fields: document_id, date_issue, date_due, purchase_order_id, terms, sender_name, sender_address, recipient_name, recipient_address, recipient_delivery_name, recipient_delivery_address, vendor_phone.

Return this shape:
{{
  "fields": {{
    "<visible_field_name>": {{"text": "...", "bbox": [x1, y1, x2, y2], "page": 1}}
  }},
  "metadata": {{"document_type": "...", "currency": "...", "language": "...", "original_filename": "...", "page_count": 1}}
}}
Only include keys that are clearly visible. Omit missing fields.
"""

AMOUNTS_PROMPT = f"""{COMPACT_RULES}

Task: extract only the grand totals / balance-summary region.
Allowed fields: amount_due, amount_total_tax, amount_total_base.
Do not include tax rows or line items.

Return this shape:
{{
  "fields": {{
    "<visible_total_field>": {{"text": "...", "bbox": [x1, y1, x2, y2], "page": 1}}
  }},
  "metadata": {{"currency": "...", "document_type": "..."}}
}}
"""

TAX_DETAILS_PROMPT = f"""{COMPACT_RULES}

Task: extract only the tax / VAT summary block.
Allowed fields: tax_name, tax_amount, sender_vat_id.
Do not include grand totals or line-item fields.

Return this shape:
{{
  "tax_rows": [
    {{
      "<visible_tax_field>": {{"text": "...", "bbox": [x1, y1, x2, y2], "page": 1}}
    }}
  ],
  "metadata": {{"currency": "...", "document_type": "..."}}
}}
Each tax row may contain `tax_name`, `tax_amount`, and `sender_vat_id` when visible. Omit missing keys and do not repeat the same row twice.
"""

LINE_ITEMS_PROMPT = f"""{COMPACT_RULES}

Task: extract only invoice line items.
Allowed fields: item_code, item_description, item_uom, item_quantity, item_amount, item_amount_total.

Return this shape:
{{
  "line_items": [
    {{
      "line_item_id": 0,
      "page": 1,
      "fields": {{"<visible_line_item_field>": {{"text": "...", "bbox": [x1, y1, x2, y2]}}}}
    }}
  ],
  "line_item_headers": []
}}
Use one object per visible row. Include only visible fields and stop once all visible rows are covered.
"""

def build_conversation(img_path: str, prompt_text: str, target_json: str) -> dict:
    """Build a single training example in the chat format Unsloth expects."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_json}],
            },
        ]
    }


def _is_present(value):
    return value not in (None, "", [], {})


def _safe_page(page_value, default=1):
    try:
        if page_value is None or page_value == "":
            return default
        return int(page_value)
    except Exception:
        return default


def _safe_line_item_id(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _compact_entry_from_docile(entry: dict) -> dict:
    bbox = entry.get("bbox") if isinstance(entry, dict) else None
    return {
        "text": entry.get("text") if isinstance(entry, dict) else None,
        "bbox": bbox if isinstance(bbox, list) and len(bbox) == 4 else None,
        "page": _safe_page(entry.get("page", 1) if isinstance(entry, dict) else 1, 1),
    }


def _metadata_subset(metadata: dict, keys) -> dict:
    subset = {}
    if not isinstance(metadata, dict):
        return subset
    for key in keys:
        value = metadata.get(key)
        if _is_present(value):
            subset[key] = value
    return subset


def _collect_doc_fields(payload: dict, allowed_fields: set) -> dict:
    grouped = defaultdict(list)
    for item in payload.get("field_extractions", []):
        if not isinstance(item, dict):
            continue
        fieldtype = item.get("fieldtype")
        if fieldtype in allowed_fields and _is_present(item.get("text")):
            grouped[fieldtype].append(item)
    return grouped


def _metadata_with_fallback(payload: dict, keys) -> dict:
    subset = _metadata_subset(payload.get("metadata", {}), keys)
    grouped = _collect_doc_fields(payload, set(keys))
    for key in keys:
        if key not in subset and grouped.get(key):
            value = grouped[key][0].get("text")
            if _is_present(value):
                subset[key] = value
    return subset


def canonicalize_docile_target(payload: dict) -> dict:
    """Keep DocILE targets in a stable top-level order for more consistent supervision."""
    if not isinstance(payload, dict):
        return payload
    return {
        "field_extractions": payload.get("field_extractions", []),
        "line_item_extractions": payload.get("line_item_extractions", []),
        "line_item_headers": payload.get("line_item_headers", []),
        "metadata": payload.get("metadata", {}),
    }


def build_identity_target(payload: dict) -> dict:
    grouped = _collect_doc_fields(payload, set(IDENTITY_FIELDS))
    fields = {
        field: _compact_entry_from_docile(grouped[field][0])
        for field in IDENTITY_FIELDS
        if grouped.get(field)
    }
    return {
        "fields": fields,
        "metadata": _metadata_with_fallback(payload, IDENTITY_METADATA_FIELDS),
    }


def build_amounts_target(payload: dict) -> dict:
    grouped = _collect_doc_fields(payload, set(AMOUNT_FIELDS))
    fields = {
        field: _compact_entry_from_docile(grouped[field][0])
        for field in AMOUNT_FIELDS
        if grouped.get(field)
    }
    return {
        "fields": fields,
        "metadata": _metadata_with_fallback(payload, SHARED_METADATA_FIELDS),
    }


def build_tax_target(payload: dict) -> dict:
    grouped = _collect_doc_fields(payload, set(TAX_FIELDS))
    max_rows = max([len(grouped.get(field, [])) for field in TAX_FIELDS] or [0])
    tax_rows = []

    for row_idx in range(max_rows):
        row = {}
        for field in TAX_FIELDS:
            values = grouped.get(field, [])
            if row_idx < len(values):
                row[field] = _compact_entry_from_docile(values[row_idx])
        if row:
            tax_rows.append(row)

    return {
        "tax_rows": tax_rows,
        "metadata": _metadata_with_fallback(payload, SHARED_METADATA_FIELDS),
    }


def build_line_items_target(payload: dict) -> dict:
    grouped_rows = {}

    for item in payload.get("line_item_extractions", []):
        if not isinstance(item, dict):
            continue
        fieldtype = item.get("fieldtype")
        if fieldtype not in LINE_ITEM_FIELDS or not _is_present(item.get("text")):
            continue

        line_item_id = _safe_line_item_id(item.get("line_item_id"), 0)
        row = grouped_rows.setdefault(
            line_item_id,
            {
                "line_item_id": line_item_id,
                "page": _safe_page(item.get("page", 1), 1),
                "fields": {},
            },
        )
        row["page"] = _safe_page(item.get("page", row.get("page", 1)), row.get("page", 1))
        row["fields"].setdefault(fieldtype, _compact_entry_from_docile(item))

    line_items = [grouped_rows[key] for key in sorted(grouped_rows.keys())]
    headers = payload.get("line_item_headers", []) if isinstance(payload.get("line_item_headers", []), list) else []
    return {
        "line_items": line_items,
        "line_item_headers": headers,
    }


def build_training_examples(img_path: str, payload: dict):
    canonical_payload = canonicalize_docile_target(payload)

    pass_examples = [
        (IDENTITY_PROMPT, build_identity_target(canonical_payload)),
        (AMOUNTS_PROMPT, build_amounts_target(canonical_payload)),
        (TAX_DETAILS_PROMPT, build_tax_target(canonical_payload)),
        (LINE_ITEMS_PROMPT, build_line_items_target(canonical_payload)),
    ]

    return [
        build_conversation(
            img_path,
            prompt_text,
            json.dumps(target_payload, ensure_ascii=False, separators=(",", ":")),
        )
        for prompt_text, target_payload in pass_examples
    ]


data = []
document_pairs = 0
if os.path.exists(JSON_DIR) and os.path.exists(IMAGE_DIR):
    for filename in sorted(os.listdir(JSON_DIR)):
        if filename.endswith(".json"):
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(JSON_DIR, filename)
            img_path = os.path.join(IMAGE_DIR, base_name + ".png")

            if os.path.exists(img_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    data.extend(build_training_examples(img_path, payload))
                    document_pairs += 1
                except Exception as e:
                    print(f"Error processing {base_name}: {e}")

    if not data:
        raise ValueError(f"No matching training pairs found in {JSON_DIR} and {IMAGE_DIR}.")

    full_dataset = Dataset.from_pandas(pd.DataFrame(data), preserve_index=False)

    if len(full_dataset) >= 8 and VAL_SPLIT > 0:
        split = full_dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = full_dataset
        eval_dataset = full_dataset.select(range(min(max(1, len(full_dataset) // 5), len(full_dataset))))

    if REPEAT_FACTOR > 1 and len(train_dataset) > 0:
        train_dataset = Dataset.from_pandas(
            pd.concat([train_dataset.to_pandas()] * REPEAT_FACTOR, ignore_index=True),
            preserve_index=False,
        )
        print(f"Repeated only the training split x{REPEAT_FACTOR} (validation kept untouched).")

    print(
        f"Prepared {document_pairs} matched documents -> {len(full_dataset)} supervised conversations | "
        f"format=compact_multi_pass | train={len(train_dataset)} | eval={len(eval_dataset)} | "
        f"repeat_factor={REPEAT_FACTOR}\n"
    )
else:
    raise FileNotFoundError("JSON_DIR or IMAGE_DIR not found. Check your .env paths.")
 
 
# 4. Formatting function
def formatting_func(examples):
    texts = []
    for msg in examples["messages"]:
        text = processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}
 
train_dataset = train_dataset.map(formatting_func, batched=True)
eval_dataset = eval_dataset.map(formatting_func, batched=True)


# ─────────────────────────────────────────────────────────────
# 5a.  Per-epoch metrics callback
# ─────────────────────────────────────────────────────────────
class TrainingMetricsCallback(TrainerCallback):
    """
    Tracks per-epoch training loss plus generation-based research-style
    extraction metrics (exact match, precision, recall, F1) on a small
    validation subset.
    """

    def __init__(self, reference_dataset, collator, num_eval_samples=6):
        n = min(max(1, num_eval_samples), len(reference_dataset)) if len(reference_dataset) else 0
        self._eval_samples = [reference_dataset[i] for i in range(n)]
        self._collator = collator
        self._step_losses = []
        self._epoch_step_losses = []
        self.epoch_metrics = defaultdict(list)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = float(logs["loss"])
            self._step_losses.append((state.global_step, loss))
            self._epoch_step_losses.append(loss)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        avg_loss = sum(self._epoch_step_losses) / len(self._epoch_step_losses) if self._epoch_step_losses else float("nan")
        self._epoch_step_losses = []

        eval_metrics = self._compute_research_metrics(model)

        self.epoch_metrics["epoch"].append(epoch)
        self.epoch_metrics["loss"].append(avg_loss)
        self.epoch_metrics["exact_match"].append(eval_metrics["exact_match"])
        self.epoch_metrics["field_precision"].append(eval_metrics["precision"])
        self.epoch_metrics["field_recall"].append(eval_metrics["recall"])
        self.epoch_metrics["field_f1"].append(eval_metrics["f1"])

        print(
            f"\n📊 [Epoch {epoch:>3}/{EPOCHS}]  "
            f"Loss: {avg_loss:.4f}  │  "
            f"EM: {eval_metrics['exact_match'] * 100:.1f}%  │  "
            f"P/R/F1: {eval_metrics['precision'] * 100:.1f}%/"
            f"{eval_metrics['recall'] * 100:.1f}%/"
            f"{eval_metrics['f1'] * 100:.1f}%"
        )

        if len(self.epoch_metrics["loss"]) >= 2:
            previous_best = min(self.epoch_metrics["loss"][:-1])
            if avg_loss > previous_best * 1.05:
                print("⚠ Loss is rising above the best prior epoch; consider fewer epochs or a lower LR if this continues.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _empty_research_metrics(self):
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    def _normalize_eval_text(self, text):
        if text is None:
            return ""
        cleaned = str(text).strip().lower().replace("$", "").replace(",", "")
        return " ".join(cleaned.split())

    def _extract_json_candidate(self, text):
        text = str(text).strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text

    def _safe_load_json(self, text):
        try:
            return json.loads(self._extract_json_candidate(text))
        except Exception:
            return None

    def _maybe_add_field(self, items, name, value):
        if isinstance(value, dict):
            text = value.get("text")
        else:
            text = value
        normalized = self._normalize_eval_text(text)
        if normalized:
            items.add((name, normalized))

    def _flatten_compact_payload(self, payload):
        items = set()
        if not isinstance(payload, dict):
            return items

        fields = payload.get("fields", {})
        if isinstance(fields, dict):
            for key, value in fields.items():
                self._maybe_add_field(items, key, value)

        tax_rows = payload.get("tax_rows", [])
        if isinstance(tax_rows, list):
            for row_idx, row in enumerate(tax_rows):
                if isinstance(row, dict):
                    for key, value in row.items():
                        self._maybe_add_field(items, f"tax_row[{row_idx}].{key}", value)

        line_items = payload.get("line_items", [])
        if isinstance(line_items, list):
            for row_idx, row in enumerate(line_items):
                if not isinstance(row, dict):
                    continue
                line_item_id = row.get("line_item_id", row_idx)
                row_fields = row.get("fields", {})
                if isinstance(row_fields, dict):
                    for key, value in row_fields.items():
                        self._maybe_add_field(items, f"line_item[{line_item_id}].{key}", value)

        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                self._maybe_add_field(items, f"metadata.{key}", value)

        return items

    def _prepare_generation_inputs(self, sample, model):
        messages = sample.get("messages") if isinstance(sample, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError("Sample is missing chat messages for evaluation.")

        prompt_messages = [messages[0]]
        text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([prompt_messages])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            k: v.to(model.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    def _extract_target_payload(self, sample):
        messages = sample.get("messages") if isinstance(sample, dict) else None
        if not isinstance(messages, list) or len(messages) < 2:
            return None
        assistant_content = messages[1].get("content", [])
        if not isinstance(assistant_content, list) or not assistant_content:
            return None
        target_text = assistant_content[0].get("text")
        return self._safe_load_json(target_text)

    def _compute_research_metrics(self, model):
        if model is None or not self._eval_samples or process_vision_info is None:
            return self._empty_research_metrics()

        model.eval()
        tp = fp = fn = exact_matches = sample_count = 0
        stop_tokens = [processor.tokenizer.eos_token_id]
        vocab = processor.tokenizer.get_vocab()
        if "<|im_end|>" in vocab:
            stop_tokens.append(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))

        for sample in self._eval_samples:
            try:
                target_payload = self._extract_target_payload(sample)
                if target_payload is None:
                    continue

                inputs = self._prepare_generation_inputs(sample, model)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=768,
                        do_sample=False,
                        eos_token_id=stop_tokens,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        use_cache=True,
                        repetition_penalty=1.05,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                pred_payload = self._safe_load_json(output_text) or {}

                target_items = self._flatten_compact_payload(target_payload)
                pred_items = self._flatten_compact_payload(pred_payload)

                tp += len(target_items & pred_items)
                fp += len(pred_items - target_items)
                fn += len(target_items - pred_items)
                exact_matches += int(pred_items == target_items)
                sample_count += 1
            except Exception as e:
                print(f"  [Research Metrics] skipped 1 sample: {e}")

        model.train()

        if sample_count == 0:
            return self._empty_research_metrics()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "exact_match": exact_matches / sample_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
 
 
# ─────────────────────────────────────────────────────────────
# 5b.  Visualisation helper (called after training)
# ─────────────────────────────────────────────────────────────
def plot_training_metrics(metrics: dict, step_losses: list, output_dir: str):
    """
    Saves a 6-panel PNG to output_dir/training_metrics.png using more
    research-style extraction metrics:
      Panel 1 – step-level loss curve
      Panel 2 – per-epoch average loss
      Panel 3 – exact match per epoch
      Panel 4 – field precision per epoch
      Panel 5 – field recall per epoch
      Panel 6 – field F1 per epoch
    Also writes step_losses.csv and epoch_metrics.csv for offline analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = metrics["epoch"]
 
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle(
        "Compact Multi-Pass Vision Fine-Tuning — Training Metrics",
        fontsize=14, fontweight="bold", y=1.02
    )
 
    # ── Panel 0: step-level loss ──
    if step_losses:
        steps, s_losses = zip(*step_losses)
        axes[0].plot(steps, s_losses, color="#E67E22", linewidth=1.2, alpha=0.85)
        axes[0].set_title("Loss per Step", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Global Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.35)
    else:
        axes[0].set_title("Loss per Step", fontsize=11, fontweight="bold")
        axes[0].text(0.5, 0.5, "No step losses logged", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].grid(True, linestyle="--", alpha=0.35)
 
    # ── Panels 1-5: per-epoch research metrics ──
    panel_cfg = [
        ("loss",            "Loss per Epoch",            "#E74C3C", "Loss",        False),
        ("exact_match",     "Exact Match per Epoch",     "#3498DB", "Exact Match", True),
        ("field_precision", "Field Precision per Epoch", "#2ECC71", "Precision",   True),
        ("field_recall",    "Field Recall per Epoch",    "#F39C12", "Recall",      True),
        ("field_f1",        "Field F1 per Epoch",        "#8E44AD", "F1 Score",    True),
    ]
 
    for ax, (key, title, color, ylabel, is_pct) in zip(axes[1:], panel_cfg):
        values = metrics.get(key, [])
        if not values:
            continue
        ax.plot(epochs, values, color=color, linewidth=2,
                marker="o", markersize=5, zorder=3)
        ax.fill_between(epochs, values, alpha=0.12, color=color)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(left=min(epochs) - 0.5)
 
        best_idx = (values.index(max(values)) if is_pct
                    else values.index(min(values)))
        ax.annotate(
            f"{values[best_idx]*100:.1f}%" if is_pct else f"{values[best_idx]:.3f}",
            xy=(epochs[best_idx], values[best_idx]),
            xytext=(10, 8), textcoords="offset points",
            fontsize=8, color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
        )
 
        if is_pct:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
            )
 
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 Metrics plot saved  → {plot_path}")
 
    # Save step-level losses as CSV
    csv_path = os.path.join(output_dir, "step_losses.csv")
    with open(csv_path, "w") as f:
        f.write("global_step,loss\n")
        for step, loss in step_losses:
            f.write(f"{step},{loss:.6f}\n")
    print(f"📄 Step losses CSV     → {csv_path}")

    epoch_csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    pd.DataFrame(metrics).to_csv(epoch_csv_path, index=False)
    print(f"📄 Epoch metrics CSV   → {epoch_csv_path}")
 
 
# ─────────────────────────────────────────────────────────────
# 5. Start Fine-Tuning
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    logging_first_step=True,
    save_steps=100,
    save_total_limit=2,
    optim="adamw_8bit",
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    remove_unused_columns=False,
    dataset_text_field="text",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=SEED,
)

# Build collator once so we can share it with the callback
data_collator = UnslothVisionDataCollator(model, processor)

# Instantiate metrics callback using the validation split when available
metrics_callback = TrainingMetricsCallback(
    reference_dataset=eval_dataset,
    collator=data_collator,
    num_eval_samples=min(8, len(eval_dataset)),
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=processor.tokenizer,
    callbacks=[metrics_callback],
)
 
# Switch model to training mode
FastVisionModel.for_training(model)
 
print(f"\nStarting compact multi-pass fine-tuning for {MODEL_ID} with Unsloth...")
trainer.train()
 
# 6. Save LoRA adapters
final_save_path = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(final_save_path)
processor.save_pretrained(final_save_path)
print(f"\n✅ Training complete! LoRA adapters saved to {final_save_path}")
 
# 7. Visualise training metrics
plot_training_metrics(
    metrics    = dict(metrics_callback.epoch_metrics),
    step_losses= metrics_callback._step_losses,
    output_dir = OUTPUT_DIR,
)