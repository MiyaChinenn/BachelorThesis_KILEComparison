import os
import sys
from pathlib import Path
import json
import numpy as np
import cv2

# DLL Fix for Windows (Torch/Paddle)
VENV_BASE = r"C:\Users\DELL\.virtualenvs\LayoutLMv3-gMFc_kGI"
if sys.platform == "win32":
    possible_paths = [
        os.path.join(VENV_BASE, "Lib", "site-packages", "torch", "lib"),
        os.path.join(VENV_BASE, "Lib", "site-packages", "paddle", "libs"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except AttributeError:
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

try:
    import torch
    from paddleocr import PaddleOCR
    import fitz  # PyMuPDF
    print(f"Environment Check: Torch {torch.__version__}, PaddleOCR and PyMuPDF loaded.")
except Exception as e:
    print(f"Environment Error: {e}")
    sys.exit(1)

# --- EMBEDDED CHARGRID.PY LOGIC (To avoid package conflicts) ---
characters = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
              '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b',
              'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', '|', '£', '°', 'À', 'Ç', 'È', 'É', 'Ô', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ô', 'û', '€']

def create_encodings_three_digit_0(chars):
    chars = ["", *chars]
    number_digits = 3
    base = np.ceil(np.cbrt(len(chars))).astype(int)
    encodings = {c: np.base_repr(i, base=base).zfill(number_digits) for i, c in enumerate(chars)}
    encodings = {c: np.array([int(v) for v in e]) for c, e in encodings.items()}
    encodings = {c: e / (base - 1) for c, e in encodings.items()}
    return encodings

def get_char_grid_easy(ocr_data, shape, encodings):
    if list(encodings.keys())[0] == "":
        char_grid = np.zeros([*shape, 3])
    else:
        char_grid = np.ones([*shape, 3])
    for o in ocr_data:
        text = o[1]
        if not text.strip(): continue
        x0, y0, x1, y1 = o[0]
        num_chars = len(text)
        w = x1 - x0
        step = w / num_chars
        x_start_idx = np.floor(np.arange(num_chars) * step + x0).astype(int)
        x_end_idx = np.ceil((np.arange(num_chars) + 1) * step + x0).astype(int)
        for i, c in enumerate(text):
            enc = encodings.get(c, encodings[""])
            # Boundary checks for image size
            x_start = max(0, x_start_idx[i])
            x_end = min(shape[1], x_end_idx[i])
            y_start = max(0, int(y0))
            y_end = min(shape[0], int(y1))
            char_grid[y_start:y_end, x_start:x_end] = enc
    return char_grid
# --- END EMBEDDED LOGIC ---

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Label Mapping: Concentrix (24) -> DocILE (52)
LABEL_MAP = {
    "date_due": 4, "date_issue": 3, "purchase_order_id": 1, "document_id": 0, "terms": 16,
    "tax_amount": 22, "sender_vat_id": 34, "tax_name": 27, "amount_due": 23,
    "amount_total_tax": 22, "amount_total_base": 21, "recipient_delivery_address": 36,
    "recipient_delivery_name": 35, "recipient_address": 36, "recipient_name": 35,
    "vendor_phone": 31, "sender_address": 30, "sender_name": 29, "item_uom": 46,
    "item_amount_total": 45, "item_amount": 44, "item_description": 43, "item_code": 42, "item_quantity": 41
}

def render_page_fitz(pdf_path, page_num=0, dpi=144):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def prepare_data():
    base_dir = Path(r"d:\Internship\Invoice Intelligence Document\YOLOv8x")
    
    # Images are stored here now
    img_src_dir = base_dir / "Fine-tuning" / "processed_data" / "images_source"
    
    # Annotations are stored in the processed_data directory now
    ann_dir = base_dir / "Fine-tuning" / "processed_data" / "annotations"
    
    output_dir = base_dir / "Fine-tuning" / "processed_data"
    output_img_dir = output_dir / "images"
    output_lbl_dir = output_dir / "labels"
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    encodings = create_encodings_three_digit_0(characters)
    # The source images were rendered at 144 DPI
    dpi = 144
    
    img_files = list(img_src_dir.glob("*.jpg"))
    print(f"Starting processing of {len(img_files)} images...")
    
    for img_path in img_files:
        try:
            doc_id = img_path.stem
            ann_path = ann_dir / f"{doc_id}.json"
            if not ann_path.exists(): continue
                
            print(f"Processing: {doc_id}")
            
            # Load image directly
            img_np = cv2.imread(str(img_path))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            h, w = img_np.shape[:2]
            
            ocr_results = ocr.ocr(img_np, cls=True)
            ocr_data = []
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    pts, (txt, score) = line[0], line[1]
                    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                    x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
                    ocr_data.append([[x0, y0, x1, y1], txt])
            
            char_grid = get_char_grid_easy(ocr_data, (h, w), encodings)
            char_grid = (char_grid * 255).astype(np.uint8)
            stacked = np.concatenate((char_grid, img_np), axis=2)
            np.save(str(output_img_dir / f"{doc_id}.npy"), stacked)
            
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            
            yolo_labels = []
            fields = ann_data.get("field_extractions", []) + ann_data.get("line_item_extractions", [])
            for field in fields:
                ft = field.get("fieldtype")
                bbox = field.get("bbox")
                if not bbox or ft not in LABEL_MAP: continue
                
                cls_id = LABEL_MAP[ft]
                x0, y0, x1, y1 = bbox
                
                # The annotations are at 150 DPI (Width 1275, Height 1650).
                # Our new image is 1280x1664.
                scale_x = 1280 / 1275.0
                scale_y = 1664 / 1650.0
                x0, y0, x1, y1 = x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y
                
                bw, bh = (x1 - x0), (y1 - y0)
                cx, cy = (x0 + (x1-x0)/2), (y0 + (y1-y0)/2)
                yolo_labels.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
                
            with open(output_lbl_dir / f"{doc_id}.txt", 'w') as f:
                f.write("\n".join(yolo_labels))
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    print("Done! All processed files are in Fine-tuning/processed_data/")

if __name__ == "__main__":
    prepare_data()
