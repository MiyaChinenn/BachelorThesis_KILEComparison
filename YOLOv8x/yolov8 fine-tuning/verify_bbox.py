import os
import cv2
import json
import numpy as np
import fitz
from pathlib import Path

def test_bbox_scaling():
    # Load PDF path
    pdf_path = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data\pdf\invoice_CPC000692108.pdf"
    
    # Target dimensions (multiples of 32)
    target_w, target_h = 1280, 1664
    
    # Render PDF to image and resize
    target_dpi = 150.5882
    zoom = target_dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Force resize to exactly 1280x1664
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Load JSON annotations
    ann_path = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data\annotations\invoice_CPC000692108.json"
    with open(ann_path, 'r') as f:
        ann_data = json.load(f)
        
    fields = ann_data.get("field_extractions", []) + ann_data.get("line_item_extractions", [])
    
    # Draw boxes
    for field in fields:
        bbox = field.get("bbox")
        ft = field.get("fieldtype")
        if not bbox: continue
        
        # Apply scaling for 1280x1664 (from 150 DPI)
        scale_x = 1280 / 1275.0
        scale_y = 1664 / 1650.0
        x0, y0, x1, y1 = bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y
        
        # Draw the rectangle
        cv2.rectangle(img_bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.putText(img_bgr, ft, (int(x0), int(y0)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save output
    output_path = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\bbox_verification.jpg"
    cv2.imwrite(output_path, img_bgr)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    test_bbox_scaling()
