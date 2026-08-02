import cv2
from paddleocr import PaddleOCR
import json
from pathlib import Path

def debug_scale():
    img_path = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data\images_source\invoice_CPC000692108.jpg"
    img = cv2.imread(img_path)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    results = ocr.ocr(img, cls=True)
    
    ocr_points = {}
    if results and results[0]:
        for line in results[0]:
            text = line[1][0]
            pts = line[0]
            x, y = min(p[0] for p in pts), min(p[1] for p in pts)
            if "CPC000692108" in text: ocr_points["document_id"] = (x, y)
            if "05/23/2025" in text: ocr_points["date_issue"] = (x, y)
            if "53-179213" in text: ocr_points["purchase_order_id"] = (x, y)
                
    ann_path = r"d:\Internship\Invoice Intelligence Document\LayoutLMv3\Preparing data-finetuned\annotations\invoice_CPC000692108.json"
    with open(ann_path, 'r') as f:
        data = json.load(f)
        for field in data.get("field_extractions", []) + data.get("line_item_extractions", []):
            ft = field.get("fieldtype")
            if ft in ocr_points:
                jx, jy = field.get("bbox")[0], field.get("bbox")[1]
                ox, oy = ocr_points[ft]
                print(f"Field: {ft}")
                print(f"  OCR: ({ox}, {oy})")
                print(f"  JSON: ({jx}, {jy})")
                print(f"  X Ratio: {ox/jx:.4f}, Y Ratio: {oy/jy:.4f}")

if __name__ == "__main__":
    debug_scale()
