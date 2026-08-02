import cv2
from paddleocr import PaddleOCR
import json
from pathlib import Path

def debug_scale():
    img_path = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data\images_source\invoice_CPC000692108.jpg"
    img = cv2.imread(img_path)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    results = ocr.ocr(img, cls=True)
    
    ocr_x = None
    if results and results[0]:
        for line in results[0]:
            text = line[1][0]
            if "CPC000692108" in text:
                pts = line[0]
                ocr_x = min(p[0] for p in pts)
                ocr_y = min(p[1] for p in pts)
                print(f"OCR found 'CPC000692108' at: x={ocr_x}, y={ocr_y}")
                break
                
    ann_path = r"d:\Internship\Invoice Intelligence Document\LayoutLMv3\Preparing data-finetuned\annotations\invoice_CPC000692108.json"
    with open(ann_path, 'r') as f:
        data = json.load(f)
        for field in data.get("field_extractions", []):
            if field.get("fieldtype") == "document_id":
                json_x = field.get("bbox")[0]
                json_y = field.get("bbox")[1]
                print(f"JSON 'document_id' at: x={json_x}, y={json_y}")
                break
                
    if ocr_x is not None:
        print(f"Calculated X scale: {ocr_x / json_x:.4f}")
        print(f"Calculated Y scale: {ocr_y / json_y:.4f}")

if __name__ == "__main__":
    debug_scale()
