import fitz
import json

pdf_path = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data\pdf\invoice_CPC000692108.pdf"
doc = fitz.open(pdf_path)
page = doc[0]

# Print page dimensions
print(f"Page rect: {page.rect}")
print(f"Page media_box: {page.mediabox}")

zoom = 144 / 72
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)
print(f"Rendered pixmap size: {pix.width}x{pix.height}")

ann_path = r"d:\Internship\Invoice Intelligence Document\LayoutLMv3\Preparing data-finetuned\annotations\invoice_CPC000692108.json"
with open(ann_path, 'r') as f:
    ann_data = json.load(f)
    
for field in ann_data.get("field_extractions", []):
    if field.get("fieldtype") == "document_id":
        print(f"document_id JSON bbox: {field.get('bbox')}")
        break
