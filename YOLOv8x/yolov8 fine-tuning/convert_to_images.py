import fitz
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

def convert_pdfs_to_images(dpi=144):
    base_dir = Path(r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data")
    pdf_dir = base_dir / "pdf"
    img_out_dir = base_dir / "images_source"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    
    # We want width exactly 1280.
    # Page width in points is 612 (for US Letter).
    # DPI = 1280 / (612 / 72) = 150.588
    target_dpi = 150.5882
    zoom = target_dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # Target dimensions (multiples of 32)
    target_w, target_h = 1280, 1664
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Converting {len(pdf_files)} PDFs to images at YOLO-optimized size (1280x1664)...")
    
    for pdf_path in tqdm(pdf_files):
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Force resize to exactly 1280x1664 to ensure it's a multiple of 32
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
            img_name = pdf_path.stem + ".jpg"
            cv2.imwrite(str(img_out_dir / img_name), img)
            doc.close()
        except Exception as e:
            print(f"Error converting {pdf_path.name}: {e}")

if __name__ == "__main__":
    convert_pdfs_to_images()
