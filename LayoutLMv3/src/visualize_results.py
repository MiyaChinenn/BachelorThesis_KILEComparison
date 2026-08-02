import json
import os
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
JSON_PATH = os.path.join(PROJECT_ROOT, "inference_results.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "visualized_inference.png")

def visualize():
    # 1. Load JSON
    if not os.path.exists(JSON_PATH):
        print(f"[Error] JSON not found at {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # 2. Load Image
    image_name = data.get("image")
    image_path = os.path.join(PROJECT_ROOT, "images", image_name)
    
    if not os.path.exists(image_path):
        print(f"[Error] Image not found at {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 3. Try to load a font
    try:
        # Common Windows font
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # 4. Draw Boxes
    print(f"Visualizing {len(data['predictions'])} predictions...")
    
    for pred in data["predictions"]:
        label = pred["label"]
        conf = pred["confidence"]
        text = pred["text"]
        box = pred["box_2d"] # [x1, y1, x2, y2] in 0-1000

        # Scale 0-1000 back to pixels
        x1 = box[0] * width / 1000
        y1 = box[1] * height / 1000
        x2 = box[2] * width / 1000
        y2 = box[3] * height / 1000

        # Draw Rectangle (red)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Prepare Label Text
        display_text = f"{label} ({conf:.2f})"
        
        # Draw label background
        tw, th = draw.textbbox((0, 0), display_text, font=font)[2:]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
        draw.text((x1, y1 - th), display_text, fill="white", font=font)

    # 5. Save
    img.save(OUTPUT_PATH)
    print(f"[OK] Visualized image saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    visualize()
