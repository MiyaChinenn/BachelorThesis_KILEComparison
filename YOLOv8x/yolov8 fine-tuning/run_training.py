import os
import sys
from pathlib import Path
import torch

# 1. SETUP PATHS (Standalone: All paths are now internal to Fine-tuning)
SCRIPT_DIR = Path(__file__).parent.absolute()

# Primary model: train10/weights/best.pt (Now inside Fine-tuning)
MODEL_PATH = SCRIPT_DIR / "train10" / "weights" / "best.pt"
DATA_YAML = SCRIPT_DIR / "single_template.yaml"

# Code: ultralytics folder (Should be moved into Fine-tuning)
YOLOV8_LIB = SCRIPT_DIR

# Add current folder to path to find the local 'ultralytics' package
sys.path.insert(0, str(YOLOV8_LIB))

# 2. WINDOWS DLL FIX
if sys.platform == "win32":
    # Try to find the venv from the current python executable
    venv_base = os.path.dirname(os.path.dirname(sys.executable))
    possible_paths = [
        os.path.join(venv_base, "Lib", "site-packages", "torch", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "paddle", "libs"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except AttributeError:
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

# Set environment variables
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    from ultralytics import YOLO
    print("Ultralytics/YOLO loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR loading 'ultralytics': {e}")
    import traceback
    traceback.print_exc()
    print("Please check the error above to see which dependency is missing.")
    sys.exit(1)

def start_training():
    print(f"--- Starting Single Template Fine-tuning ---")
    
    # 2.5 DYNAMICALLY UPDATE YAML PATH for portability
    # This ensures it works on any drive/machine
    try:
        import yaml
        with open(DATA_YAML, 'r') as f:
            y_data = yaml.safe_load(f)
        
        # Update path to current processed_data location
        y_data['path'] = str(SCRIPT_DIR / "processed_data").replace("\\", "/")
        
        with open(DATA_YAML, 'w') as f:
            yaml.safe_dump(y_data, f)
        print(f"Updated {DATA_YAML.name} path to: {y_data['path']}")
    except Exception as e:
        print(f"Non-critical Warning: Could not auto-update YAML path: {e}")

    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_YAML}")
    
    # 3. LOAD MODEL & HANDLE 6-CHANNEL WARM START
    # We load the weights manually to ensure 6-channel compatibility
    model = YOLO(str(MODEL_PATH))
    conv = model.model.model[0].conv
    
    REQUIRED_CH = 6
    if conv.weight.shape[1] == REQUIRED_CH:
        print(f"Model already has {REQUIRED_CH} channels. Continuing.")
    else:
        print(f"Migrating from {conv.weight.shape[1]} to {REQUIRED_CH} channels...")
        # Create a temporary model with 6 channels
        model_tmp = YOLO("yolov8x.yaml", ch=REQUIRED_CH)
        conv_tmp = model_tmp.model.model[0].conv
        
        with torch.no_grad():
            # Copy first 3 channels
            conv_tmp.weight[:,:3,:,:].copy_(conv.weight[:,:3,:,:])
            # Copy to extra 3 channels
            conv_tmp.weight[:,3:,:,:].copy_(conv.weight[:,:3,:,:])
            
        model.model.model[0].conv = conv_tmp
        model.model.yaml["ch"] = REQUIRED_CH
        print("Warm-start migration complete.")

    # 4. START TRAINING
    # Optimized for 40GB VRAM machine
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        batch=8,
        imgsz=(1664, 1280),
        rect=True,
        workers=4,
        optimizer='AdamW',
        lr0=0.001,
        project=str(SCRIPT_DIR / "runs"),
        name="single_template_finetune",
        device=0 if torch.cuda.is_available() else 'cpu',
        docile_data_loader=False  # Crucial: Use standard YOLO loader for our .npy files
    )

if __name__ == "__main__":
    start_training()
