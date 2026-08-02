# Loader

from utils import *
import os
import torch
from tqdm import tqdm
from PIL import Image


class dataSet:
    def __init__(self,json_path,processor=None) -> None:
        self.json_data = train_data_format(read_json(json_path))
        self.processor = processor
        
        # Load label mapping
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "inputs", "label_config.json")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.label2id = config['label2id']


    def __len__(self)->int:
        # print(self.json_data)
        return len(self.json_data)

    def __getitem__(self,index)->dict:
        imgs = []
        words = []
        label = []
        bboxes = []
        data = self.json_data[index]
        
        img_path = data['img_path']
        # Resolve relative paths against the project root (parent of src/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(img_path):
            img_path = os.path.join(project_root, img_path)
        if not os.path.exists(img_path):
            # Fallback: look for the bare filename inside images/
            img_name = os.path.basename(img_path)
            fallback_path = os.path.join(project_root, "images", img_name)
            if os.path.exists(fallback_path):
                img_path = fallback_path
            else:
                raise FileNotFoundError(f"Could not find image {img_name} at {data['img_path']} or {fallback_path}")

        imgs.append(Image.open(img_path).convert('RGB'))
        words.append(data['tokens'])
        # Convert string labels to IDs
        label_ids = [self.label2id.get(l, 0) for l in data['ner_tag']]
        label.append(label_ids)
        
        bboxes.append(data['bboxes'])

        encoding = self.processor(
            imgs,
            words,
            boxes = bboxes,
            word_labels = label,
            max_length=512,padding="max_length",truncation="longest_first",return_tensors='pt'
        )

        return {
            "input_ids" : torch.tensor(encoding["input_ids"],dtype=torch.int64).flatten(),
            "attention_mask" : torch.tensor(encoding["attention_mask"],dtype=torch.int64).flatten(),
            "bbox" : torch.tensor(encoding["bbox"],dtype=torch.int64).flatten(end_dim=1),
            "pixel_values" : torch.tensor(encoding["pixel_values"],dtype=torch.float32).flatten(end_dim=1),
            "labels" : torch.tensor(encoding["labels"],dtype=torch.int64).flatten() # fixed nesting
        }

