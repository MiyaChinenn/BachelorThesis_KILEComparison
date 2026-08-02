# Trainer
import os


import torch.nn as nn
import torch
# import torch.nn.functional as F
from transformers import LayoutLMv3ForTokenClassification
import torch.nn.functional as nnf


def loss_fn(pred,target):
    return nn.CrossEntropyLoss()(pred.view(-1,49),target.view(-1))


class ModelModule(nn.Module):
    def __init__(self,n_classes:int) -> None:
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path_local = os.path.join(base_dir, "inputs", "layoutlmv3Microsoft")
        model_path = model_path_local if os.path.exists(model_path_local) else "microsoft/layoutlmv3-base"
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path, num_labels=n_classes, ignore_mismatched_sizes=True)
        # Note: If you want to use a custom layer, ensure in_features matches model output (usually 768)
        # For now, we use the model's standard classification head.

    def forward(self,input_ids,attention_mask,bbox,pixel_values,labels=None):
        output = self.model(input_ids,attention_mask=attention_mask,bbox=bbox,pixel_values=pixel_values, labels=labels)

        logits = output.logits

        prob = nnf.softmax(logits, dim=-1)
        top_p, top_class = prob.topk(1, dim=-1)

        print("Probability score :", prob)
        print("top_p, top_class ",top_p, top_class)
        
        loss = output.loss

        return  logits, loss
