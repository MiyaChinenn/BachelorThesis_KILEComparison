import torch
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np

def train_fn(data_loader, model, optimizer, accumulation_steps=1):
    model.train()
    final_loss = 0
    optimizer.zero_grad()
    
    for i, data in enumerate(tqdm(data_loader, total=len(data_loader))):
        _, loss = model(**data)
        
        # Scale the loss if we are accumulating gradients
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
            
        loss.backward()
        
        # Update weights only after accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()
            
        final_loss += loss.item() * (accumulation_steps if accumulation_steps > 1 else 1)
        
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, id2label):
    model.eval()
    final_loss = 0
    full_preds = []
    full_labels = []
    
    for data in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            outputs, loss = model(**data)
        
        final_loss += loss.item()
        
        predictions = outputs.argmax(dim=-1).cpu().numpy()
        labels = data['labels'].cpu().numpy()
        
        for pred, label in zip(predictions, labels):
            true_labels = []
            true_preds = []
            for p, l in zip(pred, label):
                if l != -100:
                    true_labels.append(id2label[l])
                    true_preds.append(id2label[p])
            
            full_labels.append(true_labels)
            full_preds.append(true_preds)
            
    precision = precision_score(full_labels, full_preds)
    recall = recall_score(full_labels, full_preds)
    f1 = f1_score(full_labels, full_preds)
    
    return final_loss / len(data_loader), precision, recall, f1