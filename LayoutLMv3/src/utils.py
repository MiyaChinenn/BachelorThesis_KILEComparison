# utils
from paddleocr import PaddleOCR
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os

# Legacy NumPy workaround
if not hasattr(np, "int"):
    np.int = int

def read_json(json_path:str)->dict:
    with open(json_path,'r') as fp:
        data = json.loads(fp.read())
    return data


def train_data_format(json_to_dict:list):

    final_list = []
    count=0
    for item in json_to_dict:
        count = count+1
        # print(item['annotations'])
        test_dict = {"id":int,"tokens":[],"bboxes":[],"ner_tag":[]}
        test_dict["id"] = count
        test_dict["img_path"] = item['file_name']
        for cont in item['annotations']:
            test_dict['tokens'].append(cont['text'])
            test_dict['bboxes'].append(cont['box'])
            test_dict['ner_tag'].append(cont['label'])


        final_list.append(test_dict)
    #print(final_list)
    return final_list


# lazy initialization for OCR
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(lang='en', use_gpu=False)
    return _ocr



def scale_bounding_box(box:list[int], width:float, height:float) -> list[int]:
    # Normalizing to 0-1000 scale (standard for LayoutLM models)
    x1 = 1000 * box[0] / width
    y1 = 1000 * box[1] / height
    x2 = 1000 * (box[0] + box[2]) / width
    y2 = 1000 * (box[1] + box[3]) / height
    
    return [int(x1), int(y1), int(x2), int(y2)]

def process_bbox(box:list):
    # PaddleOCR [topleft, topright, bottomright, bottomleft]
    # returns [x, y, width, height]
    x = box[0][0]
    y = box[0][1]
    w = box[2][0] - box[0][0]
    h = box[2][1] - box[0][1]
    return [x, y, w, h]

def dataSetFormat(img_file):
    width, height = img_file.size
    ress = get_ocr().ocr(np.asarray(img_file))

    test_dict = {'tokens':[], "bboxes":[]}
    test_dict['img_path'] = img_file

    if ress and ress[0]:
        for item in ress[0]:
            test_dict['tokens'].append(item[1][0])
            test_dict['bboxes'].append(scale_bounding_box(process_bbox(item[0]), width, height))

    return test_dict, width, height




def plot_img(im, bbox_list, label_list, prob_list, width, height, output_name="test_image.jpg"):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    ax = plt.gca()
    
    for i, item in enumerate(bbox_list):
        # item is [x1, y1, x2, y2] in 0-1000 range
        x1 = item[0] * width / 1000
        y1 = item[1] * height / 1000
        w = (item[2] - item[0]) * width / 1000
        h = (item[3] - item[1]) * height / 1000
        
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        label = label_list[i]
        prob = prob_list[i]
        
        try:
            ax.text(
                x1, y1,
                f"{label}: {prob:.2f}",
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2},
                fontsize=8,
                color='red',
                verticalalignment='bottom'
            )
        except Exception as e:
            print(f"[Error] Failed to process {output_name}: {e}")

    plt.axis('off')
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Done - image saved to {output_name}")

def plot_metrics(train_losses, precisions):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Precision plot
    plt.subplot(2, 2, 2)
    # Wait, subplot(1, 2, 2) is better if we only have 2 plots
    plt.subplot(1, 2, 2)
    plt.plot(epochs, precisions, 'g-o', label='Precision')
    plt.title('Precision (Entity Level)')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.close()