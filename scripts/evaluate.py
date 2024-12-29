import torch
import sys
sys.path.append('yolov5')
from detect import run # yolov5 in reposunda detect.py adında bir dosya var. bunun içinde run fonksiyonu var.
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    weights='outputs/exp/weights/best.pt',
    source='data/images/test',
    conf_thres=0.25,
    iou_thres=0.45,
    device='0'
):
    # Test dataseti üzerinde değerlendirme yap
    results = run(
        weights=weights,
        source=source,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
        project='outputs',
        name='eval',
        save_txt=True,
        save_conf=True
    )
    
    return results

if __name__ == '__main__':
    evaluate_model() 