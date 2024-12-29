import yaml
from pathlib import Path
import torch
import sys
sys.path.append('yolov5')

def train_model(
    data_yaml='data/dataset.yaml',
    weights='yolov5s.pt',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='0'  # GPU kullanımı için
):
    # Eğitim komutunu oluştur
    command = f"""
    python yolov5/train.py \
        --data {data_yaml} \
        --weights {weights} \
        --epochs {epochs} \
        --batch-size {batch_size} \
        --img {img_size} \
        --device {device} \
        --project outputs \
        --name exp
    """
    
    # Eğitimi başlat
    import os
    os.system(command)

if __name__ == '__main__':
    train_model() 