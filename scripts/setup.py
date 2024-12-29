import subprocess
import sys

def setup_environment():
    # YOLOv5'i klonla
    #subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
    
    # Gereksinimleri yükle
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'yolov5/requirements.txt'])
    
    # Ek gereksinimler
    additional_requirements = [
        'opencv-python',
        'matplotlib',
        'pandas',
        'seaborn'
    ]
    for req in additional_requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', req])

if __name__ == '__main__':
    setup_environment() 