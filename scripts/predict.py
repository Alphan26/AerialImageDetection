import cv2
import torch
import sys
sys.path.append('yolov5')

class ObjectDetector:
    def __init__(self, weights='outputs/exp/weights/best.pt', conf_thres=0.25):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
        self.model.conf = conf_thres
    
    def detect(self, image_path):
        # Görüntüyü yükle ve tahmin yap
        results = self.model(image_path)
        
        # Sonuçları pandas DataFrame'e dönüştür
        predictions = results.pandas().xyxy[0]
        
        return predictions
    
    def draw_predictions(self, image_path, predictions):
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        
        # Her bir tahmin için bbox çiz
        for _, pred in predictions.iterrows():
            x1, y1, x2, y2 = map(int, [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']])
            label = f"{pred['name']} {pred['confidence']:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image

if __name__ == '__main__':
    detector = ObjectDetector()
    # Örnek kullanım
    image_path = 'path/to/test/image.jpg'
    predictions = detector.detect(image_path)
    result_image = detector.draw_predictions(image_path, predictions)
    cv2.imwrite('outputs/result.jpg', result_image) 