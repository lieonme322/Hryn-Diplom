import torch
import cv2
import numpy as np
from torchvision import transforms

class ShelfDetector:
    """Детектор пустых зон на полках (Faster R-CNN)"""
    
    def __init__(self, model_path, device='cuda', confidence_threshold=0.5):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        
        # Загрузка модели
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Трансформации
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect(self, image):
        """Обнаружение пустых зон на изображении"""
        original_height, original_width = image.shape[:2]
        
        # Предобработка
        image_tensor = self.transform(image).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Инференс
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Постобработка
        boxes = []
        scores = []
        
        for pred in predictions[0]['boxes']:
            score = predictions[0]['scores'][i].item()
            if score >= self.confidence_threshold:
                box = pred.cpu().numpy().astype(int)
                boxes.append(box)
                scores.append(score)
        
        return boxes, scores
    
    def draw_results(self, image, boxes, scores):
        """Отрисовка результатов детекции"""
        result_image = image.copy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_image, f'empty: {score:.2f}', 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        return result_image
