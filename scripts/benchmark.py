import time
import cv2
import numpy as np
from src.detection.detector import ShelfDetector

def benchmark_performance(model_path, test_images_path, num_runs=10):
    """Бенчмарк производительности системы"""
    
    detector = ShelfDetector(model_path)
    
    # Загрузка тестовых изображений
    test_images = []
    # ... (загрузка изображений)
    
    # Измерение времени обработки
    times = []
    for i in range(num_runs):
        for img in test_images:
            start_time = time.time()
            boxes, scores = detector.detect(img)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Среднее время обработки: {avg_time*1000:.2f} мс")
    print(f"FPS: {fps:.2f}")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'fps': fps,
        'total_images': len(test_images) * num_runs
    }

if __name__ == '__main__':
    benchmark_performance('results/checkpoints/best_model.pth', 'data/test/')
