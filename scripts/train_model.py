import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from src.data.dataset import ShelfDataset
from src.data.preprocessing import get_transform
import argparse

def train_model(config_path, num_epochs=35):
    """Обучение модели детекции"""
    
    # Загрузка конфигурации
    # ... (подробная реализация)
    
    # Инициализация модели
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # фон + пустая область
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
    
    # Оптимизатор
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    
    # Цикл обучения
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        # Сохранение чекпоинта
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'results/checkpoints/model_epoch_{epoch+1}.pth')
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--epochs', type=int, default=35)
    args = parser.parse_args()
    
    train_model(args.config, args.epochs)
