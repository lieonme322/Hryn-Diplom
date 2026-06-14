class FillAnalyzer:
    """Анализатор заполненности полок"""
    
    def __init__(self, shelf_area=None):
        self.shelf_area = shelf_area
    
    def calculate_fill_percentage(self, boxes, shelf_area=None):
        """
        Расчёт процента заполненности полки
        
        Параметры:
            boxes: список bounding boxes обнаруженных товаров
            shelf_area: площадь анализируемой полки (опционально)
        
        Возвращает:
            fill_percentage: процент заполненности (%)
        """
        if shelf_area is not None:
            self.shelf_area = shelf_area
        
        if self.shelf_area is None or self.shelf_area == 0:
            raise ValueError("Не указана площадь полки")
        
        total_objects_area = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            object_area = (x2 - x1) * (y2 - y1)
            total_objects_area += object_area
        
        fill_percentage = (total_objects_area / self.shelf_area) * 100
        return round(fill_percentage, 2)
    
    def classify_fill_level(self, fill_percentage):
        """
        Классификация уровня заполненности
        
        Возвращает:
            level: 'full', 'normal', 'low', 'critical'
        """
        if fill_percentage >= 90:
            return 'full', 'Полностью заполнено'
        elif fill_percentage >= 55:
            return 'normal', 'Средняя заполненность'
        elif fill_percentage >= 35:
            return 'low', 'Низкая заполненность'
        else:
            return 'critical', 'Критически пусто'
    
    def generate_alert(self, fill_percentage):
        """Генерация уведомления при низкой заполненности"""
        level, message = self.classify_fill_level(fill_percentage)
        
        if level in ['low', 'critical']:
            return {
                'alert': True,
                'level': level,
                'message': f'{message}: {fill_percentage}%',
                'action': 'Требуется пополнение товаров'
            }
        return {'alert': False, 'message': message}
