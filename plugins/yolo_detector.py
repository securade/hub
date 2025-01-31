import numpy as np
from plugins.base_plugin import BasePlugin

class YOLODetector(BasePlugin):
    """YOLO detector plugin"""
    
    SLUG = "yolo_detector"
    NAME = "YOLO Object Detector"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Extract detector from kwargs
        detector = kwargs.get('yolo_detector')
        if detector is None:
            raise ValueError("YOLODetector plugin requires 'yolo_detector' in kwargs")
            
        detector.load_cv2mat(image)
        detector.inference()
        return detector.image.copy()