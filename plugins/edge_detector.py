import cv2
import numpy as np
from plugins.base_plugin import BasePlugin

class EdgeDetector(BasePlugin):
    """Plugin that performs edge detection"""
    
    SLUG = "edge_detector"
    NAME = "Edge Detector"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Optional parameters can be passed in kwargs
        threshold1 = kwargs.get('threshold1', 100)
        threshold2 = kwargs.get('threshold2', 200)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Convert back to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb