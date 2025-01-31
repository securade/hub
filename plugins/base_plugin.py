from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BasePlugin(ABC):
    """Base class that all plugins must inherit from"""
    
    @property
    @abstractmethod
    def SLUG(self) -> str:
        """Return the unique identifier for this plugin"""
        pass
    
    @property
    @abstractmethod
    def NAME(self) -> str:
        """Return the display name for this plugin"""
        pass
    
    @abstractmethod
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image and return the annotated version
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional arguments that may be needed by the plugin
            
        Returns:
            Annotated image as numpy array
        """
        pass