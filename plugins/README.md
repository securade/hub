# Securade.ai HUB Plugins

This directory contains plugins for the Securade.ai HUB camera processing system. Plugins provide different ways to process and analyze camera feeds, from basic image processing to complex computer vision tasks.

## Plugin System Overview

The plugin system allows you to create custom image processing modules that can be dynamically loaded and selected in the HUB interface. Each plugin receives an input image and can return a processed/annotated version of that image.

## Creating a New Plugin

To create a new plugin:

1. Create a new Python file in this directory (e.g., `my_plugin.py`)
2. Import the base plugin class: `from plugins.base_plugin import BasePlugin`
3. Create your plugin class that inherits from `BasePlugin`
4. Implement the required properties and methods

Here's a template for a new plugin:

```python
import numpy as np
from plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    # Unique identifier for your plugin
    SLUG = "my_plugin"
    
    # Display name that will appear in the UI
    NAME = "My Custom Plugin"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the input image
        
        Args:
            image: Input image as numpy array (RGB format)
            **kwargs: Additional arguments passed from the UI
            
        Returns:
            Processed image as numpy array (RGB format)
        """
        # Your image processing code here
        processed_image = image  # Replace with actual processing
        return processed_image
```

### Required Properties and Methods

- `SLUG`: A unique string identifier for your plugin (lowercase, no spaces)
- `NAME`: A human-readable name that will be displayed in the UI
- `run(self, image, **kwargs)`: The main processing method that receives and returns an image

### Accessing Additional Arguments

Your plugin can access additional arguments through the `kwargs` parameter. The system provides several common arguments:

```python
def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
    # Access YOLO detector if needed
    detector = kwargs.get('yolo_detector')
    
    # Access other optional parameters
    param1 = kwargs.get('param1', default_value)
    param2 = kwargs.get('param2')
    
    # Check if required parameters are provided
    if param2 is None:
        raise ValueError("Plugin requires 'param2' parameter")
```

## Example Plugins

### 1. YOLO Detector (`yolo_detector.py`)
Demonstrates how to use the YOLOv7 detector for object detection:
```python
class YOLODetector(BasePlugin):
    SLUG = "yolo_detector"
    NAME = "YOLO Object Detector"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        detector = kwargs.get('yolo_detector')
        detector.load_cv2mat(image)
        detector.inference()
        return detector.image.copy()
```

### 2. Edge Detector (`edge_detector.py`)
Shows how to implement basic image processing using OpenCV:
```python
class EdgeDetector(BasePlugin):
    SLUG = "edge_detector"
    NAME = "Edge Detector"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        threshold1 = kwargs.get('threshold1', 100)
        threshold2 = kwargs.get('threshold2', 200)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
```

### 3. Agent Detector (`agent_detection_plugin.py`)
Uses Gemini Flash model for intelligent object detection with natural language prompts:
```python
class GeminiAgentPlugin(BasePlugin):
    SLUG = "agent_detector"
    NAME = "Agent Detector"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        prompt = kwargs.get('prompt', '')  # Natural language prompt
        debug = kwargs.get('debug', False)
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        
        # Uses Gemini Flash to detect objects based on prompt
        # Draws bounding boxes around detected objects
        # Returns annotated image
```

### 4. Safety Transform (`safety_transform_plugin.py`)
Transforms detected persons without safety gear to appear wearing safety equipment:
```python
class SafetyTransformPlugin(BasePlugin):
    SLUG = "safety_transform"
    NAME = "Safety Gear Transform"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        detector = kwargs.get('yolo_detector')  # Requires YOLO detector
        num_inference_steps = kwargs.get('num_inference_steps', 20)
        guidance_scale = kwargs.get("guidance_scale", 8.0)
        
        # 1. Detects persons without safety vest using YOLO
        # 2. Generates segmentation mask using SAM model
        # 3. Inpaints safety gear using Stable Diffusion
        # Returns transformed image
```

### 5. Content Moderator (`content_moderation_plugin.py`)
Analyzes images for inappropriate content across multiple categories:
```python
class ContentModerationPlugin(BasePlugin):
    SLUG = "content_moderator"
    NAME = "Content Moderator"
    
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        debug = kwargs.get('debug', False)
        safety_threshold = kwargs.get('safety_threshold', 0.7)
        
        # Analyzes image for categories: nudity, violence, gore,
        # hate symbols, drugs, weapons, etc.
        # Draws bounding boxes around detected inappropriate content
        # Shows overall safety status (SAFE/UNSAFE) with score
        # Returns annotated image
```

## Best Practices

1. **Error Handling**: Handle missing dependencies and invalid parameters gracefully
2. **Documentation**: Include docstrings and comments explaining your plugin's functionality
3. **Input Validation**: Verify input image format and required parameters
4. **Performance**: Optimize processing for real-time video analysis when possible
5. **Memory Management**: Clean up resources and avoid memory leaks

## Available Dependencies

Your plugins can use these pre-installed libraries:
- OpenCV (cv2)
- NumPy
- PyTorch
- Other common Python libraries

## Debugging Tips

1. Print statements will appear in the server console
2. Use try/except blocks to catch and log errors
3. Test your plugin with different image sizes and formats
4. Verify your plugin works with both still images and video streams

## Contributing

When contributing new plugins:
1. Follow the naming conventions and code style of existing plugins
2. Include example usage and documentation
3. Test thoroughly with different inputs
4. Update this README if adding new features or dependencies