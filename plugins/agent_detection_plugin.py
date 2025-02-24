import numpy as np
import cv2
import base64
import json
import logging
from typing import List, Dict, Any, Optional
from plugins.base_plugin import BasePlugin
from openai import OpenAI
from PIL import Image
import io
import os
import re

class GeminiAgentPlugin(BasePlugin):
    """Plugin that uses Gemini Flash model for intelligent object detection"""
    
    SLUG = "agent_detector"
    NAME = "Agent Detector"
    
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.MODEL = "gemini-2.0-flash"  # Using Gemini Flash model
        
        # Create debug directory if needed
        self.debug_dir = "debug_output/agent_detector"
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def save_debug_image(self, image: np.ndarray, name: str):
        """Save debug image with bounding boxes for inspection"""
        path = os.path.join(self.debug_dir, f"{name}.jpg")
        cv2.imwrite(path, image)
        self.logger.debug(f"Saved debug image: {path}")

    def encode_image(self, image: np.ndarray) -> str:
        """Convert CV2 image to base64 string, ensuring RGB format"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')
    
    def extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON array from model response text"""
        try:
            # Look for array pattern [...] 
            array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if array_match:
                return array_match.group(0)
                
            # Look for pattern starting with just {
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return f"[{json_match.group(0)}]"
                
            self.logger.error(f"No JSON pattern found in response: {response_text}")
            return "[]"
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {e}")
            return "[]"
     
    def normalize_coordinates(self, x: float, y: float, width: int, height: int) -> tuple[int, int]:
        """
        Convert normalized coordinates (0-1000) to image pixel coordinates
        
        Args:
            x: x-coordinate in 0-1000 range
            y: y-coordinate in 0-1000 range
            width: Image width in pixels
            height: Image height in pixels
        
        Returns:
            Tuple of (x, y) in pixel coordinates
        """
        # Convert from 0-1000 range to 0-1 range
        x_norm = float(x) / 1000
        y_norm = float(y) / 1000
        
        # Convert to pixel coordinates
        x_px = int(x_norm * width)
        y_px = int(y_norm * height)
        
        return x_px, y_px

    def validate_and_adjust_bbox(self, bbox: List[float], width: int, height: int) -> Optional[List[int]]:
        """
        Validate and convert normalized coordinates to pixel coordinates
        
        Args:
            bbox: List of [y1, x1, y2, x2] coordinates in 0-1000 range
            width: Image width in pixels
            height: Image height in pixels
        """
        try:
            if len(bbox) != 4:
                self.logger.warning(f"Invalid bbox length: {len(bbox)}")
                return None
                
            y1, x1, y2, x2 = map(float, bbox)
            
            # Validate normalized coordinates are in 0-1000 range
            if not all(0 <= coord <= 1000 for coord in [y1, x1, y2, x2]):
                self.logger.warning(f"Coordinates out of 0-1000 range: {bbox}")
                return None
                
            # Convert to pixel coordinates
            x1_px, y1_px = self.normalize_coordinates(x1, y1, width, height)
            x2_px, y2_px = self.normalize_coordinates(x2, y2, width, height)
            
            # Ensure correct ordering
            x1_px, x2_px = min(x1_px, x2_px), max(x1_px, x2_px)
            y1_px, y2_px = min(y1_px, y2_px), max(y1_px, y2_px)
            
            # Ensure box isn't too small
            if (x2_px - x1_px) < 10 or (y2_px - y1_px) < 10:
                self.logger.warning(f"Bounding box too small: {[x1_px, y1_px, x2_px, y2_px]}")
                return None
                
            return [x1_px, y1_px, x2_px, y2_px]
            
        except Exception as e:
            self.logger.error(f"Error processing bbox {bbox}: {e}")
            return None

    def extract_bounding_boxes(self, image: np.ndarray, analysis: str, prompt: str, confidence_threshold) -> List[Dict]:
        """Extract precise bounding box coordinates from Gemini with improved prompt"""
        try:
            base64_image = self.encode_image(image)
            height, width = image.shape[:2]
            
            bbox_prompt = f"""Focus only on finding the main {prompt} in the image.

            Rules:
            1. Only detect distinct, non-overlapping objects
            2. For each object, provide ONE tight bounding box
            3. Coordinates must be in [y1, x1, y2, x2] format and 0-1000 range
            4. Only include objects you're very confident about
            5. Label each object descriptively but concisely
            
            Return this exact JSON format:
            [
                {{
                    "id": "unique_identifier",
                    "label": "brief_description",
                    "confidence": 0.XX,
                    "box_2d": [y1, x1, y2, x2]
                }}
            ]"""

            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": bbox_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }]
            )
            
            json_str = self.extract_json_from_response(response.choices[0].message.content)
            self.logger.debug(f"Raw bbox JSON: {json_str}")
            
            detections = json.loads(json_str)
            
            # Filter by confidence first
            detections = [d for d in detections if d.get('confidence', 0) > 0.5]
            
            # Convert coordinates
            valid_detections = []
            for det in detections:
                bbox = det.get('box_2d', [])
                adjusted_bbox = self.validate_and_adjust_bbox(bbox, width, height)
                if adjusted_bbox:
                    det['bbox'] = adjusted_bbox
                    det['box_2d'] = bbox
                    valid_detections.append(det)
            
            # Apply NMS to remove overlaps
            final_detections = self.non_max_suppression(valid_detections, iou_threshold=confidence_threshold)
            
            self.logger.info(f"Found {len(final_detections)} objects after NMS")
            return final_detections
            
        except Exception as e:
            self.logger.error(f"Error in bounding box extraction: {e}")
            return []

    def draw_detections(self, image: np.ndarray, detections: List[Dict], debug: bool = False) -> np.ndarray:
        """Draw verified detections on the image"""
        output = image.copy()
        
        for det in detections:
            bbox = det.get('bbox', [])  # Use converted pixel coordinates
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            conf = det.get('confidence', 0.5)
            label = det.get('label', 'object')
            
            # Use a more visible color scheme
            color = (0, int(255 * conf), 0)  # Green with confidence-based intensity
            thickness = max(1, int(conf * 3))  # Thicker lines for higher confidence
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Add label if in debug mode
            if debug:
                text = f"{label} ({conf:.2f})"
                font_scale = 0.6
                font_thickness = 2
                
                # Get text size
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw background for text
                cv2.rectangle(
                    output, 
                    (x1, y1 - text_h - 8), 
                    (x1 + text_w + 4, y1),
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output, text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
                
        return output
        
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate intersection over union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping boxes"""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        kept_detections = []
        for det in detections:
            should_keep = True
            bbox1 = det.get('bbox', [])
            
            if not bbox1 or len(bbox1) != 4:
                continue
                
            # Check against all kept detections
            for kept_det in kept_detections:
                bbox2 = kept_det.get('bbox', [])
                if not bbox2 or len(bbox2) != 4:
                    continue
                    
                iou = self.calculate_iou(bbox1, bbox2)
                if iou > iou_threshold:
                    should_keep = False
                    break
                    
            if should_keep:
                kept_detections.append(det)
                
        return kept_detections

    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Main plugin entry point with improved detection"""
        try:
            prompt = kwargs.get('prompt', '')
            debug = kwargs.get('debug', False)
            confidence_threshold = kwargs.get('confidence_threshold', 0.5)
            
            if not prompt:
                raise ValueError("Prompt is required")
            
            self.logger.info(f"Starting detection for: {prompt}")
            if debug:
                self.save_debug_image(image, "1_input")
            
            # Streamline the process - go directly to detection
            detections = self.extract_bounding_boxes(image, "", prompt, confidence_threshold)
            
            if detections:
                self.logger.info(f"Found {len(detections)} valid detections")
                output_image = self.draw_detections(image, detections, debug=debug)
            else:
                self.logger.warning(f"No valid detections found for: {prompt}")
                output_image = image.copy()
            
            if debug:
                self.save_debug_image(output_image, "2_final_output")
            
            return output_image
                
        except Exception as e:
            self.logger.error(f"Error in plugin execution: {e}")
            return image