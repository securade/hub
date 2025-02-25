import numpy as np
import cv2
import base64
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from plugins.base_plugin import BasePlugin
from openai import OpenAI
from PIL import Image
import io
import re
import time

class ContentModerationPlugin(BasePlugin):
    """Plugin that uses Gemini Flash model to detect inappropriate content in images"""
    
    SLUG = "content_moderator"
    NAME = "Content Moderator"
    
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize OpenAI client for Gemini API
        self.client = None
        self.MODEL = "gemini-2.0-flash"
        
        # Create debug directory if needed
        self.debug_dir = "debug_output/content_moderator"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Define content categories to detect
        self.categories = [
            "nudity", 
            "sexual_content", 
            "violence", 
            "gore", 
            "hate_symbols",
            "drugs",
            "weapons"
        ]
        
        # Threshold for flagging content
        self.safety_threshold = 0.7
        
    def initialize_client(self):
        """Initialize the OpenAI client with API key from environment"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("No OPENAI_API_KEY found in environment variables")
            raise ValueError("OPENAI_API_KEY environment variable must be set")
            
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.logger.info("OpenAI client initialized for Gemini API")
        
    def save_debug_image(self, image: np.ndarray, name: str):
        """Save debug image for inspection"""
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
        """Extract JSON object from model response text"""
        try:
            # Look for JSON object pattern {...}
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json_match.group(0)
                
            # Look for array pattern [...]
            array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if array_match:
                return f'{{"results": {array_match.group(0)}}}'
                
            self.logger.error(f"No JSON pattern found in response: {response_text}")
            return '{}'
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {e}")
            return '{}'
    
    def analyze_content(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image for inappropriate content using Gemini Flash"""
        if self.client is None:
            self.initialize_client()
            
        try:
            base64_image = self.encode_image(image)
            
            # Create a prompt that asks Gemini to analyze the image for inappropriate content
            analysis_prompt = f"""
            Analyze this image for inappropriate content. Focus on detecting any of these categories:
            {', '.join(self.categories)}
            
            For each category, provide:
            1. A confidence score from 0.0 to 1.0 indicating how likely the image contains this content
            2. Precise bounding box coordinates [x1, y1, x2, y2] in 0-1000 range for any detected inappropriate content
            3. Brief description of what was detected
            
            Return your analysis in this exact JSON format:
            {{
                "overall_safety_score": 0.XX,
                "safe_for_marketplace": true/false,
                "categories": {{
                    "category_name": {{
                        "detected": true/false,
                        "confidence": 0.XX,
                        "description": "brief description",
                        "regions": [
                            {{
                                "box_2d": [x1, y1, x2, y2],
                                "confidence": 0.XX
                            }}
                        ]
                    }}
                }},
                "explanation": "Brief explanation of your assessment."
            }}
            
            If no inappropriate content is detected, set all confidence scores to 0.0 and "safe_for_marketplace" to true.
            """

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }]
            )
            
            # Extract and parse JSON response
            json_str = self.extract_json_from_response(response.choices[0].message.content)
            self.logger.debug(f"Content analysis JSON: {json_str}")
            
            try:
                analysis = json.loads(json_str)
                return analysis
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                return {
                    "overall_safety_score": 1.0,
                    "safe_for_marketplace": True,
                    "categories": {},
                    "explanation": "Failed to analyze image content."
                }
            
        except Exception as e:
            self.logger.error(f"Error in content analysis: {e}")
            return {
                "overall_safety_score": 1.0,
                "safe_for_marketplace": True,
                "categories": {},
                "explanation": f"Error during analysis: {str(e)}"
            }
    
    def normalize_coordinates(self, coords: List[float], width: int, height: int) -> List[int]:
        """Convert normalized coordinates (0-1000) to image pixel coordinates"""
        if len(coords) != 4:
            return [0, 0, 100, 100]
            
        x1, y1, x2, y2 = coords
        
        # Convert from 0-1000 range to 0-1 range
        x1_norm, y1_norm = float(x1) / 1000, float(y1) / 1000
        x2_norm, y2_norm = float(x2) / 1000, float(y2) / 1000
        
        # Convert to pixel coordinates
        x1_px = max(0, int(x1_norm * width))
        y1_px = max(0, int(y1_norm * height))
        x2_px = min(width, int(x2_norm * width))
        y2_px = min(height, int(y2_norm * height))
        
        return [x1_px, y1_px, x2_px, y2_px]
            
    def draw_analysis_results(self, image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        """Draw analysis results on the image"""
        output = image.copy()
        height, width = output.shape[:2]
        
        # Get categories
        categories = analysis.get("categories", {})
        
        # Draw box for each detected region
        for category_name, category_data in categories.items():
            if not category_data.get("detected", False):
                continue
                
            regions = category_data.get("regions", [])
            for region in regions:
                # Skip regions with low confidence
                confidence = region.get("confidence", 0)
                if confidence < self.safety_threshold:
                    continue
                    
                # Get bounding box coordinates
                bbox = region.get("box_2d", [])
                if not bbox or len(bbox) != 4:
                    continue
                    
                # Convert normalized coordinates to pixels
                x1, y1, x2, y2 = self.normalize_coordinates(bbox, width, height)
                
                # Determine color based on category (red for high confidence)
                # Scale from yellow to red based on confidence
                g_value = int(255 * (1 - confidence))
                color = (0, g_value, 255)  # BGR format - red with variable green
                
                # Draw box and label
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{category_name} ({confidence:.2f})"
                font_scale = 0.5
                font_thickness = 1
                
                # Get text size
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw background for text
                cv2.rectangle(
                    output, 
                    (x1, y1 - text_h - 5), 
                    (x1 + text_w, y1),
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output, 
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
        
        # Draw overall safety status at the top
        safe = analysis.get("safe_for_marketplace", True)
        safety_score = analysis.get("overall_safety_score", 1.0)
        
        status_color = (0, 255, 0) if safe else (0, 0, 255)  # Green if safe, red if unsafe
        status_text = "SAFE" if safe else "UNSAFE"
        
        # Draw status box at the top of the image
        cv2.rectangle(output, (0, 0), (width, 40), status_color, -1)
        cv2.putText(
            output,
            f"Content Status: {status_text} - Safety Score: {safety_score:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            2
        )
        
        return output
        
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Main plugin entry point"""
        try:
            # Get parameters
            debug = kwargs.get('debug', False)
            custom_threshold = kwargs.get('safety_threshold')
            
            # Update threshold if provided
            if custom_threshold is not None:
                self.safety_threshold = float(custom_threshold)
            
            self.logger.info(f"Starting content moderation with threshold: {self.safety_threshold}")
            
            # Save input image for debugging
            if debug:
                self.save_debug_image(image, "1_input")
                
            # Start timing for performance measurement
            start_time = time.time()
            
            # Analyze image content
            analysis = self.analyze_content(image)
            
            # Log analysis time
            analysis_time = time.time() - start_time
            self.logger.info(f"Content analysis completed in {analysis_time:.2f} seconds")
            
            # Draw results on image
            output_image = self.draw_analysis_results(image, analysis)
            
            # Save output image for debugging
            if debug:
                self.save_debug_image(output_image, "2_output")
                
            # Log overall result
            safety_score = analysis.get("overall_safety_score", 1.0)
            safe = analysis.get("safe_for_marketplace", True)
            self.logger.info(f"Content moderation result: Safe={safe}, Score={safety_score:.2f}")
            
            return output_image
            
        except Exception as e:
            self.logger.error(f"Error in content moderation plugin: {e}")
            # Return original image if anything fails
            return image
