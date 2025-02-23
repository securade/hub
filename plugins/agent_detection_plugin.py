import numpy as np
import cv2
import torch
from plugins.base_plugin import BasePlugin
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoTokenizer
)
from PIL import Image
import logging
from typing import List, Dict, Any, Tuple
import os

class AgentDetectionPlugin(BasePlugin):
    """Plugin that performs agentic object detection using VLM and reasoning"""
    
    SLUG = "agent_detector"
    NAME = "Agentic Object Detector"
    
    def __init__(self):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models as None - will load on first use
        self.vlm_model = None
        self.vlm_processor = None
        self.reasoning_model = None
        self.reasoning_tokenizer = None
        
        # Create debug directory
        self.debug_dir = "debug_output/agent_detection"
        os.makedirs(self.debug_dir, exist_ok=True)

    def load_models(self):
        """Load VLM and reasoning models if not already loaded"""
        try:
            if self.vlm_model is None:
                self.logger.info("Loading VLM model...")
                model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
                
                self.vlm_processor = AutoProcessor.from_pretrained(model_path)
                self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
                ).to(self.device)
                self.logger.info("VLM model loaded")

            if self.reasoning_model is None:
                self.logger.info("Loading reasoning model...")
                reasoning_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                
                self.reasoning_tokenizer = AutoTokenizer.from_pretrained(reasoning_path)
                self.reasoning_model = AutoModelForCausalLM.from_pretrained(
                    reasoning_path,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
                ).to(self.device)
                self.logger.info("Reasoning model loaded")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def save_debug_image(self, image: np.ndarray, name: str):
        """Save debug image"""
        path = os.path.join(self.debug_dir, f"{name}.jpg")
        cv2.imwrite(path, image)
        self.logger.debug(f"Saved debug image: {path}")

    def query_vlm(self, image: np.ndarray, question: str) -> str:
        """Query the VLM model about the image"""
        try:
            # Convert to PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Prepare messages
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image_pil}
                ]
            }]
            
            # Process input
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.inference_mode():
                generated_ids = self.vlm_model.generate(
                    **inputs, 
                    do_sample=False,
                    max_new_tokens=64
                )
                
            # Decode response
            response = self.vlm_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return response

        except Exception as e:
            self.logger.error(f"Error querying VLM: {e}")
            return ""

    def reason_about_objects(self, vlm_responses: List[str], image_context: str) -> List[Dict]:
        """Use reasoning model to analyze VLM responses and determine objects"""
        try:
            # Construct prompt
            prompt = f"""Given an image with the following context:
{image_context}

And these visual language model observations:
{' '.join(vlm_responses)}

<think>
1. Let me analyze the key objects mentioned
2. Consider their relationships and attributes
3. Determine confident object detections
4. Filter out uncertain or inconsistent detections
</think>

Please identify the key objects with their attributes and locations.
"""
            # Generate reasoning
            inputs = self.reasoning_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.reasoning_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            reasoning = self.reasoning_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract object detections from reasoning
            # This is a simplified version - you'd want more robust parsing
            objects = []
            for line in reasoning.split('\n'):
                if ':' in line:
                    obj_type, details = line.split(':', 1)
                    objects.append({
                        'type': obj_type.strip(),
                        'details': details.strip()
                    })
                    
            return objects

        except Exception as e:
            self.logger.error(f"Error in reasoning: {e}")
            return []

    def traditional_cv_detection(self, image: np.ndarray) -> List[Dict]:
        """Apply traditional CV techniques for additional detection"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze significant contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'area': area
                    })
            
            return detections

        except Exception as e:
            self.logger.error(f"Error in traditional CV: {e}")
            return []

    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Main processing pipeline"""
        try:
            self.logger.info("Starting agentic detection pipeline...")
            
            # Load models if needed
            if self.vlm_model is None or self.reasoning_model is None:
                self.load_models()
            
            # Save input image
            self.save_debug_image(image, "input")
            
            # 1. Query VLM with different perspectives
            vlm_responses = []
            questions = [
                "What objects do you see in this image?",
                "Describe the spatial relationship between objects",
                "What are the main colors and shapes you observe?",
                "Are there any people or vehicles in the scene?"
            ]
            
            for question in questions:
                response = self.query_vlm(image, question)
                vlm_responses.append(response)
            
            # 2. Get traditional CV detections
            cv_detections = self.traditional_cv_detection(image)
            
            # 3. Reason about combined observations
            image_context = f"Image size: {image.shape}, Traditional CV detected {len(cv_detections)} potential objects"
            reasoned_objects = self.reason_about_objects(vlm_responses, image_context)
            
            # 4. Visualize results
            output_image = image.copy()
            
            # Draw traditional CV detections
            for det in cv_detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add reasoned object labels
            y_offset = 30
            for obj in reasoned_objects:
                text = f"{obj['type']}: {obj['details']}"
                cv2.putText(output_image, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            self.save_debug_image(output_image, "output")
            
            return output_image

        except Exception as e:
            self.logger.error(f"Error in pipeline: {e}")
            return image