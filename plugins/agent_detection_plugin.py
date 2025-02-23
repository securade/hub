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
import json
import re

class AgentLoopPlugin(BasePlugin):
    """Plugin that implements an agent loop between reasoning and vision models"""
    
    SLUG = "agent_detector"
    NAME = "Agent Loop Detector"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        self.vlm_model = None
        self.vlm_processor = None
        self.reasoning_model = None
        self.reasoning_tokenizer = None
        
        self.debug_dir = "debug_output/agent_detection"
        os.makedirs(self.debug_dir, exist_ok=True)

    def load_models(self):
        """Load required models"""
        try:
            if self.vlm_model is None:
                self.logger.info("Loading VLM model...")
                model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
                self.vlm_processor = AutoProcessor.from_pretrained(model_path)
                self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
                ).to(self.device)

            if self.reasoning_model is None:
                self.logger.info("Loading reasoning model...")
                reasoning_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                self.reasoning_tokenizer = AutoTokenizer.from_pretrained(reasoning_path)
                self.reasoning_model = AutoModelForCausalLM.from_pretrained(
                    reasoning_path,
                    torch_dtype=torch.float32
                ).to(self.device)

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def save_debug_image(self, image: np.ndarray, name: str):
        """Save debug image to disk"""
        try:
            path = os.path.join(self.debug_dir, f"{name}.jpg")
            cv2.imwrite(path, image)
            self.logger.debug(f"Saved debug image: {path}")
        except Exception as e:
            self.logger.error(f"Error saving debug image: {e}")

    def query_vlm(self, image: np.ndarray, question: str) -> str:
        """Query VLM model with an image and question"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image_pil}
                ]
            }]
            
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            inputs = {
                k: (v.to(device=self.device, dtype=torch.long) 
                   if k in ['input_ids', 'attention_mask'] 
                   else v.to(device=self.device, dtype=torch.float32))
                if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            with torch.inference_mode():
                output_ids = self.vlm_model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9
                )
            
            response = self.vlm_processor.batch_decode(
                output_ids,
                skip_special_tokens=True
            )[0]
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error in VLM query: {e}")
            return ""

    def get_agent_reasoning(self, context: Dict) -> Dict:
        """Get next step from reasoning model"""
        try:
            # Format the conversation history and context
            prompt = f"""You are a visual analysis agent trying to detect {context['target']} in an image.
Current understanding: {context['current_understanding']}
Previous observations: {context['observations']}

Think step by step about what information you need next and output a JSON response in this format:
{{
    "analysis": "Your step-by-step analysis of the current situation",
    "next_action": {{
        "type": "locate | verify | measure",
        "question": "Specific question to ask the vision model",
        "region": ["whole_image" or specific coordinates]
    }},
    "detection_status": "continuing | complete",
    "detected_objects": [
        {{
            "id": "unique_id",
            "coordinates": [x1, y1, x2, y2],
            "confidence": 0.XX
        }}
    ]
}}

Make your question as specific as possible to get precise location information."""

            inputs = self.reasoning_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.reasoning_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            response = self.reasoning_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No valid JSON in response")
            
            return json.loads(json_match.group(0))
            
        except Exception as e:
            self.logger.error(f"Error in reasoning: {e}")
            return {}

    def run_agent_loop(self, image: np.ndarray, target: str, debug: bool = False) -> List[Dict]:
        """Run the agent loop to detect objects"""
        detections = []
        context = {
            "target": target,
            "current_understanding": "Starting analysis",
            "observations": []
        }
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            if debug:
                self.logger.debug(f"Agent loop iteration {iteration}")
                self.logger.debug(f"Current context: {context}")
            
            # Get next action from reasoning model
            reasoning = self.get_agent_reasoning(context)
            if debug:
                self.logger.debug(f"Reasoning output: {reasoning}")
            
            if reasoning.get("detection_status") == "complete":
                detections.extend(reasoning.get("detected_objects", []))
                break
            
            # Execute next action
            next_action = reasoning.get("next_action", {})
            if not next_action:
                break
                
            # Query VLM with specific question
            response = self.query_vlm(image, next_action["question"])
            if debug:
                self.logger.debug(f"VLM Query: {next_action['question']}")
                self.logger.debug(f"VLM Response: {response}")
            
            # Update context
            context["observations"].append({
                "question": next_action["question"],
                "response": response
            })
            context["current_understanding"] = reasoning.get("analysis", "")
            
            # Add any detected objects
            if reasoning.get("detected_objects"):
                detections.extend(reasoning["detected_objects"])
        
        return detections

    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Main plugin entry point"""
        try:
            user_prompt = kwargs.get('prompt', '')
            debug = kwargs.get('debug', False)
            confidence_threshold = kwargs.get('confidence_threshold', 0.5)
            
            if not user_prompt:
                raise ValueError("User prompt is required")
            
            if debug:
                self.logger.debug(f"Processing with prompt: {user_prompt}")
                self.logger.debug(f"Confidence threshold: {confidence_threshold}")
                self.save_debug_image(image, "input")
            
            # Load models if needed
            if self.vlm_model is None or self.reasoning_model is None:
                self.load_models()
            
            # Run agent loop
            detections = self.run_agent_loop(image, user_prompt, debug)
            
            # Filter by confidence
            detections = [
                det for det in detections 
                if det.get('confidence', 0) >= confidence_threshold
            ]
            
            if debug:
                self.logger.debug(f"Final detections: {detections}")
            
            # Visualize results
            output_image = image.copy()
            for det in detections:
                coords = det.get('coordinates', [])
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    conf = det.get('confidence', 0.5)
                    
                    # Color based on confidence
                    color = (0, int(255 * conf), int(255 * (1 - conf)))
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                    
                    if debug:
                        # Add ID and confidence
                        text = f"{det.get('id', 'obj')} ({conf:.2f})"
                        cv2.putText(output_image, text, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if debug:
                self.save_debug_image(output_image, "output")
            
            return output_image
            
        except Exception as e:
            self.logger.error(f"Error in plugin execution: {e}")
            return image