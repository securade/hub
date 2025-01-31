import numpy as np
import cv2
from plugins.base_plugin import BasePlugin
import torch
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

class SafetyTransformPlugin(BasePlugin):
    """Plugin that detects persons and inpaints hard hats"""
    
    SLUG = "safety_transform"
    NAME = "Hard Hat Inpainting"
    
    def __init__(self):
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize SAM model
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        # Initialize inpainting model
        self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Move models to appropriate device
        self.sam_model.to(self.device)
        self.inpaint_model.to(self.device)
        
        # Enable optimizations
        self.inpaint_model.enable_attention_slicing()
        
        # Single prompt for hard hat
        self.prompt = "person wearing a bright yellow construction hard hat, safety helmet, clear photo"

    def generate_mask(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """Generate person head mask for hard hat"""
        h, w = image.shape[:2]
        try:
            # Convert bbox to head region (upper 1/3 of person bbox)
            x1, y1, x2, y2 = map(float, bbox)
            head_height = (y2 - y1) / 3  # Take top third for head
            y2 = y1 + head_height
            
            # Bound coordinates
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            
            # Convert to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Process input
            inputs = self.sam_processor(
                images=pil_image,
                input_boxes=[[[x1, y1, x2, y2]]],
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate mask
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
                
            # Process mask
            mask = outputs.pred_masks.squeeze().cpu().numpy()
            if len(mask.shape) > 2:
                mask = mask[0]
                
            # Resize and threshold mask
            mask = cv2.resize(mask.astype(np.float32), (w, h))
            mask = (mask > 0.5).astype(np.uint8)
            
            # Dilate mask slightly
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            return mask
            
        except Exception as e:
            print(f"Error in generate_mask: {str(e)}")
            return np.zeros((h, w), dtype=np.uint8)
            
    def inpaint_hardhat(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint hard hat on person"""
        try:
            orig_h, orig_w = image.shape[:2]
            target_size = (512, 512)
            
            # Resize input image and mask
            image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            if len(mask.shape) == 3:
                mask = mask[:,:,0]
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
            mask_binary = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)
            
            # Convert to PIL
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            mask_pil = Image.fromarray(mask_binary)
            
            # Run inpainting
            with torch.no_grad():
                output = self.inpaint_model(
                    prompt=self.prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    negative_prompt="blurry, distorted",
                ).images[0]
                
            # Convert output and resize back to original dimensions
            result = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create 3D mask for blending (using original sized mask)
            mask_3d = np.stack([mask] * 3, axis=-1)
            
            # Blend with original image
            return np.where(mask_3d, result, image)
            
        except Exception as e:
            print(f"Error in inpaint_hardhat: {str(e)}")
            return image
            
    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process image adding hard hats to detected persons"""
        try:
            # Get YOLO detector from kwargs
            detector = kwargs.get('yolo_detector')
            if detector is None:
                raise ValueError("SafetyTransform plugin requires 'yolo_detector'")
                
            # Run detection
            detector.load_cv2mat(image)
            detector.inference()
            
            output_image = image.copy()
            person_found = False
            
            # Process only person detections
            for detection in detector.predicted_bboxes_PascalVOC:
                label = detection[0]
                
                # Skip if not a person detection
                if label.lower() != 'person':
                    continue
                    
                person_found = True
                confidence = float(detection[-1])
                
                # Skip low confidence detections
                if confidence < 0.5:
                    continue
                    
                # Get bounding box coordinates
                bbox = detection[1:5]
                
                # Generate mask for person's head
                mask = self.generate_mask(image, bbox)
                
                if mask is None or not mask.any():
                    continue
                    
                # Inpaint hard hat
                output_image = self.inpaint_hardhat(output_image, mask)
                
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # If no persons were detected, return original image
            if not person_found:
                print("No persons detected in image")
                return image
                
            return output_image
            
        except Exception as e:
            print(f"Error in run method: {str(e)}")
            return image