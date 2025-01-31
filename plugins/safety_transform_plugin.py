import numpy as np
import cv2
from plugins.base_plugin import BasePlugin
import torch
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os

class SafetyTransformPlugin(BasePlugin):
    """Plugin that transforms detected person to wear safety gear"""
    
    SLUG = "safety_transform"
    NAME = "Safety Gear Transform"
    
    def __init__(self):
        # Create debug directory
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Initialize models as None - will load on first use
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.sam_model = None
        self.sam_processor = None
        self.inpaint_model = None
        
        # Inpainting prompt
        self.prompt = (
            "A person wearing a bright colored safety harness and vest."
            "Only the clothing is changed; the person's face and pose remain the same."
        )     
        
    def save_debug_image(self, image, name):
        """Save debug image with automatic numbering"""
        path = os.path.join(self.debug_dir, f"{name}.jpg")
        
        if isinstance(image, Image.Image):
            image.save(path)
        else:
            cv2.imwrite(path, image)
        print(f"Saved debug image: {path}")

    def load_models(self):
        """Load models if not already loaded"""
        try:
            if self.sam_model is None:
                print("Loading SAM model...")
                self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
                self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
                print("SAM model loaded")

            if self.inpaint_model is None:
                print("Loading inpainting model...")
                self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    safety_checker=None
                ).to(self.device)
                print("Inpainting model loaded")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def get_person_detection(self, detector, image):
        """Get the best person detection from YOLO"""
        # Run YOLO detection
        detector.load_cv2mat(image)
        detector.inference()
        
        # Find person with highest confidence
        best_person = None
        highest_conf = -1
        
        for detection in detector.predicted_bboxes_PascalVOC:
            if detection[0].lower() == 'no-safety vest':
                conf = float(detection[-1])
                if conf > highest_conf:
                    highest_conf = conf
                    best_person = detection
        
        if best_person:
            print(f"Found person with confidence: {highest_conf}")
            
            # Get coordinates and map back to original image space
            box = np.array(best_person[1:5])  # [x1, y1, x2, y2]
            box -= np.array(detector.dwdh * 2)  # Remove padding
            box /= detector.ratio  # Scale back to original size
            
            # Save detection visualization
            debug_img = image.copy()
            cv2.rectangle(debug_img, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 2)
            self.save_debug_image(debug_img, "1_person_detection")
            
            return box
        
        return None

    def generate_mask(self, image, bbox):
        """Generate segmentation mask using SAM with debug prints."""
        try:
            # Convert image to RGB for SAM
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Get image dimensions
            h, w = image.shape[:2]
            x1, y1, x2, y2 = map(float, bbox)
            
            print(f"[DEBUG] Original bbox (pixels): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Ensure coordinates are valid
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            print(f"[DEBUG] Clamped bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Correctly formatted input box: [[[x1, y1, x2, y2]]]
            input_boxes = [[[x1, y1, x2, y2]]]

            print(f"[DEBUG] Input box shape (should be [[[x1, y1, x2, y2]]]): {input_boxes}")

            # Process image and generate mask
            inputs = self.sam_processor(
                images=image_pil,
                input_boxes=input_boxes,  # Correct shape
                return_tensors="pt"
            )

            print(f"[DEBUG] Processed inputs keys: {inputs.keys()}")

            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            print(f"[DEBUG] Model outputs keys: {outputs.keys()}")

            # Extract masks and scores
            masks = outputs.pred_masks.squeeze().cpu().numpy()
            scores = outputs.iou_scores.squeeze().cpu().numpy()
            
            print(f"[DEBUG] Raw masks shape: {masks.shape}")
            print(f"[DEBUG] IOU scores: {scores}")

            # Handle multiple masks if returned
            if len(masks.shape) == 3:
                # Select mask with highest score
                mask_idx = np.argmax(scores)
                mask = masks[mask_idx]
                print(f"[DEBUG] Selected mask index: {mask_idx}, score: {scores[mask_idx]}")
            else:
                mask = masks

            # Resize mask to original image size
            mask = cv2.resize(mask.astype(np.float32), (w, h))

            # Threshold mask
            mask = (mask > 0.5).astype(np.uint8)  # Try 0.5, adjust if needed

            print(f"[DEBUG] Final mask shape: {mask.shape}")
            print(f"[DEBUG] Mask value range: {mask.min()}-{mask.max()}")
            print(f"[DEBUG] Mask coverage: {(mask > 0).mean():.2%}")

            # Save intermediate masks for debugging
            self.save_debug_image(mask * 255, "2a_raw_mask")

            # Add bounding box visualization on mask
            mask_viz = np.stack([mask * 255] * 3, axis=-1)
            cv2.rectangle(mask_viz, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            self.save_debug_image(mask_viz, "2b_mask_with_bbox")

            # Save masked object visualization
            masked_image = image.copy()
            masked_image[mask == 0] = 0
            self.save_debug_image(masked_image, "2c_masked_object")

            return mask

        except Exception as e:
            print(f"[ERROR] Error generating mask: {e}")
            import traceback
            traceback.print_exc()
            return None

    def inpaint_safety_gear(self, image, mask, num_inference_steps, guidance_scale):
        """Inpaint safety gear using Stable Diffusion"""
        try:
            # Ensure image is in RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            image_pil = Image.fromarray(image_rgb).convert("RGB")  # Ensure RGB mode
            
            # Ensure mask is in grayscale format
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")  # Convert to grayscale

            # Resize images for the inpainting model
            size = (640, 640)
            image_pil_resized = image_pil.resize(size)
            mask_pil_resized = mask_pil.resize(size)

            # Save debug images before inpainting
            self.save_debug_image(image_pil_resized, "4_inpaint_input")
            self.save_debug_image(mask_pil_resized, "5_inpaint_mask")

            # Run inpainting using Stable Diffusion
            with torch.inference_mode():
                output = self.inpaint_model(
                    prompt=self.prompt,
                    image=image_pil_resized,
                    mask_image=mask_pil_resized,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

            # Save inpainting output
            self.save_debug_image(output, "6_inpaint_output")

            # Convert output back to OpenCV BGR format
            result = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

            # Resize back to original image size
            result = cv2.resize(result, (image.shape[1], image.shape[0]))

            # Create a 3-channel mask for blending
            mask_3d = np.stack([mask] * 3, axis=-1)

            # Blend the inpainted area with the original image
            final = np.where(mask_3d, result, image)

            # Save final blended result
            self.save_debug_image(final, "7_final_result")

            return final

        except Exception as e:
            print(f"Error in inpainting: {e}")
            import traceback
            traceback.print_exc()
            return image

    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Main processing pipeline"""
        try:
            print("\nStarting safety transform pipeline...")
            
            # Get YOLO detector
            detector = kwargs.get('yolo_detector')
            num_inference_steps = kwargs.get('num_inference_steps', 20)
            guidance_scale = kwargs.get("guidance_scale", 8.0)
            if detector is None:
                raise ValueError("Missing YOLO detector")
            
            # Load models if needed
            if self.sam_model is None or self.inpaint_model is None:
                self.load_models()
            
            # Save input image
            self.save_debug_image(image, "0_input")
            
            # Step 1: Detect person
            bbox = self.get_person_detection(detector, image)
            if bbox is None:
                print("No person detected")
                return image
                
            # Step 2: Generate mask
            mask = self.generate_mask(image, bbox)
            if mask is None:
                print("Failed to generate mask")
                return image
                
            # Step 3: Inpaint safety gear
            result = self.inpaint_safety_gear(image, mask, num_inference_steps, guidance_scale)
            
            print("Pipeline completed")
            return result
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return image