"""
Land Use Classification Module for VanDisha FRA Atlas

This module provides functionality to classify satellite images into 4 categories:
- forest
- farmland  
- water_body
- habitation_soil

Uses a SegFormer model from Hugging Face for semantic segmentation.
"""

import torch
import numpy as np
from PIL import Image
import io
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from typing import Dict, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LandUseClassifier:
    """
    Land Use Classification service using SegFormer model.
    
    This class handles loading the pre-trained model, processing satellite images,
    and mapping the model's output to the required 4 categories.
    """
    
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        """
        Initialize the Land Use Classifier.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Category names for the 4 output classes
        self.category_names = {
            0: "forest",
            1: "farmland", 
            2: "water_body",
            3: "habitation_soil"
        }
        
        logger.info(f"Initializing LandUseClassifier with device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained SegFormer model and processor."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess raw image bytes for model inference.
        
        Args:
            image_bytes (bytes): Raw image data
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess using the model's processor
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def run_inference(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run model inference on preprocessed inputs.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Preprocessed image inputs
            
        Returns:
            torch.Tensor: Raw model predictions (logits)
        """
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Upsample logits to original image size
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=inputs['pixel_values'].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                # Get predicted class for each pixel
                predicted_mask = upsampled_logits.argmax(dim=1).squeeze(0)
                
            return predicted_mask
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def map_labels(self, predicted_mask: torch.Tensor) -> np.ndarray:
        """
        Map the model's predictions to our 4 categories.
        
        This is a simplified mapping for demonstration purposes.
        In practice, you would use a proper land cover model trained on satellite imagery.
        
        Args:
            predicted_mask (torch.Tensor): Raw model predictions
            
        Returns:
            np.ndarray: Mapped mask with 4 categories (0-3)
        """
        try:
            # Convert to numpy for easier manipulation
            mask_np = predicted_mask.cpu().numpy()
            
            # Create mapped mask - simplified mapping for demonstration
            mapped_mask = np.zeros_like(mask_np, dtype=np.uint8)
            
            # Simple heuristic mapping based on common ADE20K classes
            # Forest: trees, grass, mountains, plants
            forest_classes = [4, 9, 16, 17, 29, 72, 77, 95, 96, 115, 116, 133, 134]  # tree, grass, mountain, plant, field, palm, shrub, etc.
            
            # Water: water bodies
            water_classes = [21, 26, 56, 60, 98, 109, 117, 135]  # water, sea, pool, river, etc.
            
            # Farmland: fields and agricultural areas
            farmland_classes = [29, 104, 122]  # field
            
            # Habitation/Soil: buildings, roads, bare areas
            habitation_classes = [0, 1, 3, 6, 11, 13, 25, 34, 46, 83, 94, 100, 101, 102, 103, 105, 106, 113, 114, 119, 120, 121, 123, 124, 131, 132]  # wall, building, floor, road, sidewalk, earth, house, rock, sand, city, etc.
            
            # Apply mappings
            for class_id in forest_classes:
                mapped_mask[mask_np == class_id] = 0  # forest
            
            for class_id in water_classes:
                mapped_mask[mask_np == class_id] = 2  # water_body
            
            for class_id in farmland_classes:
                mapped_mask[mask_np == class_id] = 1  # farmland
            
            for class_id in habitation_classes:
                mapped_mask[mask_np == class_id] = 3  # habitation_soil
            
            # Default remaining classes to habitation_soil
            mapped_mask[mapped_mask == 0] = 3  # habitation_soil
            
            return mapped_mask
            
        except Exception as e:
            logger.error(f"Error mapping labels: {e}")
            raise
    
    def classify_image(self, image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Main classification function that processes a satellite image.
        
        Args:
            image_bytes (bytes): Raw satellite image data
            
        Returns:
            Tuple[np.ndarray, Dict[str, int]]: 
                - Mapped mask with 4 categories (0-3)
                - Dictionary with category counts
        """
        try:
            logger.info("Starting image classification")
            
            # Step 1: Preprocess image
            inputs = self.preprocess_image(image_bytes)
            
            # Step 2: Run inference
            predicted_mask = self.run_inference(inputs)
            
            # Step 3: Map labels to 4 categories
            mapped_mask = self.map_labels(predicted_mask)
            
            # Step 4: Calculate category counts
            category_counts = self._calculate_category_counts(mapped_mask)
            
            logger.info("Image classification completed successfully")
            return mapped_mask, category_counts
            
        except Exception as e:
            logger.error(f"Error in classify_image: {e}")
            raise
    
    def _calculate_category_counts(self, mapped_mask: np.ndarray) -> Dict[str, int]:
        """
        Calculate pixel counts for each category.
        
        Args:
            mapped_mask (np.ndarray): Mapped mask with 4 categories
            
        Returns:
            Dict[str, int]: Category counts
        """
        unique, counts = np.unique(mapped_mask, return_counts=True)
        category_counts = {}
        
        for category_id, count in zip(unique, counts):
            category_name = self.category_names[category_id]
            category_counts[category_name] = int(count)
        
        return category_counts
    
    def get_category_info(self) -> Dict[str, int]:
        """
        Get information about the 4 categories.
        
        Returns:
            Dict[str, int]: Category ID to name mapping
        """
        return {str(k): v for k, v in self.category_names.items()}


# Convenience function for easy usage
def classify_land_use(image_bytes: bytes, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512") -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convenience function to classify land use from image bytes.
    
    Args:
        image_bytes (bytes): Raw satellite image data
        model_name (str): Hugging Face model identifier
        
    Returns:
        Tuple[np.ndarray, Dict[str, int]]: 
            - Mapped mask with 4 categories (0-3)
            - Dictionary with category counts
    """
    classifier = LandUseClassifier(model_name)
    return classifier.classify_image(image_bytes)


if __name__ == "__main__":
    # Example usage
    print("Land Use Classification Module for VanDisha FRA Atlas")
    print("Categories:", LandUseClassifier().get_category_info())