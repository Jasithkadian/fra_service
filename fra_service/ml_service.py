"""
Land Use Classification Module for VanDisha FRA Atlas

This module provides a single, efficient function to classify satellite images
into 4 categories using a SegFormer model from Hugging Face.
- forest
- farmland
- water_body
- habitation_soil
"""

import io
import logging
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_model_and_processor() -> Tuple[SegformerForSemanticSegmentation, SegformerImageProcessor, torch.device]:
    """
    Lazily loads and caches the model and processor.
    """
    # ✅ Use ESA WorldCover model (not ADE20K)
    model_name = "ESA/WorldCover_SegFormer"  # <-- replace with the correct HuggingFace repo ID if different
    logger.info(f"Loading model '{model_name}' for the first time...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    logger.info(f"✅ Model loaded successfully on device: {device}")
    return model, processor, device

def predict_land_use(image_bytes: bytes) -> Dict[str, float]:
    """
    Predicts land use percentages for a given satellite image.

    Args:
        image_bytes (bytes): The raw byte data of a satellite image.

    Returns:
        Dict[str, float]: A dictionary with the percentage breakdown of land use.
    """
    if not image_bytes:
        logger.error("predict_land_use called with empty image_bytes")
        raise ValueError("Input image_bytes cannot be empty.")

    try:
        # 1. Load model + processor
        model, processor, device = _get_model_and_processor()

        # 2. Preprocess
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

        # 4. Count categories (WorldCover already uses 0–3 classes mapped to your needs)
        total_pixels = predicted_mask.size
        if total_pixels == 0:
            return {"forest": 0.0, "farmland": 0.0, "water_body": 0.0, "habitation_soil": 0.0}

        counts = {
            "forest": np.count_nonzero(predicted_mask == 0),
            "farmland": np.count_nonzero(predicted_mask == 1),
            "water_body": np.count_nonzero(predicted_mask == 2),
            "habitation_soil": np.count_nonzero(predicted_mask == 3),
        }

        percentages = {k: round((v / total_pixels) * 100, 2) for k, v in counts.items()}
        logger.info(f"Classification successful: {percentages}")
        return percentages

    except Exception as e:
        logger.error(f"An error occurred during land use prediction: {e}", exc_info=True)
        return {"error": "Failed to classify image due to an internal error."}
