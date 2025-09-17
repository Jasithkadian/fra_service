from __future__ import annotations

import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
)


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


class LandUseClassifier:
    """Classifies satellite imagery into 4 land use categories.

    Categories (IDs → names):
      0 → "forest"
      1 → "farmland"
      2 → "water_body"
      3 → "habitation_soil"
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        # Mapping from ESA WorldCover's 11 classes (indices 0..10) to project categories (0..3)
        # forest (0): Tree cover, Shrubland, Grassland
        # farmland (1): Cropland
        # water_body (2): Permanent water bodies, Herbaceous wetland, Mangroves, Moss and lichen, Snow and ice
        # habitation_soil (3): Built-up, Bare / sparse vegetation
        self.esa_to_project_category: Dict[int, int] = {
            0: 0,  # Tree cover → forest
            1: 0,  # Shrubland → forest
            2: 0,  # Grassland → forest
            3: 1,  # Cropland → farmland
            4: 3,  # Built-up → habitation_soil
            5: 3,  # Bare / sparse vegetation → habitation_soil
            6: 2,  # Snow and ice → water_body
            7: 2,  # Permanent water bodies → water_body
            8: 2,  # Herbaceous wetland → water_body
            9: 2,  # Mangroves → water_body
            10: 2,  # Moss and lichen → water_body
        }

        self.project_categories: Dict[int, str] = {
            0: "forest",
            1: "farmland",
            2: "water_body",
            3: "habitation_soil",
        }

    def classify_image(self, image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, int]]:
        """Classify raw image bytes into the 4 categories.

        Returns a tuple of (mapped_mask, category_counts).
        - mapped_mask: 2D ndarray of shape (H, W) with values in {0,1,2,3}
        - category_counts: dict mapping category name → pixel count
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch, num_labels, h, w]

        # Upsample logits to the original image size
        upsampled_logits = F.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )

        # Predicted class per pixel (assumes model outputs ESA-like 11 classes indexed 0..10)
        predicted_mask = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        # Map ESA classes to project categories
        mapped_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
        for esa_id, proj_id in self.esa_to_project_category.items():
            mapped_mask[predicted_mask == esa_id] = proj_id

        # Count pixels per project category
        category_counts: Dict[str, int] = {}
        for proj_id, name in self.project_categories.items():
            category_counts[name] = int((mapped_mask == proj_id).sum())

        return mapped_mask, category_counts

    # Optional helper for examples
    def get_category_info(self) -> Dict[int, str]:
        return dict(self.project_categories)


_GLOBAL_CLASSIFIER: LandUseClassifier | None = None


def _get_or_create_classifier() -> LandUseClassifier:
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        _GLOBAL_CLASSIFIER = LandUseClassifier()
    return _GLOBAL_CLASSIFIER


def predict_land_use(image_bytes: bytes) -> Dict[str, float]:
    """Predict land use percentages per category from raw image bytes.

    Returns a flat dict like:
      {"forest": 61.23, "farmland": 20.45, "water_body": 5.67, "habitation_soil": 12.65}
    """
    classifier = _get_or_create_classifier()
    _, counts = classifier.classify_image(image_bytes)

    total_pixels = sum(counts.values()) or 1
    percentages = {
        name: round((count / total_pixels) * 100.0, 2)
        for name, count in counts.items()
    }
    return percentages


# Backwards-compatible alias used by some examples
def classify_land_use(image_bytes: bytes) -> Dict[str, float]:
    return predict_land_use(image_bytes)

