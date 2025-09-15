# VanDisha FRA Atlas - Land Use Classification Module

This module provides functionality to classify satellite images into 4 land use categories using the ESA/WorldCover-SegFormer model from Hugging Face.

## Features

- **4-Category Classification**: Maps satellite images to forest, farmland, water_body, and habitation_soil
- **ESA/WorldCover-SegFormer**: Uses state-of-the-art semantic segmentation model
- **Modular Design**: Easy to extend for JSON aggregation and percentage calculations
- **Raw Bytes Input**: Accepts satellite images as raw byte data
- **GPU Support**: Automatically uses CUDA if available

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from fra_service.ml_service import LandUseClassifier

# Initialize classifier
classifier = LandUseClassifier()

# Load your satellite image
with open('satellite_image.jpg', 'rb') as f:
    image_bytes = f.read()

# Classify the image
mapped_mask, category_counts = classifier.classify_image(image_bytes)

print("Category counts:", category_counts)
print("Mapped mask shape:", mapped_mask.shape)
```

## Categories

The model maps ESA WorldCover's 10 classes to 4 categories:

- **0: forest** - Tree cover, shrubland, grassland
- **1: farmland** - Cropland, built-up agricultural areas  
- **2: water_body** - Water bodies, wetlands, snow/ice
- **3: habitation_soil** - Bare areas, moss/lichen

## API Reference

### LandUseClassifier

Main class for land use classification.

#### Methods

- `classify_image(image_bytes)`: Classify a satellite image
- `get_category_info()`: Get category ID to name mapping
- `preprocess_image(image_bytes)`: Preprocess raw image bytes
- `run_inference(inputs)`: Run model inference
- `map_labels(predicted_mask)`: Map 10 classes to 4 categories

### Convenience Function

```python
from fra_service.ml_service import classify_land_use

mapped_mask, category_counts = classify_land_use(image_bytes)
```

## Output Format

The `classify_image` method returns:

1. **mapped_mask**: NumPy array with 4 categories (0-3)
2. **category_counts**: Dictionary with pixel counts per category

Example output:
```python
{
    'forest': 125000,
    'farmland': 89000, 
    'water_body': 15000,
    'habitation_soil': 21000
}
```

## Next Steps

This module provides the foundation for Person 1's deliverable. Person 2 can easily add:

- Percentage calculations from category_counts
- JSON serialization of results
- Additional data processing and formatting

## Dependencies

- torch
- torchvision  
- transformers
- timm
- Pillow
- scikit-image
- numpy
