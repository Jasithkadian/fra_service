# VanDisha FRA Atlas - Land Use Classification Module

This service classifies satellite images into 4 land use categories using a SegFormer model from Hugging Face.

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

## Quick Start (API)

1. Install dependencies: `pip install -r requirements.txt`
2. Start server: `uvicorn main:app --reload`
3. POST an image to `/classify/` (multipart form key `file`)
   Response:
   ```json
   {"forest": 61.23, "farmland": 20.45, "water_body": 5.67, "habitation_soil": 12.65}
   ```

## Categories

The model maps ESA WorldCover's 10 classes to 4 categories:

- **0: forest** - Tree cover, shrubland, grassland
- **1: farmland** - Cropland, built-up agricultural areas  
- **2: water_body** - Water bodies, wetlands, snow/ice
- **3: habitation_soil** - Bare areas, moss/lichen

## ML Service Reference

`fra_service/ml_service.py` exposes a single function:

```python
from fra_service.ml_service import predict_land_use

with open('image.jpg', 'rb') as f:
    result = predict_land_use(f.read())
    # {'forest': 61.23, 'farmland': 20.45, 'water_body': 5.67, 'habitation_soil': 12.65}
```

## Output Format

The API and `predict_land_use` return a flat JSON/dict with four keys and float percentages rounded to two decimals.

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
