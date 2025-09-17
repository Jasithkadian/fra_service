#!/usr/bin/env python3
"""
Simple command-line script to run Land Use Classification.
Usage: python run_classification.py [image_path]
"""

import sys
import os
from fra_service.ml_service import predict_land_use

def main():
    print("VanDisha FRA Atlas - Land Use Classification")
    print("=" * 45)

    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"\nProcessing image: {image_path}")
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Classify using predict_land_use
                result = predict_land_use(image_bytes)
                
                # Results
                print("\nClassification Results:")
                print("-" * 30)
                for category, percentage in result.items():
                    print(f"  {category}: {percentage:.2f}%")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            print(f"Image file not found: {image_path}")
    else:
        # No image provided, show usage
        print("\nUsage:")
        print("  python run_classification.py <image_path>")
        print("\nExample:")
        print("  python run_classification.py satellite_image.jpg")
        print("\nCategories:")
        print("  forest, farmland, water_body, habitation_soil")

if __name__ == "__main__":
    main()
