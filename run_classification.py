#!/usr/bin/env python3
"""
Simple command-line script to run Land Use Classification.
Usage: python run_classification.py [image_path]
"""

import sys
import os
from fra_service.ml_service import LandUseClassifier

def main():
    print("VanDisha FRA Atlas - Land Use Classification")
    print("=" * 45)
    
    # Initialize classifier
    print("Loading model...")
    classifier = LandUseClassifier()
    print("✅ Model loaded successfully!")
    
    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"\nProcessing image: {image_path}")
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Classify
                mapped_mask, category_counts = classifier.classify_image(image_bytes)
                
                # Results
                print("\n📊 Classification Results:")
                print("-" * 30)
                total_pixels = mapped_mask.size
                for category, count in category_counts.items():
                    percentage = (count / total_pixels) * 100
                    print(f"  {category}: {count:,} pixels ({percentage:.2f}%)")
                
                print(f"\n📐 Mask shape: {mapped_mask.shape}")
                print(f"🔢 Unique categories: {sorted(category_counts.keys())}")
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
        else:
            print(f"❌ Image file not found: {image_path}")
    else:
        # No image provided, show usage
        print("\n📖 Usage:")
        print("  python run_classification.py <image_path>")
        print("\n📝 Example:")
        print("  python run_classification.py satellite_image.jpg")
        print("\n💡 Categories:")
        for cat_id, cat_name in classifier.get_category_info().items():
            print(f"  {cat_id}: {cat_name}")

if __name__ == "__main__":
    main()
