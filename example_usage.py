"""
Example usage of the Land Use Classification module for VanDisha FRA Atlas.

This script demonstrates how to use the ml_service.py module to classify
satellite images into 4 categories: forest, farmland, water_body, habitation_soil.
"""

import os
import numpy as np
from fra_service.ml_service import LandUseClassifier, classify_land_use

def main():
    """Example usage of the Land Use Classification service."""
    
    print("VanDisha FRA Atlas - Land Use Classification Example")
    print("=" * 50)
    
    # Initialize the classifier
    print("Initializing classifier...")
    classifier = LandUseClassifier()
    
    # Display category information
    print("\nCategory mapping:")
    category_info = classifier.get_category_info()
    for category_id, category_name in category_info.items():
        print(f"  {category_id}: {category_name}")
    
    # Example with a sample image (you would replace this with actual image bytes)
    print("\nTo use with actual satellite image:")
    print("1. Load your satellite image as bytes")
    print("2. Call classifier.classify_image(image_bytes)")
    print("3. Get the mapped mask and category counts")
    
    # Create a simple test image to demonstrate
    print("\nCreating a test image to demonstrate...")
    from PIL import Image
    import io
    
    # Create a simple test image
    test_img = Image.new('RGB', (256, 256), color='green')
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    image_bytes = img_bytes.getvalue()
    
    try:
        # Classify the test image
        print("Running classification on test image...")
        mapped_mask, category_counts = classifier.classify_image(image_bytes)
        
        # Display results
        print("\nTest Results:")
        print(f"  Mapped mask shape: {mapped_mask.shape}")
        print(f"  Unique categories: {np.unique(mapped_mask)}")
        print("  Category counts:")
        for category, count in category_counts.items():
            total_pixels = mapped_mask.size
            percentage = (count / total_pixels) * 100
            print(f"    {category}: {count} pixels ({percentage:.2f}%)")
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    # Example code structure:
    example_code = '''
    # Load your satellite image
    with open('satellite_image.jpg', 'rb') as f:
        image_bytes = f.read()
    
    # Classify the image
    mapped_mask, category_counts = classifier.classify_image(image_bytes)
    
    # Results
    print("Category counts:", category_counts)
    print("Mapped mask shape:", mapped_mask.shape)
    print("Unique categories in mask:", np.unique(mapped_mask))
    '''
    
    print("\nExample code for your own images:")
    print(example_code)
    
    print("\nFor the second person to add JSON aggregation:")
    print("- The mapped_mask contains the 4-category segmentation")
    print("- The category_counts dictionary has pixel counts per category")
    print("- You can calculate percentages: count/total_pixels * 100")
    print("- Return as JSON with category names and percentages")

if __name__ == "__main__":
    main()
