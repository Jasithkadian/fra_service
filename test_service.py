"""
Test script for the FRA Land Classification Service
Run this after starting your FastAPI server to test functionality
"""

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "path/to/your/test_image.jpg"  # Update this path

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def validate_percentages(response_json: dict) -> bool:
    """Validate the expected JSON schema: flat dict with 4 float percentages."""
    expected_keys = {"forest", "farmland", "water_body", "habitation_soil"}
    if not isinstance(response_json, dict):
        return False
    if set(response_json.keys()) != expected_keys:
        return False
    try:
        for k in expected_keys:
            val = float(response_json[k])
            if val < 0 or val > 100:
                return False
        total = sum(float(response_json[k]) for k in expected_keys)
        # Allow minor rounding variance
        return 99.0 <= total <= 101.0
    except Exception:
        return False

def test_classification(image_path: str):
    """Test image classification"""
    print(f"\n🖼️ Testing image classification with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        print("Please update TEST_IMAGE_PATH in this script")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            
            # Test single classification endpoint
            print("Testing /classify/ endpoint...")
            response = requests.post(f"{BASE_URL}/classify/", files=files)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                ok = validate_percentages(result)
                if not ok:
                    print("❌ Response schema invalid (expected 4 float percentages summing ~100)")
                return ok
            else:
                print(f"❌ Classification failed: {response.text}")
                
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
    
    return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🚫 Testing error handling...")
    
    # Test with no file
    try:
        response = requests.post(f"{BASE_URL}/classify/")
        print(f"No file test - Status: {response.status_code} (should be 422)")
    except Exception as e:
        print(f"No file test error: {e}")
    
    # Test with non-image file
    try:
        files = {'file': ('test.txt', 'This is not an image', 'text/plain')}
        response = requests.post(f"{BASE_URL}/classify/", files=files)
        print(f"Non-image test - Status: {response.status_code} (should be 400)")
        if response.status_code == 400:
            print(f"Error message: {response.json()}")
    except Exception as e:
        print(f"Non-image test error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting FRA Land Classification Service Tests")
    print("=" * 50)
    
    # Basic endpoint tests
    health_ok = test_health_check()
    root_ok = test_root_endpoint()
    categories_ok = True  # categories endpoint removed
    
    # Image classification test
    classification_ok = False
    if Path(TEST_IMAGE_PATH).exists():
        classification_ok = test_classification(TEST_IMAGE_PATH)
    else:
        print(f"\n⚠️ Skipping image classification test - please set TEST_IMAGE_PATH")
        print(f"Current path: {TEST_IMAGE_PATH}")
    
    # Error handling tests
    test_error_handling()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"✅ Root Endpoint: {'PASS' if root_ok else 'FAIL'}")
    print(f"✅ Categories: {'SKIPPED' if categories_ok else 'FAIL'}")
    print(f"✅ Classification: {'PASS' if classification_ok else 'SKIPPED/FAIL'}")
    
    if all([health_ok, root_ok, categories_ok]):
        print("\n🎉 Basic service functionality is working!")
        if classification_ok:
            print("🎉 Image classification is working perfectly!")
        else:
            print("⚠️ Set up test image to verify classification functionality")
    else:
        print("\n❌ Some basic tests failed. Check your service setup.")

if __name__ == "__main__":
    main()
