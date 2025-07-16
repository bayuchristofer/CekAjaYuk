#!/usr/bin/env python3
"""
Debug OCR extraction issues
"""
import requests
import base64
import json
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add test text
    text_lines = [
        "LOWONGAN KERJA",
        "PT. TEKNOLOGI MAJU",
        "Posisi: Software Developer",
        "Gaji: Rp 8.000.000 - Rp 12.000.000",
        "Lokasi: Jakarta",
        "Kontak: hr@teknologimaju.com",
        "",
        "Syarat:",
        "- S1 Teknik Informatika",
        "- Pengalaman min 2 tahun",
        "- Menguasai Python, JavaScript"
    ]
    
    y_position = 30
    for line in text_lines:
        draw.text((50, y_position), line, fill='black', font=font)
        y_position += 35
    
    return img

def test_ocr_endpoint():
    """Test OCR endpoint with simple image"""
    print("🧪 DEBUGGING OCR EXTRACTION")
    print("=" * 50)
    
    # Create test image
    print("📷 Creating test image...")
    test_img = create_test_image()
    
    # Save test image for reference
    test_img.save('test_image.png')
    print("💾 Test image saved as 'test_image.png'")
    
    # Convert to base64
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Test with extract-text endpoint
    print("\n🔍 Testing /api/extract-text endpoint...")
    
    try:
        # Create form data
        files = {'file': ('test.png', img_buffer.getvalue(), 'image/png')}
        
        # Reset buffer position
        img_buffer.seek(0)
        files = {'file': ('test.png', img_buffer, 'image/png')}
        
        response = requests.post(
            'http://localhost:5001/api/extract-text',
            files=files,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ OCR Success!")
            print(f"Status: {data.get('status')}")
            print(f"Message: {data.get('message')}")
            
            if 'data' in data:
                ocr_data = data['data']
                print(f"\n📝 Extracted Text:")
                print("-" * 40)
                print(ocr_data.get('text', ocr_data.get('extracted_text', 'No text found')))
                print("-" * 40)
                
                print(f"\n📊 OCR Details:")
                print(f"   Confidence: {ocr_data.get('confidence', 'N/A')}")
                print(f"   Method: {ocr_data.get('method', 'N/A')}")
                print(f"   Characters: {ocr_data.get('char_count', 'N/A')}")
                print(f"   Processing Time: {ocr_data.get('processing_time', 'N/A')}s")
                print(f"   OCR Version: {ocr_data.get('ocr_version', 'N/A')}")
            else:
                print("❌ No data in response")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing OCR: {e}")
        import traceback
        traceback.print_exc()

def test_with_json_payload():
    """Test with JSON payload (like frontend)"""
    print("\n🔍 Testing with JSON payload (like frontend)...")
    
    # Create test image
    test_img = create_test_image()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    try:
        payload = {
            'image': f"data:image/png;base64,{img_base64}"
        }
        
        response = requests.post(
            'http://localhost:5001/api/extract-text',
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ JSON OCR Success!")
            print(f"Extracted Text: {data.get('data', {}).get('text', 'No text')[:100]}...")
        else:
            print(f"❌ JSON request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error with JSON payload: {e}")

def main():
    """Main debug function"""
    print("🔧 OCR DEBUGGING SUITE")
    print("🎯 Testing OCR extraction with simple image")
    print("=" * 60)
    
    # Test OCR endpoint
    test_ocr_endpoint()
    
    # Test with JSON payload
    test_with_json_payload()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUGGING COMPLETED")
    print("💡 Check the logs above for any errors")
    print("📷 Test image saved as 'test_image.png' for reference")

if __name__ == "__main__":
    main()
