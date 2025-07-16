"""
CekAjaYuk Dataset Collection Helper
Script untuk membantu pengumpulan dan validasi dataset
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

class DatasetCollector:
    """Helper class untuk pengumpulan dataset"""
    
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = Path(dataset_dir)
        self.genuine_dir = self.dataset_dir / 'genuine'
        self.fake_dir = self.dataset_dir / 'fake'
        self.stats_file = self.dataset_dir / 'collection_stats.json'
        
        # Create directories
        self.genuine_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def validate_image(self, image_path):
        """Validate if image is suitable for dataset"""
        try:
            # Check file extension
            if Path(image_path).suffix.lower() not in self.supported_formats:
                return False, "Unsupported format"
            
            # Try to open image
            with Image.open(image_path) as img:
                # Check image size
                width, height = img.size
                if width < 200 or height < 200:
                    return False, "Image too small (min 200x200)"
                
                if width > 5000 or height > 5000:
                    return False, "Image too large (max 5000x5000)"
                
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:  # 10MB
                    return False, "File too large (max 10MB)"
                
                # Check if image is readable
                img.verify()
                
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def add_image(self, image_path, category, custom_name=None):
        """Add image to dataset with validation"""
        if category not in ['genuine', 'fake']:
            return False, "Category must be 'genuine' or 'fake'"
        
        # Validate image
        is_valid, message = self.validate_image(image_path)
        if not is_valid:
            return False, f"Validation failed: {message}"
        
        # Determine destination
        dest_dir = self.genuine_dir if category == 'genuine' else self.fake_dir
        
        # Generate filename
        if custom_name:
            filename = custom_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = Path(image_path).stem
            extension = Path(image_path).suffix
            filename = f"{category}_{timestamp}_{original_name}{extension}"
        
        dest_path = dest_dir / filename
        
        # Copy file
        try:
            shutil.copy2(image_path, dest_path)
            self.update_stats()
            return True, f"Added to {dest_path}"
        except Exception as e:
            return False, f"Copy failed: {str(e)}"
    
    def batch_add_images(self, source_dir, category):
        """Add multiple images from directory"""
        source_path = Path(source_dir)
        if not source_path.exists():
            return False, "Source directory not found"
        
        results = []
        for image_file in source_path.iterdir():
            if image_file.suffix.lower() in self.supported_formats:
                success, message = self.add_image(str(image_file), category)
                results.append({
                    'file': image_file.name,
                    'success': success,
                    'message': message
                })
        
        return True, results
    
    def get_stats(self):
        """Get dataset statistics"""
        genuine_count = len(list(self.genuine_dir.glob('*')))
        fake_count = len(list(self.fake_dir.glob('*')))
        
        stats = {
            'genuine_count': genuine_count,
            'fake_count': fake_count,
            'total_count': genuine_count + fake_count,
            'balance_ratio': genuine_count / max(fake_count, 1),
            'last_updated': datetime.now().isoformat()
        }
        
        return stats
    
    def update_stats(self):
        """Update statistics file"""
        stats = self.get_stats()
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def analyze_dataset_quality(self):
        """Analyze dataset quality and provide recommendations"""
        stats = self.get_stats()
        
        analysis = {
            'total_images': stats['total_count'],
            'class_balance': 'Good' if 0.7 <= stats['balance_ratio'] <= 1.3 else 'Imbalanced',
            'size_adequacy': 'Sufficient' if stats['total_count'] >= 1000 else 'Insufficient',
            'recommendations': []
        }
        
        # Recommendations
        if stats['total_count'] < 500:
            analysis['recommendations'].append("Collect more images (target: 500+ per class)")
        
        if stats['balance_ratio'] < 0.7:
            analysis['recommendations'].append("Add more genuine job posting images")
        elif stats['balance_ratio'] > 1.3:
            analysis['recommendations'].append("Add more fake job posting images")
        
        if stats['genuine_count'] < 300:
            analysis['recommendations'].append("Target 300+ genuine images for better accuracy")
        
        if stats['fake_count'] < 300:
            analysis['recommendations'].append("Target 300+ fake images for better detection")
        
        return analysis
    
    def create_sample_structure(self):
        """Create sample directory structure with examples"""
        sample_dir = self.dataset_dir / 'samples'
        sample_dir.mkdir(exist_ok=True)
        
        # Create example structure
        structure = {
            'genuine_examples': [
                'jobstreet_software_engineer.jpg',
                'linkedin_marketing_manager.jpg',
                'company_website_analyst.jpg',
                'government_civil_servant.jpg',
                'university_lecturer.jpg'
            ],
            'fake_examples': [
                'mlm_easy_money.jpg',
                'whatsapp_only_contact.jpg',
                'unrealistic_salary.jpg',
                'urgent_immediate_start.jpg',
                'no_experience_required.jpg'
            ]
        }
        
        # Save structure info
        with open(sample_dir / 'structure_guide.json', 'w') as f:
            json.dump(structure, f, indent=2)
        
        return structure

def main():
    """Main function for interactive dataset collection"""
    print("=" * 60)
    print("CekAjaYuk Dataset Collection Helper")
    print("=" * 60)
    
    collector = DatasetCollector()
    
    while True:
        print("\nOptions:")
        print("1. Add single image")
        print("2. Batch add images from directory")
        print("3. View dataset statistics")
        print("4. Analyze dataset quality")
        print("5. Create sample structure")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Add single image
            image_path = input("Enter image path: ").strip()
            category = input("Enter category (genuine/fake): ").strip().lower()
            
            if category not in ['genuine', 'fake']:
                print("❌ Invalid category. Use 'genuine' or 'fake'")
                continue
            
            success, message = collector.add_image(image_path, category)
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
        
        elif choice == '2':
            # Batch add images
            source_dir = input("Enter source directory path: ").strip()
            category = input("Enter category (genuine/fake): ").strip().lower()
            
            if category not in ['genuine', 'fake']:
                print("❌ Invalid category. Use 'genuine' or 'fake'")
                continue
            
            success, results = collector.batch_add_images(source_dir, category)
            if success:
                successful = sum(1 for r in results if r['success'])
                print(f"✅ Added {successful}/{len(results)} images")
                
                # Show failed ones
                failed = [r for r in results if not r['success']]
                if failed:
                    print("\nFailed images:")
                    for f in failed[:5]:  # Show first 5 failures
                        print(f"  ❌ {f['file']}: {f['message']}")
            else:
                print(f"❌ {results}")
        
        elif choice == '3':
            # View statistics
            stats = collector.get_stats()
            print(f"\n📊 Dataset Statistics:")
            print(f"  Genuine images: {stats['genuine_count']}")
            print(f"  Fake images: {stats['fake_count']}")
            print(f"  Total images: {stats['total_count']}")
            print(f"  Balance ratio: {stats['balance_ratio']:.2f}")
            print(f"  Last updated: {stats['last_updated']}")
        
        elif choice == '4':
            # Analyze quality
            analysis = collector.analyze_dataset_quality()
            print(f"\n🔍 Dataset Quality Analysis:")
            print(f"  Total images: {analysis['total_images']}")
            print(f"  Class balance: {analysis['class_balance']}")
            print(f"  Size adequacy: {analysis['size_adequacy']}")
            
            if analysis['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"  • {rec}")
            else:
                print(f"\n✅ Dataset quality looks good!")
        
        elif choice == '5':
            # Create sample structure
            structure = collector.create_sample_structure()
            print(f"\n📁 Sample structure created!")
            print(f"  Check dataset/samples/ for examples")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
