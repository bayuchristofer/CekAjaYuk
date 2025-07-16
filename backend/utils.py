"""
Utility functions for CekAjaYuk backend
"""

import os
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
from textblob import TextBlob
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_image(image_path):
    """Validate if the uploaded file is a valid image"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {image_path}: {e}")
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for ML/DL models"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, target_size)
        
        # Normalize pixel values
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_normalized
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_traditional_features(image_path):
    """Extract traditional computer vision features (12 features to match trained model)"""
    try:
        image = preprocess_image(image_path)
        if image is None:
            return None

        features = []

        # Convert to grayscale for some features
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # 1. Color features (6 features)
        features.extend(np.mean(image, axis=(0, 1)))  # Mean RGB (3)
        features.extend(np.std(image, axis=(0, 1)))   # Std RGB (3)

        # 2. Brightness and contrast (2 features)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features.extend([brightness, contrast])

        # 3. Edge features (1 feature)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)

        # 4. Texture variance (1 feature)
        texture_variance = np.var(gray)
        features.append(texture_variance)

        # 5. Color diversity (1 feature)
        # Calculate color diversity as the number of unique colors normalized
        unique_colors = len(np.unique(image.reshape(-1, image.shape[-1]), axis=0))
        max_possible_colors = min(256**3, image.shape[0] * image.shape[1])
        color_diversity = unique_colors / max_possible_colors
        features.append(color_diversity)

        # 6. Layout symmetry (1 feature)
        # Simple symmetry measure: compare left and right halves
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_resized = left_half[:, :min_width]
        right_resized = right_half[:, :min_width]
        symmetry = 1.0 - np.mean(np.abs(left_resized.astype(float) - right_resized.astype(float))) / 255.0
        features.append(symmetry)

        # Ensure we have exactly 12 features
        features_array = np.array(features)
        if len(features_array) != 12:
            logger.warning(f"Expected 12 features, got {len(features_array)}. Padding or truncating.")
            if len(features_array) < 12:
                # Pad with zeros
                features_array = np.pad(features_array, (0, 12 - len(features_array)), 'constant')
            else:
                # Truncate
                features_array = features_array[:12]

        return features_array

    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None

def extract_text_ocr(image_path, config=None):
    """Extract text using Tesseract OCR with improved preprocessing"""
    try:
        # Load image
        image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL to OpenCV format for preprocessing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess image for better OCR
        processed_image = preprocess_for_ocr(cv_image)

        # Convert back to PIL
        processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Try multiple OCR configurations
        configs = [
            r'--oem 3 --psm 6 -l ind+eng',  # Indonesian + English, auto page segmentation
            r'--oem 3 --psm 3 -l ind+eng',  # Fully automatic page segmentation
            r'--oem 3 --psm 11 -l ind+eng', # Sparse text
            r'--oem 3 --psm 13 -l ind+eng', # Raw line
        ]

        if config:
            configs.insert(0, config)

        best_text = ""
        best_confidence = 0

        for cfg in configs:
            try:
                # Extract text with confidence
                data = pytesseract.image_to_data(processed_pil, config=cfg, output_type=pytesseract.Output.DICT)

                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                # Extract text
                text = pytesseract.image_to_string(processed_pil, config=cfg)
                text = clean_extracted_text(text)

                # Keep best result
                if avg_confidence > best_confidence and len(text.strip()) > len(best_text.strip()):
                    best_text = text
                    best_confidence = avg_confidence

            except Exception as e:
                logger.warning(f"OCR config {cfg} failed: {e}")
                continue

        # Fallback to original image if preprocessing didn't help
        if not best_text.strip():
            try:
                best_text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6 -l ind+eng')
                best_text = clean_extracted_text(best_text)
            except Exception as e:
                logger.error(f"Fallback OCR failed: {e}")

        logger.info(f"Extracted {len(best_text)} characters from {image_path} (confidence: {best_confidence:.1f}%)")
        return best_text

    except Exception as e:
        logger.error(f"Error extracting text from {image_path}: {e}")
        return ""

def preprocess_for_ocr(image):
    """Preprocess image for better OCR results"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert back to BGR for consistency
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        return result

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return image

def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)

    # Remove special characters that are likely OCR errors
    text = re.sub(r'[^\w\s\n\-.,!?()@#$%&*+=<>:;/\\]', '', text)

    # Fix common OCR mistakes for Indonesian text
    replacements = {
        'l': 'I',  # lowercase l to uppercase I in certain contexts
        '0': 'O',  # zero to O in certain contexts
        '5': 'S',  # 5 to S in certain contexts
        '1': 'l',  # 1 to l in certain contexts
    }

    # Apply replacements carefully (only in specific contexts)
    # This is a simplified approach - in production, you'd want more sophisticated rules

    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def analyze_text_patterns(text):
    """Analyze text for suspicious patterns"""
    if not text or len(text.strip()) < 10:
        return {
            'score': 0.2,
            'suspicious_patterns': [],
            'positive_indicators': [],
            'text_quality': 'poor'
        }
    
    text_lower = text.lower()
    
    # Suspicious patterns (indicating fake jobs)
    suspicious_patterns = [
        (r'urgent.*hiring', 'Urgent hiring claims'),
        (r'immediate.*start', 'Immediate start promises'),
        (r'no.*experience.*required', 'No experience required'),
        (r'work.*from.*home.*guaranteed', 'Work from home guarantees'),
        (r'earn.*\$?\d+.*per.*day', 'Unrealistic earning claims'),
        (r'contact.*whatsapp.*only', 'WhatsApp only contact'),
        (r'send.*money.*first', 'Upfront payment requests'),
        (r'registration.*fee', 'Registration fee requirements'),
        (r'guaranteed.*job', 'Job guarantees'),
        (r'easy.*money', 'Easy money promises'),
        (r'part.*time.*full.*time.*salary', 'Unrealistic part-time salaries')
    ]
    
    # Positive indicators (indicating genuine jobs)
    positive_patterns = [
        (r'company.*name', 'Company name mentioned'),
        (r'job.*description', 'Detailed job description'),
        (r'requirements?', 'Job requirements listed'),
        (r'qualifications?', 'Qualifications specified'),
        (r'benefits?', 'Benefits mentioned'),
        (r'salary.*range', 'Salary range provided'),
        (r'contact.*email', 'Email contact provided'),
        (r'office.*address', 'Office address given'),
        (r'interview.*process', 'Interview process described'),
        (r'working.*hours', 'Working hours specified')
    ]
    
    # Check for suspicious patterns
    found_suspicious = []
    for pattern, description in suspicious_patterns:
        if re.search(pattern, text_lower):
            found_suspicious.append(description)
    
    # Check for positive indicators
    found_positive = []
    for pattern, description in positive_patterns:
        if re.search(pattern, text_lower):
            found_positive.append(description)
    
    # Calculate score
    base_score = 0.5
    
    # Penalize for suspicious patterns
    suspicious_penalty = len(found_suspicious) * 0.15
    
    # Reward for positive indicators
    positive_bonus = len(found_positive) * 0.1
    
    # Text quality assessment
    text_quality = assess_text_quality(text)
    quality_bonus = 0.1 if text_quality == 'good' else 0.0
    
    final_score = base_score - suspicious_penalty + positive_bonus + quality_bonus
    final_score = max(0.0, min(1.0, final_score))
    
    return {
        'score': final_score,
        'suspicious_patterns': found_suspicious,
        'positive_indicators': found_positive,
        'text_quality': text_quality,
        'suspicious_count': len(found_suspicious),
        'positive_count': len(found_positive)
    }

def assess_text_quality(text):
    """Assess the quality of extracted text"""
    try:
        if len(text) < 20:
            return 'poor'
        
        # Use TextBlob for basic analysis
        blob = TextBlob(text)
        
        # Check sentence structure
        sentences = blob.sentences
        if len(sentences) == 0:
            return 'poor'
        
        # Average words per sentence
        total_words = len(blob.words)
        avg_words_per_sentence = total_words / len(sentences)
        
        # Check for reasonable sentence length
        if avg_words_per_sentence < 3:
            return 'poor'
        elif avg_words_per_sentence > 5:
            return 'good'
        else:
            return 'fair'
            
    except Exception as e:
        logger.error(f"Error assessing text quality: {e}")
        return 'poor'

def create_response(status='success', data=None, error=None, message=None):
    """Create standardized API response"""
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    if error is not None:
        response['error'] = error
    
    if message is not None:
        response['message'] = message
    
    return response

def validate_request_data(data, required_fields):
    """Validate request data contains required fields"""
    if not data:
        return False, "No data provided"
    
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, None

def safe_filename(filename):
    """Create a safe filename with timestamp"""
    from werkzeug.utils import secure_filename
    
    # Secure the filename
    filename = secure_filename(filename)
    
    # Add timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    
    return f"{timestamp}_{name}{ext}"

def cleanup_old_files(directory, max_age_hours=24):
    """Clean up old uploaded files"""
    try:
        current_time = datetime.now()
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                # Get file modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # Calculate age in hours
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                # Remove if older than max_age_hours
                if age_hours > max_age_hours:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filename}")
                    
    except Exception as e:
        logger.error(f"Error cleaning up old files: {e}")

def get_file_info(filepath):
    """Get information about uploaded file"""
    try:
        stat = os.stat(filepath)
        
        return {
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting file info for {filepath}: {e}")
        return None
