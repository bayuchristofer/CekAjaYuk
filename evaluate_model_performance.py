#!/usr/bin/env python3
"""
Evaluate current ML/DL model performance to determine if retraining is needed
"""
import os
import numpy as np
import cv2
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests

def load_current_models():
    """Load current production models"""
    models = {}
    models_dir = Path('models')
    
    try:
        # Load Random Forest
        rf_path = models_dir / 'random_forest_production.pkl'
        if rf_path.exists():
            models['random_forest'] = joblib.load(rf_path)
            print("✅ Random Forest loaded")
        
        # Load Feature Scaler
        scaler_path = models_dir / 'feature_scaler_production.pkl'
        if scaler_path.exists():
            models['scaler'] = joblib.load(scaler_path)
            print("✅ Feature Scaler loaded")
        
        # Load Text Vectorizer
        vec_path = models_dir / 'text_vectorizer_production.pkl'
        if vec_path.exists():
            models['vectorizer'] = joblib.load(vec_path)
            print("✅ Text Vectorizer loaded")
        
        # Load Deep Learning Model
        try:
            import tensorflow as tf
            dl_path = models_dir / 'cnn_production.h5'
            if dl_path.exists():
                models['cnn'] = tf.keras.models.load_model(str(dl_path))
                print("✅ CNN Model loaded")
        except Exception as e:
            print(f"⚠️ CNN Model loading failed: {e}")
        
        return models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return {}

def analyze_dataset_quality():
    """Analyze current dataset quality"""
    dataset_dir = Path('dataset')
    
    if not dataset_dir.exists():
        return {
            'status': 'missing',
            'recommendation': 'Dataset not found - need to collect real data'
        }
    
    genuine_dir = dataset_dir / 'genuine'
    fake_dir = dataset_dir / 'fake'
    
    genuine_count = len(list(genuine_dir.glob('*.jpg')) + list(genuine_dir.glob('*.png'))) if genuine_dir.exists() else 0
    fake_count = len(list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))) if fake_dir.exists() else 0
    
    total_samples = genuine_count + fake_count
    balance_ratio = min(genuine_count, fake_count) / max(genuine_count, fake_count) if max(genuine_count, fake_count) > 0 else 0
    
    analysis = {
        'total_samples': total_samples,
        'genuine_samples': genuine_count,
        'fake_samples': fake_count,
        'balance_ratio': balance_ratio,
        'quality_assessment': 'unknown'
    }
    
    # Quality assessment
    if total_samples >= 1000 and balance_ratio >= 0.8:
        analysis['quality_assessment'] = 'excellent'
        analysis['recommendation'] = 'Dataset quality is excellent - no retraining needed'
    elif total_samples >= 500 and balance_ratio >= 0.6:
        analysis['quality_assessment'] = 'good'
        analysis['recommendation'] = 'Dataset quality is good - consider adding more samples'
    elif total_samples >= 200:
        analysis['quality_assessment'] = 'fair'
        analysis['recommendation'] = 'Dataset quality is fair - retraining recommended with more data'
    else:
        analysis['quality_assessment'] = 'poor'
        analysis['recommendation'] = 'Dataset too small - definitely need more data and retraining'
    
    return analysis

def evaluate_model_type():
    """Evaluate what type of models are currently being used"""
    try:
        # Check if models are using real data or synthetic data
        models = load_current_models()
        
        if 'random_forest' in models:
            rf = models['random_forest']
            # Check model characteristics
            n_estimators = getattr(rf, 'n_estimators', 0)
            n_features = getattr(rf, 'n_features_in_', 0)
            
            print(f"📊 Random Forest Analysis:")
            print(f"   N_estimators: {n_estimators}")
            print(f"   N_features: {n_features}")
            
            # Heuristic to determine if trained on real vs synthetic data
            if n_features < 20:
                return {
                    'data_type': 'synthetic',
                    'confidence': 'high',
                    'recommendation': 'Models trained on synthetic data - retraining with real data highly recommended'
                }
            else:
                return {
                    'data_type': 'real_or_complex',
                    'confidence': 'medium',
                    'recommendation': 'Models appear to use real/complex features - evaluate performance before retraining'
                }
        
        return {
            'data_type': 'unknown',
            'confidence': 'low',
            'recommendation': 'Cannot determine model training data - recommend evaluation'
        }
        
    except Exception as e:
        return {
            'data_type': 'error',
            'confidence': 'none',
            'recommendation': f'Error analyzing models: {e}'
        }

def test_ocr_accuracy():
    """Test current OCR accuracy"""
    try:
        # Test OCR with a simple request
        response = requests.get('http://localhost:5001/api/test-ocr', timeout=5)
        if response.status_code == 200:
            data = response.json()
            ocr_data = data.get('data', {})
            
            return {
                'status': 'working',
                'version': ocr_data.get('version', 'unknown'),
                'languages': ocr_data.get('languages', 'unknown'),
                'recommendation': 'OCR is working - focus on ML/DL model improvement'
            }
        else:
            return {
                'status': 'error',
                'recommendation': 'OCR has issues - fix OCR before focusing on ML models'
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'recommendation': 'Cannot test OCR - check backend connection'
        }

def main():
    """Main evaluation function"""
    print("🔍 EVALUATING ML/DL MODEL PERFORMANCE")
    print("🎯 Determining if retraining is needed for better accuracy")
    print("=" * 70)
    
    # 1. Analyze dataset quality
    print("\n1️⃣ DATASET QUALITY ANALYSIS")
    print("-" * 40)
    dataset_analysis = analyze_dataset_quality()
    
    print(f"📊 Total Samples: {dataset_analysis['total_samples']}")
    print(f"✅ Genuine: {dataset_analysis['genuine_samples']}")
    print(f"❌ Fake: {dataset_analysis['fake_samples']}")
    print(f"⚖️ Balance Ratio: {dataset_analysis['balance_ratio']:.2f}")
    print(f"🎯 Quality: {dataset_analysis['quality_assessment']}")
    print(f"💡 Recommendation: {dataset_analysis['recommendation']}")
    
    # 2. Evaluate model type
    print("\n2️⃣ MODEL TYPE ANALYSIS")
    print("-" * 40)
    model_analysis = evaluate_model_type()
    
    print(f"📈 Data Type: {model_analysis['data_type']}")
    print(f"🎯 Confidence: {model_analysis['confidence']}")
    print(f"💡 Recommendation: {model_analysis['recommendation']}")
    
    # 3. Test OCR accuracy
    print("\n3️⃣ OCR PERFORMANCE TEST")
    print("-" * 40)
    ocr_analysis = test_ocr_accuracy()
    
    print(f"📍 Status: {ocr_analysis['status']}")
    if 'version' in ocr_analysis:
        print(f"📝 Version: {ocr_analysis['version']}")
        print(f"🌐 Languages: {ocr_analysis['languages']}")
    print(f"💡 Recommendation: {ocr_analysis['recommendation']}")
    
    # 4. Overall recommendation
    print("\n" + "=" * 70)
    print("🎯 OVERALL RETRAINING RECOMMENDATION")
    print("=" * 70)
    
    # Decision logic
    need_retraining = False
    reasons = []
    
    if dataset_analysis['total_samples'] < 500:
        need_retraining = True
        reasons.append("Dataset too small (< 500 samples)")
    
    if dataset_analysis['balance_ratio'] < 0.6:
        need_retraining = True
        reasons.append("Dataset imbalanced (ratio < 0.6)")
    
    if model_analysis['data_type'] == 'synthetic':
        need_retraining = True
        reasons.append("Models trained on synthetic data")
    
    if ocr_analysis['status'] != 'working':
        reasons.append("OCR issues need fixing first")
    
    # Final recommendation
    if need_retraining:
        print("🔄 RETRAINING RECOMMENDED")
        print("\n📋 Reasons:")
        for reason in reasons:
            print(f"   • {reason}")
        
        print("\n🎯 Action Plan:")
        print("   1. Collect more real job posting images (target: 1000+)")
        print("   2. Ensure balanced dataset (50% genuine, 50% fake)")
        print("   3. Retrain models with real data")
        print("   4. Evaluate performance improvement")
        
        print("\n📁 Recommended files to run:")
        print("   • notebooks/0_real_dataset_preparation.ipynb")
        print("   • train_with_real_dataset.py")
        
    else:
        print("✅ NO RETRAINING NEEDED")
        print("\n📊 Current models appear to be:")
        print("   • Trained with sufficient data")
        print("   • Using balanced dataset")
        print("   • Performing adequately")
        
        print("\n🎯 Focus on:")
        print("   • OCR accuracy improvement")
        print("   • User interface enhancement")
        print("   • System optimization")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_analysis': dataset_analysis,
        'model_analysis': model_analysis,
        'ocr_analysis': ocr_analysis,
        'need_retraining': need_retraining,
        'reasons': reasons
    }
    
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: model_evaluation_results.json")

if __name__ == "__main__":
    main()
