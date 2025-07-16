"""
Model loading and prediction utilities for CekAjaYuk
"""

import os
import joblib
import numpy as np
from datetime import datetime
import logging

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    logger = logging.getLogger(__name__)
    logger.warning(f"TensorFlow import failed: {e}. Deep learning features will be disabled.")

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and prediction with ML/DL models"""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.rf_model = None
        self.dl_model = None
        self.feature_scaler = None
        self.text_vectorizer = None
        self.models_loaded = False
        
    def load_models(self):
        """Load all available models"""
        loaded_count = 0
        total_count = 4

        # Load Random Forest
        try:
            self._load_random_forest()
            if self.rf_model is not None:
                loaded_count += 1
                logger.info("✅ Random Forest model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Random Forest model failed: {e}")

        # Load Deep Learning
        try:
            self._load_deep_learning()
            if self.dl_model is not None:
                loaded_count += 1
                logger.info("✅ Deep Learning model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Deep Learning model failed: {e}")

        # Load Feature Scaler
        try:
            self._load_feature_scaler()
            if self.feature_scaler is not None:
                loaded_count += 1
                logger.info("✅ Feature Scaler loaded")
        except Exception as e:
            logger.warning(f"⚠️ Feature Scaler failed: {e}")

        # Load Text Vectorizer
        try:
            self._load_text_vectorizer()
            if self.text_vectorizer is not None:
                loaded_count += 1
                logger.info("✅ Text Vectorizer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Text Vectorizer failed: {e}")

        self.models_loaded = loaded_count > 0
        logger.info(f"📊 Models loaded: {loaded_count}/{total_count}")

        if loaded_count == 0:
            logger.error("❌ No models could be loaded")
        elif loaded_count < total_count:
            logger.warning(f"⚠️ Partial loading: {loaded_count}/{total_count} models loaded")
        else:
            logger.info("✅ All models loaded successfully")
            self.models_loaded = False
    
    def _load_random_forest(self):
        """Load Random Forest model"""
        try:
            rf_path = os.path.join(self.models_dir, 'random_forest_classifier_latest.pkl')
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
            else:
                logger.warning(f"Random Forest model not found at {rf_path}")
        except Exception as e:
            logger.error(f"Error loading Random Forest model: {e}")
    
    def _load_deep_learning(self):
        """Load TensorFlow/Keras model"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Deep learning features disabled.")
            return

        try:
            # Try to load the CNN model that was actually trained
            dl_path = os.path.join(self.models_dir, 'cnn_best_real.h5')
            if os.path.exists(dl_path):
                self.dl_model = tf.keras.models.load_model(dl_path)
                logger.info("CNN model loaded from cnn_best_real.h5")
            else:
                # Fallback to the original name
                dl_path = os.path.join(self.models_dir, 'tensorflow_model_latest.h5')
                if os.path.exists(dl_path):
                    self.dl_model = tf.keras.models.load_model(dl_path)
                    logger.info("TensorFlow model loaded")
                else:
                    logger.warning(f"No TensorFlow model found. Checked: cnn_best_real.h5 and tensorflow_model_latest.h5")
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            self.dl_model = None
    
    def _load_feature_scaler(self):
        """Load feature scaler"""
        try:
            scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded")
            else:
                logger.warning(f"Feature scaler not found at {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading feature scaler: {e}")
    
    def _load_text_vectorizer(self):
        """Load text vectorizer for text analysis"""
        try:
            vectorizer_path = os.path.join(self.models_dir, 'text_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.text_vectorizer = joblib.load(vectorizer_path)
                logger.info("Text vectorizer loaded")
            else:
                logger.warning(f"Text vectorizer not found at {vectorizer_path}")
        except Exception as e:
            logger.error(f"Error loading text vectorizer: {e}")
    
    def predict_with_random_forest(self, features):
        """Make prediction using Random Forest model"""
        try:
            if self.rf_model is None or self.feature_scaler is None:
                return None

            # Ensure features have the right shape
            if len(features) < 12:
                logger.warning(f"Insufficient features: {len(features)}, expected 12")
                return None

            # Use the first 12 features (matching training data)
            features_subset = features[:12].reshape(1, -1)

            # Scale features
            features_scaled = self.feature_scaler.transform(features_subset)

            # Make prediction
            prediction = self.rf_model.predict(features_scaled)[0]
            probabilities = self.rf_model.predict_proba(features_scaled)[0]

            return {
                'prediction': 'genuine' if prediction == 1 else 'fake',
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'fake': float(probabilities[0]),
                    'genuine': float(probabilities[1])
                },
                'model': 'Random Forest'
            }

        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            return None
    
    def predict_with_deep_learning(self, image_array):
        """Make prediction using Deep Learning model"""
        try:
            if not TF_AVAILABLE:
                logger.warning("TensorFlow not available for deep learning prediction")
                return None

            if self.dl_model is None:
                logger.warning("Deep learning model not loaded")
                return None
            
            # Ensure image has the right shape
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            prediction = self.dl_model.predict(image_array, verbose=0)[0][0]
            
            return {
                'prediction': 'genuine' if prediction > 0.5 else 'fake',
                'confidence': float(prediction if prediction > 0.5 else 1 - prediction),
                'raw_score': float(prediction),
                'model': 'Deep Learning (CNN)'
            }
            
        except Exception as e:
            logger.error(f"Error in Deep Learning prediction: {e}")
            return None
    
    def predict_text_analysis(self, text_features):
        """Make prediction using text analysis model"""
        try:
            if self.text_vectorizer is None:
                # Use rule-based analysis if no model available
                return self._rule_based_text_analysis(text_features)
            
            # If we have a trained text model, use it here
            # For now, fall back to rule-based analysis
            return self._rule_based_text_analysis(text_features)
            
        except Exception as e:
            logger.error(f"Error in text analysis prediction: {e}")
            return None
    
    def _rule_based_text_analysis(self, text_features):
        """Rule-based text analysis as fallback"""
        try:
            score = text_features.get('score', 0.5)
            
            return {
                'prediction': 'genuine' if score > 0.5 else 'fake',
                'confidence': score if score > 0.5 else (1 - score),
                'suspicious_patterns': text_features.get('suspicious_patterns', []),
                'positive_indicators': text_features.get('positive_indicators', []),
                'text_quality': text_features.get('text_quality', 'unknown'),
                'model': 'Rule-based Text Analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based text analysis: {e}")
            return {
                'prediction': 'fake',
                'confidence': 0.3,
                'model': 'Rule-based Text Analysis (Error)'
            }
    
    def combine_predictions(self, rf_result, dl_result, text_result):
        """Combine predictions from multiple models"""
        try:
            predictions = []
            confidences = []
            weights = []
            
            # Random Forest
            if rf_result:
                predictions.append(1 if rf_result['prediction'] == 'genuine' else 0)
                confidences.append(rf_result['confidence'])
                weights.append(0.4)  # 40% weight
            
            # Deep Learning
            if dl_result:
                predictions.append(1 if dl_result['prediction'] == 'genuine' else 0)
                confidences.append(dl_result['confidence'])
                weights.append(0.4)  # 40% weight
            
            # Text Analysis
            if text_result:
                predictions.append(1 if text_result['prediction'] == 'genuine' else 0)
                confidences.append(text_result['confidence'])
                weights.append(0.2)  # 20% weight
            
            if not predictions:
                return {
                    'prediction': 'fake',
                    'confidence': 0.3,
                    'method': 'No models available'
                }
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted average of predictions
            weighted_prediction = sum(p * w for p, w in zip(predictions, weights))
            
            # Weighted average of confidences
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
            
            # Final prediction
            final_prediction = 'genuine' if weighted_prediction > 0.5 else 'fake'
            
            return {
                'prediction': final_prediction,
                'confidence': weighted_confidence,
                'weighted_score': weighted_prediction,
                'individual_results': {
                    'random_forest': rf_result,
                    'deep_learning': dl_result,
                    'text_analysis': text_result
                },
                'method': 'Weighted Ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return {
                'prediction': 'fake',
                'confidence': 0.3,
                'method': 'Error in combination'
            }
    
    def get_model_info(self):
        """Get information about loaded models"""
        loaded_count = 0
        total_count = 4

        info = {
            'models_loaded': self.models_loaded,
            'timestamp': datetime.now().isoformat(),
            'available_models': {}
        }

        # Random Forest Model
        if self.rf_model is not None:
            info['available_models']['random_forest'] = {
                'type': 'RandomForestClassifier',
                'n_estimators': getattr(self.rf_model, 'n_estimators', 'unknown'),
                'loaded': True,
                'status': '✅ Ready'
            }
            loaded_count += 1
        else:
            info['available_models']['random_forest'] = {
                'type': 'RandomForestClassifier',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'Model file not found or failed to load'
            }

        # Deep Learning Model
        if self.dl_model is not None:
            try:
                info['available_models']['deep_learning'] = {
                    'type': 'TensorFlow/Keras CNN',
                    'input_shape': str(self.dl_model.input_shape) if hasattr(self.dl_model, 'input_shape') else 'unknown',
                    'output_shape': str(self.dl_model.output_shape) if hasattr(self.dl_model, 'output_shape') else 'unknown',
                    'loaded': True,
                    'status': '✅ Ready'
                }
                loaded_count += 1
            except Exception as e:
                info['available_models']['deep_learning'] = {
                    'type': 'TensorFlow/Keras CNN',
                    'loaded': True,
                    'status': '⚠️ Loaded with issues',
                    'error': str(e)
                }
                loaded_count += 1
        else:
            info['available_models']['deep_learning'] = {
                'type': 'TensorFlow/Keras CNN',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'TensorFlow not available or model file not found'
            }

        # Feature Scaler
        if self.feature_scaler is not None:
            info['available_models']['feature_scaler'] = {
                'type': type(self.feature_scaler).__name__,
                'loaded': True,
                'status': '✅ Ready'
            }
            loaded_count += 1
        else:
            info['available_models']['feature_scaler'] = {
                'type': 'StandardScaler/MinMaxScaler',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'Scaler file not found'
            }

        # Text Vectorizer
        if self.text_vectorizer is not None:
            info['available_models']['text_vectorizer'] = {
                'type': type(self.text_vectorizer).__name__,
                'loaded': True,
                'status': '✅ Ready'
            }
            loaded_count += 1
        else:
            info['available_models']['text_vectorizer'] = {
                'type': 'TfidfVectorizer/CountVectorizer',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'Vectorizer file not found'
            }

        # Summary
        info['summary'] = {
            'loaded_count': loaded_count,
            'total_count': total_count,
            'load_percentage': round((loaded_count / total_count) * 100, 1),
            'status': 'Ready' if loaded_count > 0 else 'No models loaded'
        }

        return info
    
    def create_mock_prediction(self, prediction_type='random'):
        """Create mock prediction for demonstration"""
        if prediction_type == 'random':
            confidence = np.random.uniform(0.7, 0.95)
            prediction = np.random.choice(['genuine', 'fake'])
        elif prediction_type == 'genuine':
            confidence = np.random.uniform(0.8, 0.95)
            prediction = 'genuine'
        else:  # fake
            confidence = np.random.uniform(0.7, 0.9)
            prediction = 'fake'
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model': 'Mock Model (Demo)',
            'note': 'This is a demonstration prediction'
        }
