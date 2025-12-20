"""
Batting Service - Research-Ready Intent Scoring
Complete rewrite with temporal features, ensemble models, and visual feedback
"""

import numpy as np
import joblib
import os
import json
from typing import Dict, List
from google import genai

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.temporal_feature_engineer import TemporalFeatureEngineer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.mistake_analyzer import MistakeAnalyzer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.visual_feedback_generator import VisualFeedbackGenerator
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    MODEL_FOLDER_PATH
)
from features.SHOT_CLASSIFICATION_SYSTEM.utils.json_utils import to_json_safe


class BattingService:
    """Advanced batting analysis service with ensemble models and visual feedback"""
    
    def __init__(self, model_dir: str = MODEL_FOLDER_PATH):
        self.model_dir = model_dir
        
        # Load ensemble models
        self.models = self._load_models()
        self.scaler = joblib.load(f"{model_dir}/ensemble/scaler.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/ensemble/label_encoder.pkl")
        
        with open(f"{model_dir}/ensemble/feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
        
        # Initialize components
        self.frame_extractor = FrameExtractor(fps=10)
        self.pose_estimator = PoseEstimator()
        self.feature_engineer = TemporalFeatureEngineer()
        self.mistake_analyzer = MistakeAnalyzer()
        self.visual_generator = VisualFeedbackGenerator()
        
        # Initialize AI feedback (optional)
        api_key = os.getenv('GEMINI_API_KEY')
        self.ai_client = genai.Client(api_key=api_key) if api_key else None
        
        print("Batting service initialized with ensemble models")
    
    def _load_models(self) -> Dict:
        """Load all trained models"""
        models = {}
        
        for model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            model_path = f"{self.model_dir}/{model_name}/model_latest.pkl"
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded {model_name}")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        return models
    
    def process_video(self, video_path: str) -> Dict:
        """Process video and extract temporal features with metadata"""
        # Extract frames
        frames, fps = self.frame_extractor.extract_frames(video_path)
        
        # Get pose sequence
        pose_sequence = self.pose_estimator.estimate_pose_batch(frames)
        
        # Extract temporal features
        features, metadata = self.feature_engineer.extract_temporal_features(pose_sequence)
        
        # Store frames and poses for visual feedback
        contact_frame_idx = metadata['contact_frame']
        contact_frame = frames[contact_frame_idx]
        contact_pose = pose_sequence[contact_frame_idx]
        
        return {
            'features': features,
            'metadata': metadata,
            'contact_frame': contact_frame,
            'contact_keypoints': contact_pose['keypoints'],
            'contact_scores': contact_pose['scores']
        }
    
    def ensemble_predict(self, features: np.ndarray) -> Dict:
        """
        Get predictions from ensemble with voting
        
        Returns:
            Dictionary with ensemble results
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from each model
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            predictions[name] = self.label_encoder.inverse_transform([pred])[0]
            probabilities[name] = {
                shot: float(prob) 
                for shot, prob in zip(self.label_encoder.classes_, proba)
            }
        
        # Ensemble voting (average probabilities)
        ensemble_proba = {}
        for shot_class in self.label_encoder.classes_:
            avg_prob = np.mean([probabilities[model][shot_class] for model in self.models.keys()])
            ensemble_proba[shot_class] = float(avg_prob)
        
        # Final prediction (highest probability)
        final_prediction = max(ensemble_proba, key=ensemble_proba.get)
        
        return {
            'final_prediction': final_prediction,
            'ensemble_probabilities': ensemble_proba,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
    
    def calculate_intent_score(self, intended_shot: str, ensemble_proba: Dict[str, float]) -> float:
        """Calculate intent execution score using ensemble probabilities"""
        intended_prob = ensemble_proba.get(intended_shot, 0.0)
        max_prob = max(ensemble_proba.values())
        
        if max_prob > 0:
            score = (intended_prob / max_prob) * 100
        else:
            score = 0.0
        
        return round(score, 2)
    
    def generate_ai_feedback(self, intended_shot: str, predicted_shot: str,
                           intent_score: float, mistakes: List[Dict]) -> str:
        """Generate AI-powered coaching feedback"""
        if not self.ai_client:
            return self._generate_rule_based_feedback(intended_shot, predicted_shot, 
                                                      intent_score, mistakes)
        
        try:
            # Prepare mistake summary
            mistake_summary = "\n".join([
                f"- {m['joint']}: {m['issue']} (actual: {m['actual_angle']}, expected: {m['expected_range']})"
                for m in mistakes[:5]
            ])
            
            prompt = f"""You are an expert cricket batting coach analyzing a player's shot execution.

Player's Intent: {intended_shot.upper()}
Actual Execution: {predicted_shot.upper()}
Intent Score: {intent_score}% (how well they executed their intended shot)

Technical Issues Detected:
{mistake_summary if mistakes else "No major technical issues detected."}

Provide coaching feedback in 2-3 concise sentences:
1. Acknowledge what they did well (if applicable)
2. Point out the main technical issue affecting their intent score
3. Give ONE specific, actionable correction

Be direct, supportive, and coaching-focused. No bullet points."""

            response = self.ai_client.models.generate_content(
                model="gemini-2.5-flash",  
                contents=prompt
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"AI feedback failed: {e}")
            return self._generate_rule_based_feedback(intended_shot, predicted_shot, 
                                                      intent_score, mistakes)
    
    def _generate_rule_based_feedback(self, intended_shot: str, predicted_shot: str,
                                     intent_score: float, mistakes: List[Dict]) -> str:
        """Generate rule-based feedback (fallback)"""
        if intent_score >= 85:
            base = f"Excellent {intended_shot} execution! Your technique is nearly perfect."
        elif intent_score >= 70:
            base = f"Good {intended_shot} attempt. Your technique is solid with minor adjustments needed."
        elif intent_score >= 50:
            base = f"Your {intended_shot} execution needs refinement. The model detected it as {predicted_shot}."
        else:
            base = f"Your shot significantly deviated from the intended {intended_shot}, appearing as {predicted_shot}."
        
        if mistakes:
            top_issue = mistakes[0]
            addition = f" Main issue: {top_issue['issue']}. {top_issue['recommendation']}"
            return base + addition
        
        return base + " Keep practicing to maintain consistency!"
    
    def analyze_shot(self, video_path: str, intended_shot: str) -> Dict:
        """
        Complete shot analysis with visual feedback
        
        Args:
            video_path: Path to video file
            intended_shot: User's intended shot
            
        Returns:
            Comprehensive analysis with images
        """
        # Process video
        video_data = self.process_video(video_path)
        
        # Get ensemble predictions
        ensemble_result = self.ensemble_predict(video_data['features'])
        
        # Calculate intent score
        intent_score = self.calculate_intent_score(
            intended_shot, 
            ensemble_result['ensemble_probabilities']
        )
        
        # Analyze mistakes
        mistakes = self.mistake_analyzer.analyze_execution(
            intended_shot,
            ensemble_result['final_prediction'],
            video_data['metadata']['angles']
        )
        
        # Generate visual feedback
        visual_feedback = self.visual_generator.generate_feedback_images(
            video_data['contact_frame'],
            video_data['contact_keypoints'],
            video_data['contact_scores'],
            mistakes
        )
        
        # Generate AI coaching feedback
        coaching_feedback = self.generate_ai_feedback(
            intended_shot,
            ensemble_result['final_prediction'],
            intent_score,
            mistakes
        )
        
        # Get correction summary
        correction_summary = self.mistake_analyzer.generate_correction_summary(
            mistakes, intended_shot
        )
        
        # Compile results
        result = {
            'intended_shot': intended_shot,
            'predicted_shot': ensemble_result['final_prediction'],
            'intent_score': intent_score,
            'is_correct': ensemble_result['final_prediction'] == intended_shot,
            
            # Visual feedback
            'images': visual_feedback,
            
            # Detailed analysis
            'mistake_analysis': mistakes,
            'correction_summary': correction_summary,
            'coaching_feedback': coaching_feedback,
            
            # Technical details
            'ensemble_probabilities': ensemble_result['ensemble_probabilities'],
            'model_predictions': ensemble_result['individual_predictions'],
            
            # Metadata
            'contact_frame_index': video_data['metadata']['contact_frame'],
            'analysis_metadata': {
                'pre_frame': video_data['metadata']['pre_frame'],
                'contact_frame': video_data['metadata']['contact_frame'],
                'follow_frame': video_data['metadata']['follow_frame']
            }
        }
        
        return to_json_safe(result)
    
    def get_shot_types(self) -> List[str]:
        """Get available shot types"""
        return self.label_encoder.classes_.tolist()


# Global service instance
_batting_service = None

def get_batting_service() -> BattingService:
    """Get or create batting service instance"""
    global _batting_service
    if _batting_service is None:
        _batting_service = BattingService()
    return _batting_service