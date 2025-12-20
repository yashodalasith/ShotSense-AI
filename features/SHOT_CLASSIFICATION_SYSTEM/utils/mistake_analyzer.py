"""
Mistake Analyzer
Analyzes execution errors by comparing actual vs expected angles
"""

import numpy as np
from typing import Dict, List


class MistakeAnalyzer:
    """Analyze mistakes in shot execution"""
    
    def __init__(self):
        # Expected angle ranges for each shot type
        self.expected_angles = {
            'drive': {
                'front_elbow': (160, 175),
                'back_elbow': (140, 165),
                'front_knee': (160, 180),
                'back_knee': (155, 175),
                'torso_bend': (160, 180),
                'shoulder_rotation': (70, 90)
            },
            'cut': {
                'front_elbow': (90, 120),
                'back_elbow': (100, 130),
                'front_knee': (140, 170),
                'back_knee': (130, 160),
                'torso_bend': (140, 170),
                'shoulder_rotation': (45, 75)
            },
            'pull': {
                'front_elbow': (80, 110),
                'back_elbow': (70, 100),
                'front_knee': (90, 130),
                'back_knee': (100, 140),
                'torso_bend': (120, 150),
                'shoulder_rotation': (30, 60)
            },
            'flick': {
                'front_elbow': (130, 160),
                'back_elbow': (120, 150),
                'front_knee': (145, 175),
                'back_knee': (140, 170),
                'torso_bend': (150, 175),
                'shoulder_rotation': (60, 85)
            },
            'sweep': {
                'front_elbow': (100, 135),
                'back_elbow': (90, 125),
                'front_knee': (70, 110),
                'back_knee': (80, 120),
                'torso_bend': (110, 140),
                'shoulder_rotation': (35, 65)
            },
            'slog': {
                'front_elbow': (110, 145),
                'back_elbow': (95, 130),
                'front_knee': (120, 160),
                'back_knee': (115, 155),
                'torso_bend': (140, 165),
                'shoulder_rotation': (50, 80)
            },
            'misc': {
                'front_elbow': (90, 170),
                'back_elbow': (80, 160),
                'front_knee': (90, 180),
                'back_knee': (90, 175),
                'torso_bend': (120, 180),
                'shoulder_rotation': (30, 90)
            }
        }
        
        # Friendly names for joints
        self.joint_names = {
            'front_elbow': 'Front Elbow',
            'back_elbow': 'Back Elbow',
            'front_knee': 'Front Knee',
            'back_knee': 'Back Knee',
            'torso_bend': 'Torso',
            'shoulder_rotation': 'Shoulder Rotation'
        }
        
        # Error descriptions
        self.error_descriptions = {
            'too_bent': {
                'front_elbow': 'Elbow collapsed during shot execution',
                'back_elbow': 'Back arm too bent, reducing power transfer',
                'front_knee': 'Front leg too bent, affecting balance',
                'back_knee': 'Back leg bent excessively',
                'torso_bend': 'Torso bent too far forward',
                'shoulder_rotation': 'Insufficient shoulder rotation'
            },
            'too_straight': {
                'front_elbow': 'Front arm too rigid, limiting control',
                'back_elbow': 'Back arm over-extended',
                'front_knee': 'Front leg too straight, reducing power',
                'back_knee': 'Back leg locked',
                'torso_bend': 'Torso too upright',
                'shoulder_rotation': 'Excessive shoulder rotation'
            }
        }
    
    def analyze_execution(self, intended_shot: str, actual_shot: str, 
                         actual_angles: Dict[str, float]) -> List[Dict]:
        """
        Analyze execution mistakes
        
        Args:
            intended_shot: What player intended to play
            actual_shot: What model predicted
            actual_angles: Actual angles from the shot
            
        Returns:
            List of mistake dictionaries
        """
        mistakes = []
        
        if intended_shot not in self.expected_angles:
            return mistakes
        
        expected_ranges = self.expected_angles[intended_shot]
        
        for joint, (min_angle, max_angle) in expected_ranges.items():
            # Get actual angle (with fallback)
            actual_angle = actual_angles.get(f'contact_{joint}', None)
            
            if actual_angle is None:
                continue
            
            # Check if angle is out of range
            if actual_angle < min_angle:
                # Too bent
                severity = self._calculate_severity(actual_angle, min_angle, 'below')
                mistakes.append({
                    'joint': self.joint_names[joint],
                    'joint_id': joint,
                    'issue': self.error_descriptions['too_bent'][joint],
                    'expected_range': f"{min_angle:.0f}° – {max_angle:.0f}°",
                    'actual_angle': f"{actual_angle:.1f}°",
                    'deviation': f"{min_angle - actual_angle:.1f}° below minimum",
                    'severity': severity,
                    'recommendation': self._get_recommendation(joint, 'too_bent')
                })
            elif actual_angle > max_angle:
                # Too straight
                severity = self._calculate_severity(actual_angle, max_angle, 'above')
                mistakes.append({
                    'joint': self.joint_names[joint],
                    'joint_id': joint,
                    'issue': self.error_descriptions['too_straight'][joint],
                    'expected_range': f"{min_angle:.0f}° – {max_angle:.0f}°",
                    'actual_angle': f"{actual_angle:.1f}°",
                    'deviation': f"{actual_angle - max_angle:.1f}° above maximum",
                    'severity': severity,
                    'recommendation': self._get_recommendation(joint, 'too_straight')
                })
        
        # Sort by severity
        mistakes.sort(key=lambda x: x['severity'], reverse=True)
        
        return mistakes
    
    def _calculate_severity(self, actual: float, boundary: float, direction: str) -> str:
        """Calculate how severe the mistake is"""
        if direction == 'below':
            diff = boundary - actual
        else:  # above
            diff = actual - boundary
        
        if diff > 30:
            return 'critical'
        elif diff > 15:
            return 'major'
        elif diff > 5:
            return 'minor'
        else:
            return 'negligible'
    
    def _get_recommendation(self, joint: str, error_type: str) -> str:
        """Get specific recommendation for fixing the error"""
        recommendations = {
            'front_elbow': {
                'too_bent': 'Extend front arm more during shot execution. Focus on pushing through the ball with a straighter front arm.',
                'too_straight': 'Allow slight bend in front elbow for better control. Avoid locking the arm completely.'
            },
            'back_elbow': {
                'too_bent': 'Keep back elbow higher and straighter during backswing. This helps generate more power.',
                'too_straight': 'Allow natural bend in back elbow. Over-extension reduces control and timing.'
            },
            'front_knee': {
                'too_bent': 'Straighten front leg more on contact. Transfer weight forward with a firmer base.',
                'too_straight': 'Bend front knee slightly for better balance. A completely locked leg reduces power.',
            },
            'back_knee': {
                'too_bent': 'Keep back leg more upright during execution. This maintains balance and head position.',
                'too_straight': 'Allow slight flex in back knee for stability. Too straight reduces mobility.'
            },
            'torso_bend': {
                'too_bent': 'Keep head and chest more upright. Excessive bending affects timing and shot selection.',
                'too_straight': 'Get into better position by bending slightly forward. This improves reach and timing.'
            },
            'shoulder_rotation': {
                'too_bent': 'Rotate shoulders more during shot. Increased rotation generates more power.',
                'too_straight': 'Control shoulder rotation. Over-rotation causes loss of balance.'
            }
        }
        
        return recommendations.get(joint, {}).get(error_type, 'Focus on proper technique for this body part.')
    
    def generate_correction_summary(self, mistakes: List[Dict], intended_shot: str) -> str:
        """Generate overall correction summary"""
        if not mistakes:
            return f"Excellent technique! Your {intended_shot} execution was within optimal ranges for all key body positions."
        
        critical = [m for m in mistakes if m['severity'] == 'critical']
        major = [m for m in mistakes if m['severity'] == 'major']
        minor = [m for m in mistakes if m['severity'] == 'minor']
        
        summary_parts = []
        
        if critical:
            summary_parts.append(f"🔴 Critical issues ({len(critical)}): " + 
                               ", ".join([m['joint'] for m in critical]))
        
        if major:
            summary_parts.append(f"🟡 Major issues ({len(major)}): " + 
                               ", ".join([m['joint'] for m in major]))
        
        if minor:
            summary_parts.append(f"🟢 Minor adjustments ({len(minor)}): " + 
                               ", ".join([m['joint'] for m in minor]))
        
        summary = " | ".join(summary_parts)
        
        # Add overall advice
        if critical or major:
            summary += f"\n\nPriority: Focus on correcting the {mistakes[0]['joint']} first, as this has the biggest impact on your {intended_shot} execution."
        
        return summary