"""
modules/__init__.py — Package exports for the facial analysis modules.
"""

from .face_detector        import FaceDetector, FaceResult, FaceDetectionResult
from .face_detector        import FACE_STATUS_NO_FACE, FACE_STATUS_SINGLE, FACE_STATUS_MULTIPLE
from .landmark_extractor   import LandmarkExtractor
from .eye_contact          import EyeContactAnalyzer
from .head_pose            import HeadPoseEstimator
from .emotion_detector     import EmotionDetector
from .smile_detector       import SmileDetector
from .feature_aggregator   import FeatureAggregator
from .interview_classifier import InterviewClassifier
from .behavioral_window    import BehavioralWindowAggregator, BehavioralWindowSnapshot
from .analysis_output      import build_analysis_frame
from .pipeline             import FacialAnalysisPipeline, PipelineFrameResult

__all__ = [
    # Face detection
    "FaceDetector",
    "FaceResult",
    "FaceDetectionResult",
    "FACE_STATUS_NO_FACE",
    "FACE_STATUS_SINGLE",
    "FACE_STATUS_MULTIPLE",
    # Analysis modules
    "LandmarkExtractor",
    "EyeContactAnalyzer",
    "HeadPoseEstimator",
    "EmotionDetector",
    "SmileDetector",
    "FeatureAggregator",
    "InterviewClassifier",
    # Aggregation
    "BehavioralWindowAggregator",
    "BehavioralWindowSnapshot",
    # Output serialization
    "build_analysis_frame",
    # Integrated Pipeline
    "FacialAnalysisPipeline",
    "PipelineFrameResult",
]
