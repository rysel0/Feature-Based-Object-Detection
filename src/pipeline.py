# src/pipeline.py
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from src.config_parser import Config
from src.detectors import FeatureExtractor
from src.matchers import Matcher
from src.geometry import HomographyEstimator

class ObjectMatchingPipeline:
    """Основной пайплайн для поиска объекта на изображении."""
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = FeatureExtractor(config.detector)
        self.matcher = Matcher(config.matcher)
        self.homography = HomographyEstimator(config.homography)
    
    def match_object(
        self, 
        query_path: Path, 
        scene_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list]:

        query_img = cv2.imread(str(query_path))
        scene_img = cv2.imread(str(scene_path))
        
        if query_img is None or scene_img is None:
            raise ValueError("Failed to load images")
            
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
        
        # Детекция ключевых точек
        kp1, desc1 = self.detector.detect_and_compute(query_img)
        kp2, desc2 = self.detector.detect_and_compute(scene_img)
        
        if desc1 is None or desc2 is None:
            return None, None, []
        
        # Матчинг
        matches = self.matcher.match(desc1, desc2)
        
        if len(matches) < self.homography.min_matches:
            return None, None, []
        
        # Гомография
        H, inliers = self.homography.compute_homography(kp1, kp2, matches)
        
        if H is None:
            return None, None, []
        
        result_img = self._draw_result(query_img, scene_img, kp1, kp2, inliers, H)
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        return H, result_img_bgr, inliers
    
    def _draw_result(
        self,
        query_img: np.ndarray,
        scene_img: np.ndarray,
        kp1: list,
        kp2: list,
        inliers: list,
        H: np.ndarray
    ) -> np.ndarray:
        """Отрисовка bounding box"""
        h, w = query_img.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        result_img = cv2.polylines(scene_img.copy(), [np.int32(dst)], True, (255, 0, 0), 5)
        
        return result_img