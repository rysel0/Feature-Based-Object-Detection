# src/geometry.py
import numpy as np
import cv2
from typing import Tuple, Optional

from src.config_parser import HomographyConfig

class HomographyEstimator:
    """Вычисление гомографии между двумя изображениями."""
    
    def __init__(self, config: HomographyConfig):
        self.ransac_threshold = config.ransac_threshold
        self.min_matches = config.min_matches
    
    def compute_homography(
        self, 
        kp1: list, 
        kp2: list, 
        matches: list
    ) -> Tuple[Optional[np.ndarray], list]:
        if len(matches) < self.min_matches:
            return None, []
        
        # Подготовка точек для гомографии
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # RANSAC гомография
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if H is None:
            return None, []
        
        # Фильтрация финальных inliers
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        
        return H, inliers
