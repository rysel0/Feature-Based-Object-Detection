import numpy as np
import cv2

from src.config_parser import MatcherConfig

class Matcher():

    def __init__(self, config: MatcherConfig):
        self.type = config.type
        self.ratio_threshold = config.ratio_threshold
        self.norm_type = config.norm_type
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        norm_mapping = {
            "L2": cv2.NORM_L2,
            "HAMMING": cv2.NORM_HAMMING,
            "HAMMING2": cv2.NORM_HAMMING2
        }
        if self.type == "bf":
            return cv2.BFMatcher(
                normType=norm_mapping[self.norm_type],
                crossCheck=False)
        elif self.type == "flann":
            return cv2.FlannBasedMatcher()
        else:
            raise ValueError(f"Unknown matcher type: {self.type}")
    
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        """
        Находит соответствия между дескрипторами с ratio test.
        
        Args:
            desc1: дескрипторы с первого изображения
            desc2: дескрипторы со второго изображения
            
        Returns:
            list: "хорошие" матчи после ratio test
        """
        if desc1 is None or desc2 is None:
            return []
        
        # knnMatch с k=2 для ratio test
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Ratio test (Lowe's algorithm)
        good_matches = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
