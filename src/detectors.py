from typing import Tuple, Optional
import numpy as np
import cv2

from src.config_parser import DetectorConfig

class FeatureExtractor:
    
    def __init__(self, config: DetectorConfig):
        """
        Args:
            config: словарь с параметрами детектора из config.yaml
                   например: {"method": "sift", "nfeatures": 1000}
        """
        self.method = config.method
        self.params = config.model_dump(exclude={"method"})
        self.detector = self._create_detector()
    
    def _create_detector(self):
        """Создает детектор в зависимости от method"""

        if self.method == "sift":
            return cv2.SIFT_create(**self.params)
        elif self.method == "orb":
            return cv2.ORB_create(**self.params)
        elif self.method == "akaze":
            return cv2.AKAZE_create(**self.params)
        else:
            raise ValueError(f"Unknown detector method: {self.method}")

    def detect_and_compute(self, image: np.ndarray) -> Tuple[list, Optional[np.ndarray]]:
        """
        Находит ключевые точки и дескрипторы на изображении.
        """
        if image is None:
            raise ValueError("Image is None. Check if image was loaded correctly.")
        
        # Все детекторы работают с изображением в оттенках серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
        