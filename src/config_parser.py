import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Literal

class PathsConfig(BaseModel):
    query_dir: str = "data/query"
    scene_dir: str = "data/scenes"
    output_dir: str = "data/output"

class DetectorConfig(BaseModel):
    method: Literal["sift", "orb", "akaze"] = "sift"
    nfeatures: int = 1000

class MatcherConfig(BaseModel):
    norm_type: Literal["L2", "HAMMING", "HAMMING2"] = "L2"
    type: Literal["bf", "flann"] = "bf"
    ratio_threshold: float = 0.7

class HomographyConfig(BaseModel):
    ransac_threshold: float = 5.0
    min_matches: int = 10

class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    detector: DetectorConfig = DetectorConfig()
    matcher: MatcherConfig = MatcherConfig()
    homography: HomographyConfig = HomographyConfig()

    @classmethod
    def load(cls, config_path: str | Path):
        with open(config_path, 'r', encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)