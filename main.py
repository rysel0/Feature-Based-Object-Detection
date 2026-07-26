import logging
import cv2
from pathlib import Path

from src.config_parser import Config
from src.pipeline import ObjectMatchingPipeline

def setup_logging(output_dir: Path) -> None:
    """Настройка стандартного логирования в файл и консоль"""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "report.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def main():
    config = Config.load("config/config.yaml")
    
    query_dir = Path(config.paths.query_dir)
    scenes_dir = Path(config.paths.scene_dir)
    output_dir = Path(config.paths.output_dir)
    
    for directory in (query_dir, scenes_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    # Находим эталон
    query_files = sorted([f for f in query_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
    if not query_files:
        logger.error(f"Please drop any query image into: {query_dir}")
        return
        
    query_path = query_files[0]
    logger.info(f"Using object template (query): '{query_path.name}'")
    
    # Находим все сцены
    scene_files = sorted([f for f in scenes_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
    if not scene_files:
        logger.warning(f"No images to process. Drop scenes into: {scenes_dir}")
        return

    logger.info(f"Found {len(scene_files)} scene(s) to process. Starting pipeline...")
    
    pipeline = ObjectMatchingPipeline(config)

    for scene_path in scene_files:
        try:
            H, result_img, inliers = pipeline.match_object(query_path, scene_path)
            
            if H is not None:
                out_name = f"res_{scene_path.name}"
                cv2.imwrite(str(output_dir / out_name), result_img)
                logger.info(f"Scene '{scene_path.name}': Object FOUND. Inliers: {len(inliers)}")
            else:
                logger.info(f"Scene '{scene_path.name}': Object NOT found")
                
        except Exception as e:
            logger.error(f"Failed to process scene '{scene_path.name}': {e}", exc_info=True)

    logger.info("Pipeline processing completed successfully")

if __name__ == "__main__":
    main()
