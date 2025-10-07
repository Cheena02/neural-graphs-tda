import sys
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.io.enhanced_loader import EnhancedDataLoader
from src.utils.logger import TDALogger

print("--- Starting Loader Debug Script ---")

# Initialize components
logger = TDALogger(name="DebugLoader", level="DEBUG")
loader = EnhancedDataLoader(logger=logger)
config_path = "datasets/synthetic_verification.yaml"

logger.info(f"Attempting to load configuration: {config_path}")

try:
    # Load the dataset configuration
    config = loader.load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Discover files based on the configuration
    files = loader.discover_files(config)
    logger.info(f"File discovery complete. Found {len(files)} file(s).")

    if files:
        logger.info("Files found:")
        for f in files:
            logger.info(f"- {f}")
    else:
        logger.warning("No files were found. Please check the path and extensions in the YAML config.")

except Exception as e:
    logger.error(f"An error occurred during the debug run: {e}", exc_info=True)

print("--- Finished Loader Debug Script ---")

