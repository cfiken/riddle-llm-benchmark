import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_assets_path() -> Path:
    """Return the path to the assets directory."""
    return Path(__file__).parent / "assets"


def get_image_assets_path() -> Path:
    """Return the path to the image assets directory."""
    return get_assets_path() / "images"


def get_prompt_assets_path() -> Path:
    """Return the path to the prompt assets directory."""
    return get_assets_path() / "prompts"
