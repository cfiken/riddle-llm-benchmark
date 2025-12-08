from pathlib import Path

def hello(name: str = "World") -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

def get_assets_path() -> Path:
    """Return the path to the assets directory."""
    return Path(__file__).parent / "assets"

def get_image_assets_path() -> Path:
    """Return the path to the image assets directory."""
    return get_assets_path() / "images"

def get_prompt_assets_path() -> Path:
    """Return the path to the prompt assets directory."""
    return get_assets_path() / "prompts"
