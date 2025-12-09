from riddle_benchmark.utils import get_assets_path, get_image_assets_path, get_prompt_assets_path


def test_assets_paths():
    assets_path = get_assets_path()
    assert assets_path.exists()
    assert assets_path.is_dir()
    assert assets_path.name == "assets"

    images_path = get_image_assets_path()
    assert images_path.exists()
    assert images_path.is_dir()
    assert images_path.name == "images"

    prompts_path = get_prompt_assets_path()
    assert prompts_path.exists()
    assert prompts_path.is_dir()
    assert prompts_path.name == "prompts"
