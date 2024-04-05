from pathlib import Path

IMAGES_PATH = Path(__file__).parent.parent / "images"
TEMPLATES_PATH = Path(__file__).parent / "templates"
TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)
