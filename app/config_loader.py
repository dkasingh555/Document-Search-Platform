import os
import yaml
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

def load_yaml(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load application config
CONFIG_PATH = BASE_DIR / "config.yaml"
config = load_yaml(CONFIG_PATH)

# Load prompts
PROMPTS_PATH = BASE_DIR / config["paths"]["prompts_file"]
prompts = load_yaml(PROMPTS_PATH)

def get_config():
    return config

def get_prompts():
    return prompts
