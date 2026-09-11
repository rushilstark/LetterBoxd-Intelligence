"""
config_store.py
Stores API keys persistently in ~/.culturaliq/config.json
Users enter keys ONCE — never asked again.
"""
import json
from pathlib import Path

CONFIG_DIR  = Path.home() / '.culturaliq'
CONFIG_FILE = CONFIG_DIR / 'config.json'

def load_config() -> dict:
    """Load persisted config. Returns empty dict if not found."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_config(data: dict):
    """Save config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_config()
    existing.update(data)
    CONFIG_FILE.write_text(json.dumps(existing, indent=2))

def get_key(key: str, fallback: str = '') -> str:
    """Get a single config key."""
    cfg = load_config()
    return cfg.get(key) or fallback

def set_key(key: str, value: str):
    """Set a single config key."""
    save_config({key: value})
