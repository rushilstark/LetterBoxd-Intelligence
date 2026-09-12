"""
config_store.py
Stores API keys persistently in ~/.culturaliq/config.json
Users enter keys ONCE — never asked again.
"""
import json
from pathlib import Path

CONFIG_DIR = Path.home() / '.movinator'
CONFIG_FILE = CONFIG_DIR / 'config.json'

_OLD_CONFIG_DIR = Path.home() / '.culturaliq'
_OLD_CONFIG_FILE = _OLD_CONFIG_DIR / 'config.json'

# Migrate from old location if needed
if _OLD_CONFIG_FILE.exists() and not CONFIG_FILE.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(_OLD_CONFIG_FILE), str(CONFIG_FILE))

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
