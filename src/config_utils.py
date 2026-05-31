"""
Shared configuration helpers for the modular pipeline.

Small utilities used by both the extract and merge entry points so the
game-config lookup lives in exactly one place.
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_game_config(game_name: str) -> Dict[str, Any]:
    """Load game configuration from current_config.yaml"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "models" / "current_config.yaml"

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please run 1-config.ps1 first to configure a game.")
        exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Get the game configuration
    games = config.get('games', {})
    if game_name not in games:
        print(f"Error: Game '{game_name}' not found in configuration.")
        print(f"Available games: {', '.join(games.keys())}")
        exit(1)

    return games[game_name]
