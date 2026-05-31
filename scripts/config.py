"""
Character Discovery and Configuration Script
Discovers characters from game translation files and configures game settings
"""

import os
import sys
import re
from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config_selector import select_item


def show_banner():
    """Display the configuration banner"""
    print()
    print("=" * 70)
    print("         Character Discovery & Game Configuration          ")
    print("=" * 70)
    print()


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def select_game(current_game_name: Optional[str] = None) -> tuple[str, Path]:
    """
    Select a game from the games directory
    Returns: (game_name, game_path)
    """
    if current_game_name:
        print(f"[Config] Current configured game: {current_game_name}\n")

    games_dir = get_project_root() / "games"

    if not games_dir.exists():
        games_dir.mkdir(parents=True, exist_ok=True)

    game_dirs = [d for d in games_dir.iterdir() if d.is_dir()]

    if not game_dirs:
        print("   No games found in 'games' directory!")
        print(f"   Please add your game folders to: {games_dir}")
        sys.exit(1)

    # Sort game directories by name
    game_dirs = sorted(game_dirs, key=lambda d: d.name)

    def game_formatter(game_dir, index):
        return f"   [{index}] {game_dir.name}"

    selected_game = select_item(
        title="[Game] Available Games:",
        items=game_dirs,
        item_formatter_func=game_formatter,
        item_type_name="game",
        step_info=""
    )

    return selected_game.name, selected_game


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml_config(config_path: Path, data: Dict[str, Any]):
    """Save YAML configuration file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def select_language(current_language: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Select a language from available installed languages
    Returns: language object with 'name' and 'code' keys
    """
    if current_language:
        print(f"[Config] Current configured language: {current_language.get('name')} ({current_language.get('code')})\n")

    models_config_path = get_project_root() / "models" / "models_config.yaml"

    if not models_config_path.exists():
        print(f"Models configuration not found at {models_config_path}. Please run 0-setup.ps1.")
        sys.exit(1)

    models_config = load_yaml_config(models_config_path)

    current_config_path = get_project_root() / "models" / "current_config.yaml"
    if not current_config_path.exists():
        print("No languages configured. Please run 0-setup.ps1 and select languages.")
        sys.exit(1)
    current_config_languages = load_yaml_config(current_config_path)
    installed_languages = current_config_languages.get('installed_languages', [])

    if not installed_languages:
        print("No languages configured. Please run 0-setup.ps1 and select languages.")
        sys.exit(1)

    def language_formatter(lang, index):
        return f"   [{index}] {lang['name']} ({lang['code']})"

    selected_language = select_item(
        title="[Language] Available Languages:",
        items=installed_languages,
        item_formatter_func=language_formatter,
        item_type_name="language",
        step_info=""
    )

    return selected_language


def select_model(current_model: Optional[str] = None) -> str:
    """
    Select a model from available installed models
    Returns: model key (string)
    """
    models_config_path = get_project_root() / "models" / "models_config.yaml"

    if not models_config_path.exists():
        print(f"[ERROR] Models configuration not found at {models_config_path}")
        print("Please run 0-setup.ps1 first to configure models.")
        sys.exit(1)

    models_config = load_yaml_config(models_config_path)

    current_config_path = get_project_root() / "models" / "current_config.yaml"
    installed_models = []
    if current_config_path.exists():
        current_config = load_yaml_config(current_config_path)
        installed_models = current_config.get('installed_models', [])

    if not installed_models:
        print("[ERROR] No models are installed!")
        print("Please run 0-setup.ps1 first to install models.")
        sys.exit(1)

    # Build list of installed model info
    available_models = models_config.get('available_models', {})
    model_items = []

    for model_key in installed_models:
        model_config = available_models.get(model_key, {})
        if model_config:
            model_items.append({
                'key': model_key,
                'name': model_config.get('name', model_key),
                'size': model_config.get('size', ''),
                'description': model_config.get('description', '')
            })

    if current_model:
        current_model_config = available_models.get(current_model, {})
        if current_model_config:
            print(f"[Config] Current configured model: {current_model_config.get('name')} ({current_model})\n")
        else:
            print(f"[Config] Current configured model: {current_model}\n")

    def model_formatter(model, index):
        return f"   [{index}] {model['name']} ({model['size']})"

    selected_model = select_item(
        title="[Model] Installed Models:",
        items=model_items,
        item_formatter_func=model_formatter,
        item_type_name="model",
        step_info=""
    )

    return selected_model['key']


def discover_characters(tl_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Discover characters from .rpy files
    Returns: dictionary of character variables and their metadata
    """
    print()
    print("[Search] Discovering characters from .rpy files...")

    character_vars = {}
    character_files = {}  # Track which files each character appears in

    # Find source .rpy files, excluding generated merge outputs (*.translated.rpy)
    rpy_files = [f for f in tl_path.rglob("*.rpy") if '.translated.' not in f.name]

    for rpy_file in rpy_files:
        try:
            content = rpy_file.read_text(encoding='utf-8')
            file_name = rpy_file.stem

            # Find dialogue patterns: character_var "text"
            matches = re.finditer(r'^\s*(\w+)\s+"[^"\\]*(?:\\.[^"\\]*)*"', content, re.MULTILINE)

            for match in matches:
                char_var = match.group(1)

                # Skip if it's a keyword
                if re.match(r'^(translate|old|new)$', char_var):
                    continue

                if char_var not in character_vars:
                    character_vars[char_var] = {
                        'name': char_var.upper(),
                        'gender': 'neutral',
                        'type': 'supporting',
                        'description': ''
                    }
                    character_files[char_var] = []

                # Track this file for the character
                if file_name not in character_files[char_var]:
                    character_files[char_var].append(file_name)

        except Exception as e:
            print(f"   Warning: Could not read {rpy_file}: {e}")
            continue

    # Extract character names from script.rpy define statements
    print("   [Search] Extracting character names from script.rpy...")

    # Go up from tl_path to game root
    game_path = tl_path.parent.parent

    # Find script*.rpy files, excluding tl directory
    script_files = []
    for script_file in game_path.rglob("script*.rpy"):
        if "tl" not in script_file.parts:
            script_files.append(script_file)

    for script_file in script_files:
        try:
            content = script_file.read_text(encoding='utf-8')

            # Match: define var = Character('Name', ...) or Character(None)
            define_matches = re.finditer(
                r'define\s+(\w+)\s*=\s*Character\((?:[\'"](.+?)[\'"]|None)\s*[,)]',
                content
            )

            for match in define_matches:
                char_var = match.group(1)
                char_name = match.group(2) if match.group(2) else ""

                # Initialize character in character_vars if it doesn't exist
                if char_var not in character_vars:
                    character_vars[char_var] = {
                        'name': char_var.upper(),
                        'gender': 'neutral',
                        'type': 'supporting',
                        'description': ''
                    }

                # Handle special cases
                if char_var == "narrator" or char_name == "":
                    character_vars[char_var]['name'] = "Narrator"
                    character_vars[char_var]['type'] = "narrator"
                # Detect protagonist (common patterns: mc, u, player)
                elif re.match(r'^(mc|u|player)$', char_var) or re.match(r'^\[.*name.*\]$', char_name):
                    # Use proper name if not a placeholder
                    if not re.match(r'^\[.*\]$', char_name) and char_name:
                        character_vars[char_var]['name'] = char_name
                    else:
                        character_vars[char_var]['name'] = "MainCharacter"
                    character_vars[char_var]['type'] = "protagonist"
                # Regular characters
                elif not re.match(r'^\?+$|\[.*\]$', char_name) and char_name:
                    character_vars[char_var]['name'] = char_name
                    character_vars[char_var]['type'] = "main"

        except Exception as e:
            print(f"   Warning: Could not read {script_file}: {e}")
            continue

    # Generate descriptions based on file appearances
    for char_var in character_vars:
        if char_var in character_files:
            files = character_files[char_var]
            file_types = []

            # Categorize file types
            cell_files = [f for f in files if f.startswith('Cell')]
            room_files = [f for f in files if f.startswith('Room')]
            exped_files = [f for f in files if f.startswith('Exped')]
            chara_files = [f for f in files if f.startswith('Chara')]

            if cell_files:
                file_types.append("Cell character")
            if room_files:
                file_types.append("Room character")
            if exped_files:
                file_types.append("Expedition character")
            if chara_files:
                file_types.append("Character definition")

            if file_types:
                description = ", ".join(file_types) + f" (appears in {len(files)} files)"
            else:
                description = "Appears in: " + ", ".join(files[:3])

            character_vars[char_var]['description'] = description

    # Add special narrator character
    character_vars[""] = {
        'name': "narrator",
        'gender': "neutral",
        'type': "narrator",
        'description': "narration without character"
    }

    print(f"   Found {len(character_vars)} unique character variables")

    return character_vars


def save_configuration(
    game_name: str,
    game_path: Path,
    selected_language: Dict[str, str],
    model: str,
    characters: Dict[str, Dict[str, str]]
):
    """Save configuration to YAML files"""
    print()
    print("[Save] Saving configuration...")

    # Save to current_config.yaml
    config_path = get_project_root() / "models" / "current_config.yaml"

    # Load existing config or create new
    if config_path.exists():
        config = load_yaml_config(config_path)
    else:
        config = {
            'games': {},
            'current_game': None
        }

    # Ensure games dict exists
    if 'games' not in config:
        config['games'] = {}

    # Add/update game config
    game_config = {
        'name': game_name,
        'path': str(game_path),
        'target_language': selected_language,
        'source_language': 'english',
        'model': model,
        'context_before': 3,
        'context_after': 1
    }

    config['games'][game_name] = game_config
    config['current_game'] = game_name

    # Save config
    save_yaml_config(config_path, config)
    print(f"   [OK] Saved game config to: {config_path}")

    # Save characters.yaml
    language_name = selected_language.get('name', 'unknown').lower()
    print(f"   [Save] Saving characters for language: {language_name}")
    characters_path = game_path / "game" / "tl" / language_name / "characters.yaml"

    save_yaml_config(characters_path, characters)
    print(f"   [OK] Saved characters to: {characters_path}")


def main():
    """Main execution flow"""
    import argparse

    parser = argparse.ArgumentParser(description='Character Discovery and Game Configuration')
    parser.add_argument('--game-path', type=str, default='', help='Path to game directory')
    parser.add_argument('--language', type=str, default='', help='Language code')
    parser.add_argument('--model', type=str, default='', help='Model key')

    args = parser.parse_args()

    show_banner()

    # Load existing config to get current game
    config_path = get_project_root() / "models" / "current_config.yaml"
    existing_config = load_yaml_config(config_path)
    current_game_config = None

    if existing_config.get('current_game'):
        current_game = existing_config['current_game']
        current_game_config = existing_config.get('games', {}).get(current_game)

    # Step 1: Select Game
    if args.game_path:
        game_path = Path(args.game_path)
        game_name = game_path.name
    else:
        current_game_name = current_game_config.get('name') if current_game_config else None
        game_name, game_path = select_game(current_game_name)

    print(f"Selected game: {game_name}")
    print(f"   Path: {game_path}\n")

    # Step 2: Select Language
    if args.language:
        # Find language object by code
        current_config_path = get_project_root() / "models" / "current_config.yaml"
        current_config_for_lang = load_yaml_config(current_config_path)
        installed_languages = current_config_for_lang.get('installed_languages', [])

        selected_language = None
        for lang in installed_languages:
            if lang.get('code') == args.language:
                selected_language = lang
                break

        if not selected_language:
            print(f"ERROR: Invalid language code provided: {args.language}")
            sys.exit(1)
    else:
        current_language = current_game_config.get('target_language') if current_game_config else None
        selected_language = select_language(current_language)

    print(f"[Language] Selected language: {selected_language['name']} ({selected_language['code']})\n")

    # Step 3: Select Model
    if args.model:
        model = args.model
    else:
        current_model = current_game_config.get('model') if current_game_config else None
        model = select_model(current_model)

    # Display selected model with friendly name
    models_config_path = get_project_root() / "models" / "models_config.yaml"
    models_config = load_yaml_config(models_config_path)
    selected_model_config = models_config.get('available_models', {}).get(model, {})

    if selected_model_config:
        print(f"[Model] Selected model: {selected_model_config['name']} ({model})")
    else:
        print(f"[Model] Selected model: {model}")

    # Step 4: Discover Characters
    tl_path = game_path / "game" / "tl" / selected_language['code']
    characters = discover_characters(tl_path)

    # Step 5: Save Configuration
    save_configuration(game_name, game_path, selected_language, model, characters)

    print()
    print("[Done] Configuration complete!")
    print()
    print("[Note] You can now manually edit the characters.yaml file to:")
    print("   - Add proper character names")
    print("   - Set correct gender (male/female/neutral)")
    print("   - Update character types (main/protagonist/supporting)")
    print()
    language_code = selected_language['code']
    print(f"[Location] Characters file: {game_path / 'game' / 'tl' / language_code / 'characters.yaml'}")
    print()


if __name__ == "__main__":
    main()
