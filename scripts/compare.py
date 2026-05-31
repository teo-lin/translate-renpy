"""
Benchmark Translation Script

Translates ALL blocks in .parsed.yaml files for benchmarking purposes.
Unlike normal translation, this:
- Translates ALL blocks regardless of existing translations
- Saves translations under abbreviated model keys (e.g., ay, he, ma) for comparison

Usage:
    python compare.py --game <game_name> --model <model_key> --key <key_id>
    Example: python compare.py --game Example --model aya23 --key ay
"""

import sys
import yaml
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import ParsedBlock, is_separator_block, parse_block_id
from translators.aya23_translator import Aya23Translator
from translators.helsinkyRo_translator import QuickMTTranslator
from translators.llama_cpp_translator import LlamaCppTranslator
from translators.madlad400_translator import MADLAD400Translator
from translators.mbartRo_translator import MBARTTranslator
from translators.nllb200_translator import NLLB200Translator
from translators.seamless96_translator import SeamlessM4Tv2Translator
from translators.translator_utils import load_prompt_template
from renpy_utils import show_progress

# Short column keys written into parsed YAMLs. Overrides where first-2-chars
# would collide (e.g. aya23 and ayaExpanse8b both start with "ay").
# Keep this in sync with benchmark.py's _KEY_OVERRIDES.
MODEL_KEY_OVERRIDES: dict[str, str] = {
    'ayaExpanse8b': 'ae',
    'euroLLM9b':    'eu',
    'euroLLM22b':   'el',
    'nllb1300':     'nb',
    'opusTCBig':    'tc',
    'seamlessm96':  'se',
}


class BenchmarkTranslator:
    """
    Benchmark translator that translates ALL blocks and saves to numbered keys.
    """

    def __init__(
        self,
        translator,
        characters: Dict,
        save_key: str,  # e.g., "01", "02", "03"
        context_before: int = 3,
        context_after: int = 1
    ):
        self.translator = translator
        self.characters = characters
        self.save_key = save_key
        self.context_before = context_before
        self.context_after = context_after

    def translate_file(
        self,
        parsed_yaml_path: Path,
        tags_yaml_path: Path,
        output_yaml_path: Optional[Path] = None
    ) -> Dict[str, int]:
        """
        Translate ALL blocks in a parsed YAML file.

        Args:
            parsed_yaml_path: Path to .parsed.yaml file
            tags_yaml_path: Path to .tags.yaml file
            output_yaml_path: Path to output YAML (default: overwrite input)

        Returns:
            Dict with statistics: {'total', 'translated', 'failed'}
        """
        if output_yaml_path is None:
            output_yaml_path = parsed_yaml_path

        print(f"\n  Processing: {parsed_yaml_path.name}")

        # Load files
        with open(parsed_yaml_path, 'r', encoding='utf-8') as f:
            parsed_blocks: Dict[str, ParsedBlock] = yaml.safe_load(f)

        with open(tags_yaml_path, 'r', encoding='utf-8') as f:
            tags_file = yaml.safe_load(f)

        metadata = tags_file['metadata']
        structure = tags_file['structure']
        block_order = structure['block_order']

        # Only translate blocks that don't already have this key (enables resuming)
        all_block_ids = [
            bid for bid in parsed_blocks
            if not is_separator_block(bid, parsed_blocks[bid])
            and not str(parsed_blocks[bid].get(self.save_key, '')).strip()
        ]
        total_blocks = len([bid for bid in parsed_blocks if not is_separator_block(bid, parsed_blocks[bid])])

        print(f"    Total blocks: {total_blocks}  |  Missing key '{self.save_key}': {len(all_block_ids)}")

        if not all_block_ids:
            print("    [OK] No blocks to translate!")
            return {
                'total': 0,
                'translated': 0,
                'failed': 0
            }

        # Extract context for each block
        contexts = self._extract_contexts(
            all_block_ids,
            parsed_blocks,
            block_order
        )

        # Translate blocks with progress tracking
        print(f"    Translating {len(contexts)} blocks...")
        translated_count = 0
        failed_count = 0
        start_time = time.time()

        for idx, context_item in enumerate(contexts, start=1):
            block_id = context_item['block_id']
            char_name = context_item['character_name']
            text_to_translate = context_item['text']
            context_list = context_item['context']
            is_choice = context_item['is_choice']

            # Show progress
            show_progress(idx, len(contexts), start_time, prefix="    ")

            # Get speaker for context
            speaker = None if char_name in ['Narrator', 'Choice'] else char_name

            try:
                # Translate
                translation = self.translator.translate(
                    text=text_to_translate,
                    context=context_list if context_list else None,
                    speaker=speaker
                )

                # Update block with numbered key
                parsed_blocks[block_id][self.save_key] = translation
                translated_count += 1

            except Exception as e:
                print(f"\n    [ERROR] Translation failed for {block_id}: {e}")
                failed_count += 1

        # Clear progress line
        print()

        # Save updated YAML
        try:
            self._save_yaml(parsed_blocks, output_yaml_path, metadata)
            print(f"    [OK] Saved to: {output_yaml_path.name}")

        except Exception as e:
            print(f"    [ERROR] Failed to save YAML file: {e}")
            import traceback
            traceback.print_exc()
            failed_count += translated_count
            translated_count = 0

        # Return statistics
        stats = {
            'total': total_blocks,
            'translated': translated_count,
            'failed': failed_count
        }

        print(f"    [OK] Translated: {stats['translated']}, Failed: {stats['failed']}")

        return stats

    def _extract_contexts(
        self,
        block_ids: List[str],
        parsed_blocks: Dict[str, ParsedBlock],
        block_order: List[str]
    ) -> List[Dict]:
        """Extract context for each block."""
        contexts: List[Dict] = []

        # Build index map: block_id -> position in order
        block_index = {block_id: idx for idx, block_id in enumerate(block_order)}

        for block_id in block_ids:
            idx = block_index.get(block_id)
            if idx is None:
                continue

            # Parse block ID to get character name
            _, char_name = parse_block_id(block_id)

            # Check if this is a CHOICE block
            is_choice = char_name == 'Choice' or block_id.endswith('-Choice')

            # Get text to translate
            text_to_translate = parsed_blocks[block_id]['en']

            # Extract context based on block type
            if is_choice:
                context_list = []
            else:
                context_list = self._extract_dialogue_context(
                    block_id, idx, parsed_blocks, block_order
                )

            contexts.append({
                'block_id': block_id,
                'character_name': char_name,
                'text': text_to_translate,
                'context': context_list,
                'is_choice': is_choice
            })

        return contexts

    def _extract_dialogue_context(
        self,
        block_id: str,
        idx: int,
        parsed_blocks: Dict[str, ParsedBlock],
        block_order: List[str]
    ) -> List[str]:
        """Extract dialogue context from surrounding blocks."""
        context_before: List[str] = []
        context_after: List[str] = []

        # Extract context before (use English only for consistency)
        for i in range(idx - 1, max(-1, idx - self.context_before - 10), -1):
            if len(context_before) >= self.context_before:
                break

            prev_id = block_order[i]
            prev_block = parsed_blocks.get(prev_id)

            if not prev_block or is_separator_block(prev_id, prev_block):
                continue

            prev_text = prev_block.get('en', '')
            if prev_text.strip():
                prev_char = parse_block_id(prev_id)[1]
                context_before.insert(0, f"{prev_char}: {prev_text}")

        # Extract context after (use English only for consistency)
        for i in range(idx + 1, min(len(block_order), idx + self.context_after + 10)):
            if len(context_after) >= self.context_after:
                break

            next_id = block_order[i]
            next_block = parsed_blocks.get(next_id)

            if not next_block or is_separator_block(next_id, next_block):
                continue

            next_text = next_block.get('en', '')
            if next_text.strip():
                next_char = parse_block_id(next_id)[1]
                context_after.append(f"{next_char}: {next_text}")

        return context_before + context_after

    def _save_yaml(
        self,
        parsed_blocks: Dict[str, ParsedBlock],
        output_path: Path,
        metadata: dict
    ):
        """Save parsed blocks to YAML file with header."""
        from datetime import datetime

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create header
        header = (
            f"# {output_path.stem} - Benchmark Translations\n"
            f"# Original extraction: {metadata.get('extracted_at', 'unknown')}\n"
            f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "\n"
        )

        # Remove ALL 'ro' keys from all blocks (only for compare.py flow)
        # In benchmark mode, we only want model comparison keys (ay, he, ma, mb, se)
        cleaned_blocks = {}
        for block_id, block_data in parsed_blocks.items():
            cleaned_block = {}
            for key, value in block_data.items():
                # Skip all 'ro' keys in compare flow
                if key == 'ro':
                    continue
                cleaned_block[key] = value
            cleaned_blocks[block_id] = cleaned_block

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            yaml.dump(cleaned_blocks, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            f.flush()

        # Verify file was created and has content
        if not output_path.exists():
            raise IOError(f"File was not created: {output_path}")

        file_size = output_path.stat().st_size
        if file_size == 0:
            raise IOError(f"File was created but is empty: {output_path}")


def load_config(project_root: Path, game_name: str) -> Dict:
    """Load game configuration from current_config.yaml."""
    config_file = project_root / "models" / "current_config.yaml"

    if not config_file.exists():
        print(f"ERROR: Configuration not found at {config_file}")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if game_name not in config.get('games', {}):
        print(f"ERROR: Game '{game_name}' not found in configuration.")
        sys.exit(1)

    return config['games'][game_name]


def load_resources(project_root: Path, target_lang_code: str):
    """Load glossary and prompt template."""
    # Load glossary: merge base + uncensored when both exist
    glossary = {}
    base_gloss = project_root / "data" / f"{target_lang_code}_glossary.yaml"
    uncensored_gloss = project_root / "data" / f"{target_lang_code}_uncensored_glossary.yaml"
    if base_gloss.exists():
        with open(base_gloss, 'r', encoding='utf-8') as f:
            glossary = yaml.safe_load(f) or {}
        print(f"[OK] Using glossary: {target_lang_code}_glossary.yaml")
    if uncensored_gloss.exists():
        with open(uncensored_gloss, 'r', encoding='utf-8') as f:
            glossary = {**glossary, **(yaml.safe_load(f) or {})}
        print(f"[OK] Using glossary: {target_lang_code}_uncensored_glossary.yaml")
    if not glossary:
        glossary = None

    # Load prompt template (per-language override -> generic fallback)
    prompt_template = load_prompt_template(target_lang_code, project_root)

    return glossary, prompt_template


def _resolve_model_file(project_root: Path, model_key: str, model_config: dict) -> Path:
    """Resolve the actual file path for a model (handles multi-quant GGUF and directory models)."""
    dest_path = project_root / model_config['destination']
    if 'files' not in model_config:
        return dest_path
    profile_path = project_root / 'models' / 'compute_profile.yaml'
    if profile_path.exists():
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = yaml.safe_load(f)
        file_rel = profile.get('models', {}).get(model_key, {}).get('file')
        if file_rel:
            return project_root / file_rel
    files = model_config['files']
    filename = files.get('Q4_K_M') or files.get('Q3_K_M') or next(iter(files.values()))
    return dest_path / filename


def _load_profile_params(project_root: Path, model_key: str) -> dict:
    """Load llama-cpp inference params from compute_profile.yaml."""
    profile_path = project_root / 'models' / 'compute_profile.yaml'
    if profile_path.exists():
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = yaml.safe_load(f)
        params = profile.get('models', {}).get(model_key, {})
        if params:
            return {
                'n_gpu_layers': params.get('n_gpu_layers', -1),
                'n_ctx': params.get('n_ctx', 8192),
                'n_batch': params.get('n_batch', 256),
            }
    return {'n_gpu_layers': -1, 'n_ctx': 8192, 'n_batch': 256}


def initialize_translator(model_key: str, model_path: Path, target_language: str, glossary: dict, prompt_template: str, project_root: Path = None, lang_code: str = None):
    """Initialize the appropriate translator based on model key."""
    # Map model keys to translator classes
    translator_map = {
        'aya23': Aya23Translator,
        'helsinkiRo': QuickMTTranslator,
        'opusTCBig': QuickMTTranslator,
        'madlad400': MADLAD400Translator,
        'mbartRo': MBARTTranslator,
        'nllb200': NLLB200Translator,
        'nllb1300': NLLB200Translator,
        'seamlessm96': SeamlessM4Tv2Translator,
        'ayaExpanse8b': LlamaCppTranslator,
        'euroLLM9b': LlamaCppTranslator,
    }

    translator_class = translator_map.get(model_key)
    if not translator_class:
        print(f"ERROR: Unknown model key: {model_key}")
        sys.exit(1)

    # Initialize translator with appropriate parameters
    if model_key == 'aya23':
        # Aya23 uses GGUF model file
        return translator_class(
            model_path=str(model_path),
            target_language=target_language,
            prompt_template=prompt_template,
            glossary=glossary
        )
    elif model_key in ['helsinkiRo', 'opusTCBig', 'mbartRo']:
        # HuggingFace models - use model directory, not file
        return translator_class(
            model_path=str(model_path),
            target_language=target_language,
            glossary=glossary
        )
    elif model_key == 'madlad400':
        # MADLAD400 doesn't use model_path parameter
        return translator_class(
            target_language=target_language,
            glossary=glossary,
            trust_remote_code=True
        )
    elif model_key == 'seamlessm96':
        return translator_class(
            model_name=str(model_path),
            target_language=target_language,
            glossary=glossary
        )
    elif model_key in ('nllb200', 'nllb1300'):
        return translator_class(
            model_path=str(model_path),
            target_language=target_language,
            lang_code=lang_code or 'ro',
            glossary=glossary
        )
    elif model_key in ('ayaExpanse8b', 'euroLLM9b'):
        profile_params = _load_profile_params(project_root, model_key) if project_root else {}
        return translator_class(
            model_path=str(model_path),
            target_language=target_language,
            prompt_template=prompt_template,
            glossary=glossary,
            **profile_params
        )
    else:
        return translator_class(target_language=target_language)


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark translation - translate ALL blocks')
    parser.add_argument('--game', type=str, default=None, help='Game name (not needed with --tl-dir)')
    parser.add_argument('--model', type=str, required=True, help='Model key (e.g., aya23, helsinkiRo)')
    parser.add_argument('--key', type=str, required=True, help='Save key (e.g., r0, r1, r2)')
    parser.add_argument('--tl-dir', type=str, default=None,
                        help='Direct path to directory with .parsed.yaml files (bypasses config lookup)')
    parser.add_argument('--lang-code', type=str, default='ro')
    parser.add_argument('--lang-name', type=str, default='Romanian')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent

    # Load configuration
    print("\n" + "=" * 70)
    print(f"  Benchmark Translation - Model: {args.model}, Key: {args.key}")
    print("=" * 70)

    if args.tl_dir:
        tl_dir = Path(args.tl_dir)
        target_language_code = args.lang_code
        target_language_name = args.lang_name
        context_before = 3
        context_after = 1
        print(f"\n  Language: {target_language_name} ({target_language_code})")
        print(f"  Model: {args.model}")
        print(f"  Save key: {args.key}")
    else:
        if not args.game:
            parser.error("--game is required when --tl-dir is not specified")
        print("\nLoading configuration...")
        game_config = load_config(project_root, args.game)
        game_path = Path(game_config['path'])
        target_language_obj = game_config['target_language']
        context_before = game_config.get('context_before', 3)
        context_after = game_config.get('context_after', 1)
        target_language_code = target_language_obj['code']
        target_language_name = target_language_obj['name']
        tl_dir = game_path / "game" / "tl" / target_language_name.lower()
        print(f"  Game: {game_config['name']}")
        print(f"  Language: {target_language_name} ({target_language_code})")
        print(f"  Model: {args.model}")
        print(f"  Save key: {args.key}")

    # Load model configuration
    models_config_path = project_root / "models" / "models_config.yaml"
    with open(models_config_path, 'r', encoding='utf-8') as f:
        all_models_config = yaml.safe_load(f)['available_models']

    model_config = all_models_config.get(args.model)
    if not model_config:
        print(f"ERROR: Model '{args.model}' not found in models_config.yaml")
        sys.exit(1)

    model_path = _resolve_model_file(project_root, args.model, model_config)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Load resources
    print("\nLoading resources...")
    glossary, prompt_template = load_resources(project_root, target_language_code)

    # Load characters.yaml
    characters_file = tl_dir / "characters.yaml"

    characters = {}
    if characters_file.exists():
        with open(characters_file, 'r', encoding='utf-8') as f:
            characters = yaml.safe_load(f)
        print(f"[OK] Loaded {len(characters)} characters")

    # Initialize translator
    print("\n" + "=" * 70)
    print("  Initializing Translator")
    print("=" * 70)

    translator = initialize_translator(
        args.model,
        model_path,
        target_language_name.capitalize(),
        glossary,
        prompt_template,
        project_root=project_root,
        lang_code=target_language_code,
    )

    # Initialize benchmark translator
    benchmark_translator = BenchmarkTranslator(
        translator=translator,
        characters=characters,
        save_key=args.key,
        context_before=context_before,
        context_after=context_after
    )

    # Find all .parsed.yaml files
    print("\n" + "=" * 70)
    print("  Scanning for Files")
    print("=" * 70)

    parsed_files = list(tl_dir.glob("*.parsed.yaml"))

    if not parsed_files:
        print(f"\nERROR: No .parsed.yaml files found in {tl_dir}")
        sys.exit(1)

    print(f"\nFound {len(parsed_files)} file(s) to process")

    # Translate all files
    print("\n" + "=" * 70)
    print("  Translating Files")
    print("=" * 70)

    total_stats = {
        'files': 0,
        'total_blocks': 0,
        'translated_blocks': 0,
        'failed_blocks': 0
    }

    overall_start = time.time()

    for file_idx, parsed_file in enumerate(parsed_files, 1):
        # Show overall progress
        if len(parsed_files) > 1:
            print()
            show_progress(file_idx - 1, len(parsed_files), overall_start, prefix="Overall: ")
            print(f"\n[File {file_idx}/{len(parsed_files)}]")

        # Find corresponding tags.yaml file
        base_name = parsed_file.name.removesuffix('.parsed.yaml')
        tags_file = parsed_file.parent / f"{base_name}.tags.yaml"
        if not tags_file.exists():
            print(f"\n  [WARNING] Skipping {parsed_file.name} - no matching .tags.yaml file")
            continue

        # Translate file
        stats = benchmark_translator.translate_file(
            parsed_yaml_path=parsed_file,
            tags_yaml_path=tags_file,
            output_yaml_path=None
        )

        total_stats['files'] += 1
        total_stats['total_blocks'] += stats['total']
        total_stats['translated_blocks'] += stats['translated']
        total_stats['failed_blocks'] += stats['failed']

    # Calculate total duration
    total_duration = time.time() - overall_start

    # Final summary
    print("\n" + "=" * 70)
    print("  BENCHMARK TRANSLATION COMPLETE")
    print("=" * 70)
    print(f"  Model:               {args.model}")
    print(f"  Key:                 {args.key}")
    print(f"  Files processed:     {total_stats['files']}")
    print(f"  Total blocks:        {total_stats['total_blocks']}")
    print(f"  Translated:          {total_stats['translated_blocks']}")
    print(f"  Failed:              {total_stats['failed_blocks']}")
    print(f"  Duration:            {total_duration:.2f} seconds")
    print("=" * 70)

    # Return duration for PowerShell to capture
    print(f"\nBENCHMARK_DURATION:{total_duration:.2f}")


def _select_game_interactive() -> str:
    """Prompt the user to pick a game from the games/ folder."""
    from config_selector import select_item

    project_root = Path(__file__).parent.parent
    games_dir = project_root / "games"
    if not games_dir.exists():
        print(f"ERROR: games directory not found: {games_dir}")
        sys.exit(1)

    games = sorted(
        ({"name": d.name} for d in games_dir.iterdir() if d.is_dir()),
        key=lambda g: g["name"].lower(),
    )
    if not games:
        print(f"ERROR: No games found under {games_dir}")
        sys.exit(1)

    selected = select_item(
        title="Step 1: Select Game to Compare",
        items=games,
        item_formatter_func=lambda g, i: f"  [{i}] {g['name']}",
        item_type_name="game",
    )
    return selected["name"]


def run_full_comparison(game_name: str, language: str, model_filter: list = None) -> int:
    """
    Run the full model comparison workflow.

    Args:
        game_name: Name of the game to compare
        language: Target language code (e.g., 'ro')

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import os
    from datetime import datetime

    project_root = Path(__file__).parent.parent

    print()
    print("=" * 70)
    print("                 Translation Model Comparison")
    print("=" * 70)
    print()

    # Step 1: Load models configuration
    print("[1/5] Loading models configuration...")
    models_config_path = project_root / "models" / "models_config.yaml"
    current_config_path = project_root / "models" / "current_config.yaml"

    if not models_config_path.exists():
        print(f"ERROR: Models configuration not found at {models_config_path}")
        print("Please run 0-setup.ps1 first to install models.")
        return 1

    with open(models_config_path, 'r', encoding='utf-8') as f:
        models_config = yaml.safe_load(f)

    if current_config_path.exists():
        with open(current_config_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)
        installed_models = current_config.get('installed_models', [])
    else:
        installed_models = []

    if not installed_models:
        print("ERROR: No models are installed!")
        print("Please run 0-setup.ps1 first to install models.")
        return 1

    if model_filter:
        unknown = [m for m in model_filter if m not in models_config['available_models']]
        if unknown:
            print(f"ERROR: Unknown model key(s): {', '.join(unknown)}")
            return 1
        installed_models = [m for m in model_filter if m in installed_models]
        if not installed_models:
            print(f"ERROR: None of the requested models are installed: {', '.join(model_filter)}")
            return 1

    print(f"   Found {len(installed_models)} installed models:")
    for model_key in installed_models:
        model_info = models_config['available_models'][model_key]
        print(f"      - {model_info['name']}")
    print()

    # Get full game path (needed for steps 2-3 and for locating parsed YAMLs)
    full_game_path = project_root / "games" / game_name
    if not full_game_path.exists():
        print(f"ERROR: Game directory not found: {full_game_path}")
        return 1

    if model_filter:
        # Skip configure + extract: preserve all existing translations in the parsed YAML
        print("[2/5] Skipping configuration (model filter mode — preserving existing translations)")
        print("[3/5] Skipping extraction (model filter mode — parsed YAML already exists)")
        print()
    else:
        # Step 2: Configure game
        print(f"[2/5] Configuring game: {game_name} with language: {language}...")
        first_model = installed_models[0]
        print(f"   Using initial model: {first_model}")

        config_script = project_root / "1-config.ps1"
        result = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File",
             str(config_script), "-GamePath", str(full_game_path),
             "-Language", language, "-Model", first_model],
            capture_output=True,
            text=True
        )

        # Check if config file exists
        current_config_path = project_root / "models" / "current_config.yaml"
        if not current_config_path.exists():
            print("ERROR: Configuration failed - config file not created!")
            return 1
        print("   [OK] Configuration successful")
        print()

        # Step 3: Extract translation files
        print("[3/5] Extracting translation files...")
        extract_script = project_root / "2-extract.ps1"
        result = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File",
             str(extract_script), "-GameName", game_name, "-All"],
            capture_output=False
        )

        if result.returncode != 0:
            print("ERROR: Extraction failed!")
            return 1
        print()

    # Step 4: Compare each model
    print("[4/5] Running comparison translations...")
    print()

    python_exe = project_root / "venv" / "Scripts" / "python.exe"
    compare_script = project_root / "scripts" / "compare.py"

    if not python_exe.exists():
        print(f"ERROR: Python executable not found at {python_exe}")
        return 1

    # Set up environment
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    # Add PyTorch lib directory to PATH for CUDA DLLs
    torch_lib_path = project_root / "venv" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib_path.exists():
        env['PATH'] = f"{torch_lib_path};{env.get('PATH', '')}"

    # Track results
    benchmark_results = []
    benchmark_start_time = time.time()

    for model_idx, model_key in enumerate(installed_models):
        model_info = models_config['available_models'][model_key]
        key_number = MODEL_KEY_OVERRIDES.get(model_key, model_key[:2].lower())

        print()
        print(f"   [{model_idx + 1}/{len(installed_models)}] Model: {model_info['name']} -> Key: {key_number}")
        print("   " + ("=" * 65))

        model_start_time = time.time()

        # Run comparison translation
        result = subprocess.run(
            [str(python_exe), str(compare_script),
             "--game", game_name, "--model", model_key, "--key", key_number],
            env=env,
            capture_output=True,
            text=True
        )

        # Display output
        if result.stdout:
            print(result.stdout)

        model_end_time = time.time()
        model_duration = model_end_time - model_start_time

        if result.returncode != 0:
            print(f"   [ERROR] Translation failed for model {model_key}!")
            if result.stderr:
                print(result.stderr)
            benchmark_results.append({
                'model': model_info['name'],
                'key': key_number,
                'duration': model_duration,
                'status': 'FAILED',
                'size': model_info.get('size', 'N/A'),
                'params': model_info.get('params', 'N/A')
            })
        else:
            # Try to extract duration from output
            import re
            match = re.search(r'BENCHMARK_DURATION:(\d+\.?\d*)', result.stdout)
            if match:
                actual_duration = float(match.group(1))
            else:
                actual_duration = model_duration

            print(f"   [OK] Completed in {actual_duration:.2f} seconds")

            benchmark_results.append({
                'model': model_info['name'],
                'key': key_number,
                'duration': actual_duration,
                'status': 'SUCCESS',
                'size': model_info.get('size', 'N/A'),
                'params': model_info.get('params', 'N/A')
            })

    benchmark_end_time = time.time()
    total_duration = benchmark_end_time - benchmark_start_time

    print()
    print()

    # Step 5: Display comparison results
    print("[5/5] Benchmark Results")
    print()
    print("=" * 70)
    print("                   MODEL COMPARISON")
    print("=" * 70)
    print()

    # Sort by duration (fastest first)
    sorted_results = sorted(benchmark_results, key=lambda x: x.get('duration', float('inf')))

    print(f"   {'Key':<3} {'Model':<20} {'Size':<10} {'Duration':<12} {'Status':<10}")
    print("   " + ("-" * 65))

    for result in sorted_results:
        status_color = result['status']
        duration_str = f"{result['duration']:.2f}s" if result['status'] == 'SUCCESS' else "N/A"

        print(f"   {result['key']:<3} {result['model']:<20} {result['size']:<10} {duration_str:<12} {status_color:<10}")

    print()
    print(f"   Total benchmark duration: {total_duration:.2f} seconds")
    print()

    # Find fastest and slowest
    successful = [r for r in benchmark_results if r['status'] == 'SUCCESS']
    if len(successful) > 1:
        fastest = min(successful, key=lambda x: x['duration'])
        slowest = max(successful, key=lambda x: x['duration'])

        speedup = slowest['duration'] / fastest['duration']

        print(f"   Fastest: {fastest['model']} ({fastest['duration']:.2f}s)")
        print(f"   Slowest: {slowest['model']} ({slowest['duration']:.2f}s)")
        print(f"   Speedup: {speedup:.2f}x faster")
        print()

    print("=" * 70)
    print()
    print("   Benchmark complete!")
    print()
    print("   Translation files saved with numbered keys (ay, he, ma, etc.)")
    print("   Review the .parsed.yaml files to compare model outputs.")
    print()
    print(f"   Location: games\\{game_name}\\game\\tl\\romanian\\*.parsed.yaml")
    print()
    print("   Each block now contains:")
    print("      en:  Original English text")
    for result in benchmark_results:
        print(f"      {result['key']}: Translation from {result['model']}")
    print()
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    import argparse

    # Check if running in orchestrator mode
    if len(sys.argv) > 1 and sys.argv[1] == "orchestrate":
        parser = argparse.ArgumentParser(description='Run full model comparison workflow')
        parser.add_argument('_', help='orchestrate command')
        parser.add_argument('--game', type=str, default=None,
                            help='Game name (prompts interactively if omitted)')
        parser.add_argument('--language', type=str, required=True, help='Language code (e.g., ro)')
        parser.add_argument('--models', type=str, default=None,
                            help='Comma-separated model keys to run (e.g. euroLLM22b,opusTCBig); runs all if omitted')
        args = parser.parse_args()

        game_name = args.game or _select_game_interactive()
        model_filter = [m.strip() for m in args.models.split(',')] if args.models else None
        exit_code = run_full_comparison(game_name, args.language, model_filter)
        sys.exit(exit_code)
    else:
        # Regular single-model mode
        main()
