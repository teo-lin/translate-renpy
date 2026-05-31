"""
Modular Translation Pipeline — Hardware-Aware Version

Reads models/compute_profile.yaml (written by 0-setup.ps1) for tier-resolved
model params (n_ctx, n_batch, n_gpu_layers, quant, file path).
Supports llama models (aya23, ayaExpanse8b, euroLLM9b, euroLLM22b) and HF
seq2seq models (nllb200, madlad400, seamlessm96, helsinkiRo, mbartRo).

Usage:
    python translate.py --game <game_name>
    python translate.py  # Uses current_game from config
"""

import sys
import importlib
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import ParsedBlock, is_separator_block, parse_block_id
from translators.translator_utils import load_prompt_template
from hardware import load_profile
from renpy_utils import show_progress

_HF_TRANSLATORS = {
    'nllb200':    ('translators.nllb200_translator',    'NLLB200Translator'),
    'nllb1300':   ('translators.nllb200_translator',    'NLLB200Translator'),
    'madlad400':  ('translators.madlad400_translator',  'MADLAD400Translator'),
    'seamlessm96':('translators.seamless96_translator', 'SeamlessM4Tv2Translator'),
    'helsinkiRo': ('translators.helsinkyRo_translator', 'QuickMTTranslator'),
    'opusTCBig':  ('translators.helsinkyRo_translator', 'QuickMTTranslator'),
    'mbartRo':    ('translators.mbartRo_translator',    'MBARTTranslator'),
}
_LLAMA_MODELS = {'aya23', 'ayaExpanse8b', 'euroLLM9b', 'euroLLM22b'}


class ModularBatchTranslator:
    """
    Batch translator for parsed YAML files with custom context logic.

    - DIALOGUE blocks: Use context_before + context_after from surrounding blocks
    - CHOICE blocks: No dialogue context, only character info from characters.yaml
    """

    def __init__(
        self,
        translator,
        characters: Dict,
        target_lang_code: str,
        context_before: int = 3,
        context_after: int = 1,
        hf_batch_size: int = 1,
    ):
        self.translator = translator
        self.characters = characters
        self.target_lang_code = target_lang_code
        self.context_before = context_before
        self.context_after = context_after
        self.hf_batch_size = hf_batch_size

    def translate_file(
        self,
        parsed_yaml_path: Path,
        tags_yaml_path: Path,
        output_yaml_path: Optional[Path] = None
    ) -> Dict[str, int]:
        if output_yaml_path is None:
            output_yaml_path = parsed_yaml_path

        print(f"\n  Processing: {parsed_yaml_path.name}")

        with open(parsed_yaml_path, 'r', encoding='utf-8') as f:
            parsed_blocks: Dict[str, ParsedBlock] = yaml.safe_load(f)

        with open(tags_yaml_path, 'r', encoding='utf-8-sig') as f:
            tags_file = yaml.safe_load(f)

        metadata = tags_file['metadata']
        structure = tags_file['structure']
        block_order = structure['block_order']

        untranslated_ids = self._identify_untranslated(parsed_blocks, self.target_lang_code)
        total_blocks = len([bid for bid in parsed_blocks if not is_separator_block(bid, parsed_blocks[bid])])

        print(f"    Total blocks: {total_blocks}")
        print(f"    Untranslated: {len(untranslated_ids)}")
        print(f"    Already done: {total_blocks - len(untranslated_ids)}")

        if not untranslated_ids:
            print("    [OK] All blocks already translated!")
            return {'total': total_blocks, 'translated': 0, 'skipped': total_blocks, 'failed': 0}

        contexts = self._extract_contexts(untranslated_ids, parsed_blocks, block_order)

        has_batch = (
            self.hf_batch_size > 1
            and callable(getattr(self.translator, 'translate_batch', None))
        )
        if has_batch:
            print(f"    Translating {len(contexts)} blocks... (batch_size={self.hf_batch_size})")
        else:
            print(f"    Translating {len(contexts)} blocks...")
        translated_count = 0
        failed_count = 0
        start_time = time.time()

        if has_batch:
            for batch_start in range(0, len(contexts), self.hf_batch_size):
                batch = contexts[batch_start:batch_start + self.hf_batch_size]
                texts = [item['text'] for item in batch]
                show_progress(batch_start + len(batch), len(contexts), start_time, prefix="    ")
                try:
                    translations = self.translator.translate_batch(texts)
                    for item, translation in zip(batch, translations):
                        block_id = item['block_id']
                        if translated_count < 3:
                            print(f"\n    [DEBUG] Block {block_id}")
                            print(f"            EN: {item['text'][:40]}...")
                            print(f"            RO: {translation[:40]}...")
                        parsed_blocks[block_id][self.target_lang_code] = translation
                        translated_count += 1
                except Exception as e:
                    print(f"\n    [ERROR] Batch failed at {batch_start}: {e}")
                    failed_count += len(batch)
        else:
            for idx, context_item in enumerate(contexts, start=1):
                block_id = context_item['block_id']
                char_name = context_item['character_name']
                text_to_translate = context_item['text']
                context_list = context_item['context']
                show_progress(idx, len(contexts), start_time, prefix="    ")
                speaker = None if char_name in ['Narrator', 'Choice'] else char_name
                try:
                    translation = self.translator.translate(
                        text=text_to_translate,
                        context=context_list if context_list else None,
                        speaker=speaker
                    )
                    if translated_count < 3:
                        print(f"\n    [DEBUG] Block {block_id}")
                        print(f"            EN: {text_to_translate[:40]}...")
                        print(f"            RO: {translation[:40]}...")
                    parsed_blocks[block_id][self.target_lang_code] = translation
                    translated_count += 1
                except Exception as e:
                    print(f"\n    [ERROR] Translation failed for {block_id}: {e}")
                    failed_count += 1

        print()

        try:
            self._save_yaml(parsed_blocks, output_yaml_path, metadata)
            print(f"    [OK] Saved to: {output_yaml_path.name}")
        except Exception as e:
            print(f"    [ERROR] Failed to save YAML file: {e}")
            import traceback
            traceback.print_exc()
            failed_count += translated_count
            translated_count = 0

        stats = {
            'total': total_blocks,
            'translated': translated_count,
            'skipped': total_blocks - len(untranslated_ids),
            'failed': failed_count
        }
        print(f"    [OK] Translated: {stats['translated']}, Failed: {stats['failed']}")
        return stats

    def _identify_untranslated(self, parsed_blocks: Dict[str, ParsedBlock], lang_code: str) -> List[str]:
        untranslated = []
        for block_id, block in parsed_blocks.items():
            if is_separator_block(block_id, block):
                continue
            target_text = block.get(lang_code, '')
            if not target_text or not target_text.strip():
                untranslated.append(block_id)
        return untranslated

    def _extract_contexts(
        self,
        untranslated_ids: List[str],
        parsed_blocks: Dict[str, ParsedBlock],
        block_order: List[str]
    ) -> List[Dict]:
        contexts: List[Dict] = []
        block_index = {block_id: idx for idx, block_id in enumerate(block_order)}

        for block_id in untranslated_ids:
            idx = block_index.get(block_id)
            if idx is None:
                continue

            _, char_name = parse_block_id(block_id)
            is_choice = char_name == 'Choice' or block_id.endswith('-Choice')
            text_to_translate = parsed_blocks[block_id]['en']

            if is_choice:
                context_list = []
            else:
                context_list = self._extract_dialogue_context(block_id, idx, parsed_blocks, block_order)

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
        context_before: List[str] = []
        context_after: List[str] = []

        for i in range(idx - 1, max(-1, idx - self.context_before - 10), -1):
            if len(context_before) >= self.context_before:
                break
            prev_id = block_order[i]
            prev_block = parsed_blocks.get(prev_id)
            if not prev_block or is_separator_block(prev_id, prev_block):
                continue
            prev_text = prev_block.get(self.target_lang_code, '') or prev_block.get('en', '')
            if prev_text.strip():
                prev_char = parse_block_id(prev_id)[1]
                context_before.insert(0, f"{prev_char}: {prev_text}")

        for i in range(idx + 1, min(len(block_order), idx + self.context_after + 10)):
            if len(context_after) >= self.context_after:
                break
            next_id = block_order[i]
            next_block = parsed_blocks.get(next_id)
            if not next_block or is_separator_block(next_id, next_block):
                continue
            next_text = next_block.get(self.target_lang_code, '') or next_block.get('en', '')
            if next_text.strip():
                next_char = parse_block_id(next_id)[1]
                context_after.append(f"{next_char}: {next_text}")

        return context_before + context_after

    def _save_yaml(self, parsed_blocks: Dict[str, ParsedBlock], output_path: Path, metadata: dict):
        from datetime import datetime

        output_path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            f"# {output_path.stem} - Parsed Translations\n"
            f"# Original extraction: {metadata.get('extracted_at', 'unknown')}\n"
            f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "\n"
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            yaml.dump(parsed_blocks, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            f.flush()

        if not output_path.exists():
            raise IOError(f"File was not created: {output_path}")
        if output_path.stat().st_size == 0:
            raise IOError(f"File was created but is empty: {output_path}")


def load_config(project_root: Path, game_name: Optional[str] = None) -> Dict:
    config_file = project_root / "models" / "current_config.yaml"

    if not config_file.exists():
        print(f"ERROR: Configuration not found at {config_file}")
        print("Please run 1-config.ps1 first to set up your game.")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if game_name:
        if game_name not in config.get('games', {}):
            print(f"ERROR: Game '{game_name}' not found in configuration.")
            print(f"Available games: {', '.join(config.get('games', {}).keys())}")
            sys.exit(1)
        game_config = config['games'][game_name]
    else:
        current_game = config.get('current_game')
        if not current_game:
            print("ERROR: No current_game set in configuration.")
            print("Please run 1-config.ps1 first.")
            sys.exit(1)
        game_config = config['games'][current_game]

    return game_config


def load_resources(project_root: Path, game_config: Dict, target_lang_code: str):
    """Load glossary, corrections, and prompt template with merge fallback hierarchy."""
    # Glossary: load base first, then overlay uncensored on top (both merged)
    glossary = {}
    base_gloss_path = project_root / "data" / f"{target_lang_code}_glossary.yaml"
    uncensored_gloss_path = project_root / "data" / f"{target_lang_code}_uncensored_glossary.yaml"
    if base_gloss_path.exists():
        with open(base_gloss_path, 'r', encoding='utf-8-sig') as f:
            glossary = yaml.safe_load(f) or {}
        print(f"[OK] Using glossary: {target_lang_code}_glossary.yaml")
    if uncensored_gloss_path.exists():
        with open(uncensored_gloss_path, 'r', encoding='utf-8-sig') as f:
            glossary = {**glossary, **(yaml.safe_load(f) or {})}
        print(f"[OK] Using glossary: {target_lang_code}_uncensored_glossary.yaml")
    if not glossary:
        glossary = None
        print(f"[WARNING] No glossary found for language code '{target_lang_code}'")

    # Corrections: load base first, then deep-merge uncensored on top
    def _merge_corrections(base, overlay):
        merged = dict(base)
        for k, v in overlay.items():
            if k in merged:
                bv = merged[k]
                if isinstance(bv, dict) and isinstance(v, dict):
                    merged[k] = {**bv, **v}
                elif isinstance(bv, list) and isinstance(v, list):
                    merged[k] = bv + v
                else:
                    merged[k] = v
            else:
                merged[k] = v
        return merged

    corrections = {}
    base_corr_path = project_root / "data" / f"{target_lang_code}_corrections.yaml"
    uncensored_corr_path = project_root / "data" / f"{target_lang_code}_uncensored_corrections.yaml"
    if base_corr_path.exists():
        with open(base_corr_path, 'r', encoding='utf-8-sig') as f:
            corrections = yaml.safe_load(f) or {}
        print(f"[OK] Using corrections: {target_lang_code}_corrections.yaml")
    if uncensored_corr_path.exists():
        with open(uncensored_corr_path, 'r', encoding='utf-8-sig') as f:
            corrections = _merge_corrections(corrections, yaml.safe_load(f) or {})
        print(f"[OK] Using corrections: {target_lang_code}_uncensored_corrections.yaml")
    if not corrections:
        corrections = None

    # Prompt template: 4-variant fallback via translator_utils
    prompt_template = load_prompt_template(target_lang_code, project_root)
    if not prompt_template:
        print("[WARNING] No prompt template found, using default")

    return glossary, corrections, prompt_template


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Translate parsed YAML files — hardware-aware pipeline')
    parser.add_argument('--game', type=str, help='Game name (uses current_game from config if not specified)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    print("\n" + "=" * 70)
    print("  Modular Translation Pipeline (hardware-aware)")
    print("=" * 70)

    print("\nLoading compute profile...")
    try:
        compute_profile = load_profile()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Tier : {compute_profile['tier']}")
    print(f"  GPU  : {compute_profile.get('gpu', 'unknown')} ({compute_profile.get('vram_gb', 0)}GB)")

    print("\nLoading configuration...")
    game_config = load_config(project_root, args.game)

    game_name = game_config['name']
    game_path = Path(game_config['path'])
    target_language_obj = game_config['target_language']
    model_name = game_config['model']
    context_before = game_config.get('context_before', 3)
    context_after = game_config.get('context_after', 1)

    target_language_code = target_language_obj['code']
    target_language_name = target_language_obj['name']
    target_lang_code = target_language_code

    print(f"  Game    : {game_name}")
    print(f"  Language: {target_language_name} ({target_language_code})")
    print(f"  Model   : {model_name}")
    print(f"  Context : {context_before} before, {context_after} after")

    # Determine hf_batch_size from tier profile
    hf_batch_size = 8  # fallback default
    tier = compute_profile.get('tier', 'medium')
    compute_profiles_path = project_root / "models" / "compute_profiles.yaml"
    if compute_profiles_path.exists():
        with open(compute_profiles_path, 'r', encoding='utf-8') as f:
            compute_profiles = yaml.safe_load(f) or {}
        hf_batch_size = compute_profiles.get('profiles', {}).get(tier, {}).get('hf_batch_size', hf_batch_size)

    # Resolve model path and hw params — LLAMA from compute_profile.yaml, HF from models_config.yaml
    if model_name in _LLAMA_MODELS:
        model_profile = compute_profile.get("models", {}).get(model_name)
        if not model_profile:
            available = ', '.join(compute_profile.get("models", {}).keys())
            print(f"\nERROR: Model '{model_name}' is not available for tier '{compute_profile['tier']}'.")
            print(f"Available models in this tier: {available}")
            print("Check models/compute_profile.yaml or re-run 0-setup.ps1.")
            sys.exit(1)
        print(f"\n  n_ctx   : {model_profile['n_ctx']}")
        print(f"  n_batch : {model_profile['n_batch']}")
        print(f"  quant   : {model_profile['quant']}")
        model_path = project_root / model_profile["file"]

    elif model_name in _HF_TRANSLATORS:
        model_profile = compute_profile.get("models", {}).get(model_name)
        if model_profile and 'repo' in model_profile:
            model_path = model_profile['repo']
        else:
            models_config_path = project_root / "models" / "models_config.yaml"
            with open(models_config_path, 'r', encoding='utf-8') as f:
                all_models_config = yaml.safe_load(f)['available_models']
            hf_model_config = all_models_config.get(model_name)
            if not hf_model_config:
                print(f"ERROR: Model '{model_name}' not found in models_config.yaml")
                sys.exit(1)
            model_path = hf_model_config['repo']

    else:
        print(f"ERROR: Model '{model_name}' is not a recognized LLAMA or HF model.")
        print(f"  LLAMA models : {sorted(_LLAMA_MODELS)}")
        print(f"  HF models    : {sorted(_HF_TRANSLATORS)}")
        sys.exit(1)

    if not model_path.exists():
        print(f"\nERROR: Model file not found: {model_path}")
        print("Please run 0-setup.ps1 to download the model.")
        sys.exit(1)

    print("\nLoading resources...")
    glossary, corrections, prompt_template = load_resources(project_root, game_config, target_lang_code)

    tl_dir = game_path / "game" / "tl" / target_language_name.lower()
    characters_file = tl_dir / "characters.yaml"

    characters = {}
    if characters_file.exists():
        with open(characters_file, 'r', encoding='utf-8') as f:
            characters = yaml.safe_load(f)
        print(f"[OK] Loaded {len(characters)} characters from characters.yaml")
    else:
        print(f"[WARNING] No characters.yaml found at {characters_file}")

    print("\n" + "=" * 70)
    print("  Initializing Translator")
    print("=" * 70)

    if model_name in _LLAMA_MODELS:
        from translators.llama_cpp_translator import LlamaCppTranslator
        translator = LlamaCppTranslator(
            model_path=str(model_path),
            target_language=target_language_name.capitalize(),
            n_gpu_layers=model_profile["n_gpu_layers"],
            n_ctx=model_profile["n_ctx"],
            n_batch=model_profile["n_batch"],
            prompt_template=prompt_template,
            glossary=glossary,
        )
        hf_batch_size = 1  # llama models use single-item translate()

    else:  # _HF_TRANSLATORS
        module_path, class_name = _HF_TRANSLATORS[model_name]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        kwargs = dict(
            target_language=target_language_name.capitalize(),
            lang_code=target_lang_code,
            glossary=glossary,
        )
        if model_name == 'seamlessm96':
            kwargs['model_name'] = str(model_path)
        elif model_name != 'madlad400':
            kwargs['model_path'] = str(model_path)
        translator = cls(**kwargs)
        print(f"  Batch size: {hf_batch_size}")

    batch_translator = ModularBatchTranslator(
        translator=translator,
        characters=characters,
        target_lang_code=target_lang_code,
        context_before=context_before,
        context_after=context_after,
        hf_batch_size=hf_batch_size,
    )

    print("\n" + "=" * 70)
    print("  Scanning for Files")
    print("=" * 70)

    parsed_files = [f for f in tl_dir.glob("*.parsed.yaml") if '.translated.' not in f.name]

    if not parsed_files:
        print(f"\nERROR: No .parsed.yaml files found in {tl_dir}")
        print("Please run 2-extract.ps1 first to extract translation files.")
        sys.exit(1)

    print(f"\nFound {len(parsed_files)} file(s) to process")

    print("\n" + "=" * 70)
    print("  Translating Files")
    print("=" * 70)

    total_stats = {
        'files': 0,
        'total_blocks': 0,
        'translated_blocks': 0,
        'skipped_blocks': 0,
        'failed_blocks': 0
    }

    overall_start = time.time()

    for file_idx, parsed_file in enumerate(parsed_files, 1):
        if len(parsed_files) > 1:
            print()
            show_progress(file_idx - 1, len(parsed_files), overall_start, prefix="Overall: ")
            print(f"\n[File {file_idx}/{len(parsed_files)}]")

        base_name = parsed_file.name.removesuffix('.parsed.yaml')
        tags_file = parsed_file.parent / f"{base_name}.tags.yaml"
        if not tags_file.exists():
            print(f"\n  [WARNING] Skipping {parsed_file.name} - no matching .tags.yaml file")
            print(f"             Expected: {tags_file.name}")
            continue

        stats = batch_translator.translate_file(
            parsed_yaml_path=parsed_file,
            tags_yaml_path=tags_file,
            output_yaml_path=None
        )

        total_stats['files'] += 1
        total_stats['total_blocks'] += stats['total']
        total_stats['translated_blocks'] += stats['translated']
        total_stats['skipped_blocks'] += stats['skipped']
        total_stats['failed_blocks'] += stats['failed']

    print("\n" + "=" * 70)
    print("  TRANSLATION COMPLETE")
    print("=" * 70)
    print(f"  Files processed:     {total_stats['files']}")
    print(f"  Total blocks:        {total_stats['total_blocks']}")
    print(f"  Translated:          {total_stats['translated_blocks']}")
    print(f"  Already done:        {total_stats['skipped_blocks']}")
    print(f"  Failed:              {total_stats['failed_blocks']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
