"""
Shared utilities for all translators

Common functionality used across Aya23, MADLAD400, Seamless96, Helsinki, mBART translators
to reduce code duplication and maintain consistency.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple


def get_project_root() -> Path:
    """
    Get the project root directory (3 levels up from this file).

    Returns:
        Path: Project root directory
    """
    return Path(__file__).parent.parent.parent


def from_pretrained_cached(loader, model_path, **kwargs):
    """
    Load a model / tokenizer / processor offline-first.

    Tries the local HF cache (local_files_only=True) for a fast, network-free
    load. If a required file is missing from the cache (e.g. a processor's
    preprocessor_config.json that setup didn't fetch), falls back to an online
    load that downloads only the missing files. Keeps fully-cached models fast
    while self-healing partially-cached ones — and avoids the slow per-file HEAD
    checks that online loading does on every file.
    """
    try:
        return loader.from_pretrained(model_path, local_files_only=True, **kwargs)
    except Exception:
        return loader.from_pretrained(model_path, local_files_only=False, **kwargs)


def _merge_dicts(base: dict, overlay: dict) -> dict:
    """Merge two dicts. Dicts recurse, lists concatenate (base first), scalars use overlay."""
    merged = dict(base)
    for key, overlay_val in overlay.items():
        if key in merged:
            base_val = merged[key]
            if isinstance(base_val, dict) and isinstance(overlay_val, dict):
                merged[key] = {**base_val, **overlay_val}
            elif isinstance(base_val, list) and isinstance(overlay_val, list):
                merged[key] = base_val + overlay_val
            else:
                merged[key] = overlay_val
        else:
            merged[key] = overlay_val
    return merged


def load_glossary(lang_code: str, project_root: Optional[Path] = None) -> Optional[Dict]:
    """
    Load glossary for a language, merging base + uncensored when both exist.

    If {lang_code}_uncensored_glossary.yaml exists, it is merged with
    {lang_code}_glossary.yaml (uncensored entries take priority).
    Falls back to just the base glossary when no uncensored file exists.

    Args:
        lang_code: Language code (e.g., 'ro', 'es', 'fr')
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Merged dictionary with glossary terms or None if no glossary found
    """
    if project_root is None:
        project_root = get_project_root()

    base_path = project_root / "data" / f"{lang_code}_glossary.yaml"
    uncensored_path = project_root / "data" / f"{lang_code}_uncensored_glossary.yaml"

    base = {}
    if base_path.exists():
        with open(base_path, 'r', encoding='utf-8') as f:
            base = yaml.safe_load(f) or {}
        print(f"[OK] Using glossary: {lang_code}_glossary.yaml")

    if uncensored_path.exists():
        with open(uncensored_path, 'r', encoding='utf-8') as f:
            uncensored = yaml.safe_load(f) or {}
        print(f"[OK] Using glossary: {lang_code}_uncensored_glossary.yaml")
        return {**base, **uncensored}

    return base if base else None


def load_corrections(lang_code: str, project_root: Optional[Path] = None) -> Optional[Dict]:
    """
    Load corrections for a language, merging base + uncensored when both exist.

    If {lang_code}_uncensored_corrections.yaml exists, its entries are merged with
    {lang_code}_corrections.yaml (dicts merge with uncensored winning, lists concatenate).
    Falls back to just the base corrections when no uncensored file exists.

    Args:
        lang_code: Language code (e.g., 'ro', 'es', 'fr')
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Merged corrections dict or None if no corrections found
    """
    if project_root is None:
        project_root = get_project_root()

    base_path = project_root / "data" / f"{lang_code}_corrections.yaml"
    uncensored_path = project_root / "data" / f"{lang_code}_uncensored_corrections.yaml"

    base = {}
    if base_path.exists():
        with open(base_path, 'r', encoding='utf-8') as f:
            base = yaml.safe_load(f) or {}
        print(f"[OK] Using corrections: {lang_code}_corrections.yaml")

    if uncensored_path.exists():
        with open(uncensored_path, 'r', encoding='utf-8') as f:
            uncensored = yaml.safe_load(f) or {}
        print(f"[OK] Using corrections: {lang_code}_uncensored_corrections.yaml")
        return _merge_dicts(base, uncensored)

    return base if base else None


def get_language_code_map() -> Dict[str, str]:
    """
    Get mapping of language codes to language names.

    Returns:
        Dictionary mapping language codes to names
    """
    return {
        'ro': 'Romanian',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'tr': 'Turkish',
        'cs': 'Czech',
        'pl': 'Polish',
        'uk': 'Ukrainian',
        'bg': 'Bulgarian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'id': 'Indonesian',
        'ar': 'Arabic',
        'he': 'Hebrew',
        'fa': 'Persian',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'el': 'Greek',
        'hu': 'Hungarian'
    }


def parse_cli_language_arg() -> Tuple[Optional[str], Optional[str]]:
    """
    Parse --language argument from command line.

    Returns:
        Tuple of (language_name, language_code) or (None, None) if not found
    """
    import sys

    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--language' and i + 1 < len(sys.argv):
            lang_code = sys.argv[i + 1]
            lang_map = get_language_code_map()
            lang_name = lang_map.get(lang_code, lang_code.capitalize())
            return lang_name, lang_code

    return None, None


def load_prompt_template(lang_code: str, project_root: Optional[Path] = None) -> Optional[str]:
    """
    Load prompt template for a language with fallback hierarchy.

    Tries to load in this order:
    1. prompts/{lang_code}_uncensored_prompt.txt   (per-language uncensored override)
    2. prompts/{lang_code}_prompt.txt              (per-language override)
    3. prompts/translate_uncensored.txt            (generic uncensored — default)
    4. prompts/translate.txt                       (generic — default)

    Args:
        lang_code: Language code (e.g., 'ro', 'es', 'fr')
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Prompt template string or None if no template found
    """
    if project_root is None:
        project_root = get_project_root()

    prompt_variants = [
        f"prompts/{lang_code}_uncensored_prompt.txt",
        f"prompts/{lang_code}_prompt.txt",
        "prompts/translate_uncensored.txt",
        "prompts/translate.txt",
    ]

    for prompt_variant in prompt_variants:
        prompt_path = project_root / "data" / prompt_variant
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            print(f"[OK] Using prompt template: {prompt_variant}")
            return prompt_template

    return None


def load_models_config(project_root: Optional[Path] = None) -> Dict:
    """
    Load models configuration from models_config.yaml.

    Args:
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Dictionary with models configuration
    """
    if project_root is None:
        project_root = get_project_root()

    config_path = project_root / "models" / "models_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Models configuration not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_current_config(project_root: Optional[Path] = None) -> Dict:
    """
    Load current configuration from current_config.yaml.

    Args:
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Dictionary with current configuration
    """
    if project_root is None:
        project_root = get_project_root()

    config_path = project_root / "models" / "current_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Current configuration not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_sys_path():
    """
    Add parent directory to sys.path for imports.
    Useful for CLI entry points that need to import from src/.
    """
    import sys
    parent_dir = str(get_project_root() / "src")
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


def safe_generate(model, inputs: dict, device: str, generate_fn):
    """
    Call generate_fn(inputs) inside torch.no_grad().
    On a CUDA RuntimeError, moves model and inputs to CPU and retries once.
    Returns (outputs, model, device) — caller should reassign self.model / self.device.
    """
    import torch
    with torch.no_grad():
        try:
            return generate_fn(inputs), model, device
        except RuntimeError as e:
            if "cuda" in str(e).lower() and device != "cpu":
                print(f"  CUDA error during generate, retrying on CPU: {e}")
                model = model.to("cpu")
                device = "cpu"
                cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
                return generate_fn(cpu_inputs), model, device
            raise


_GLOSSARY_PAREN_SUFFIX = None  # compiled lazily to avoid top-level import cost


def _glossary_base_form(key: str) -> str:
    """Strip a trailing parenthetical disambiguator from a glossary key."""
    import re
    global _GLOSSARY_PAREN_SUFFIX
    if _GLOSSARY_PAREN_SUFFIX is None:
        _GLOSSARY_PAREN_SUFFIX = re.compile(r'\s*\([^)]*\)\s*$')
    return _GLOSSARY_PAREN_SUFFIX.sub('', key).strip()


def glossary_prompt_entries(glossary: dict, text: str, limit: int = 30) -> list:
    """Pick glossary entries to include in an LLM translation prompt.

    Matches on the base form of each key (parenthetical disambiguators stripped).
     When a base matches, ALL variants sharing that base
    are emitted together so the model sees the full conjugation table in context.

    Returns formatted '"en" = "target"' strings, capped at `limit` entries
    with longer (more specific) base forms prioritized first.
    """
    if not glossary:
        return []
    text_lower = text.lower()
    by_base = {}
    for en, tgt in glossary.items():
        en_str = str(en)
        if en_str.startswith("_comment"):
            continue
        if not isinstance(tgt, str):
            continue
        base = _glossary_base_form(en_str).lower()
        if not base:
            continue
        by_base.setdefault(base, []).append((en_str, tgt))

    matched = []
    for base in sorted(by_base.keys(), key=len, reverse=True):
        if base in text_lower:
            for en, tgt in by_base[base]:
                matched.append(f'"{en}" = "{tgt}"')
                if len(matched) >= limit:
                    return matched
    return matched


def apply_glossary(source_text: str, translation: str, glossary: dict) -> str:
    """
    Replace English terms left verbatim in translation with glossary targets.
    Only substitutes when the term appears in the source AND as a whole word
    in the translation (i.e. the model left it untranslated).
    """
    import re
    if not glossary:
        return translation
    for en_term, target_term in glossary.items():
        if en_term.startswith("_comment"):
            continue
        if not isinstance(target_term, str):
            continue
        if en_term.lower() not in source_text.lower():
            continue
        pattern = r'\b' + re.escape(en_term) + r'\b'
        if re.search(pattern, translation, flags=re.IGNORECASE):
            translation = re.sub(pattern, target_term, translation, flags=re.IGNORECASE)
    return translation


def apply_source_conditioned(source_text: str, translation: str, replacements) -> str:
    """
    Apply Romanian-side substitutions that depend on the English source.

    `replacements` is a list of dicts, each with `source_contains`, `wrong`,
    `right`. A replacement fires when `source_contains` appears anywhere in
    the source (case-insensitive) AND `wrong` appears as a whole word in the
    translation. Used to rescue HF translation models that never see the
    prompt-injected glossary and emit wrong domain terms.
    """
    import re
    if not replacements:
        return translation
    src_lower = source_text.lower()
    for entry in replacements:
        if not isinstance(entry, dict):
            continue
        en = entry.get("source_contains")
        wrong = entry.get("wrong")
        right = entry.get("right")
        if not (en and wrong and right):
            continue
        # Use word-boundary prefix match
        if not re.search(r'\b' + re.escape(en.lower()), src_lower):
            continue
        pattern = r'\b' + re.escape(wrong) + r'\b'
        if re.search(pattern, translation, flags=re.IGNORECASE):
            translation = re.sub(pattern, right, translation, flags=re.IGNORECASE)
    return translation


# Romanian class-III verbs: indicative 3sg → subjunctive 3sg (infinitive ends in -e)
_RO_SUBJ_PAIRS = [
    ('fute', 'fută'), ('suge', 'sugă'), ('linge', 'lingă'),
    ('merge', 'meargă'), ('vede', 'vadă'), ('spune', 'spună'),
    ('vine', 'vină'), ('face', 'facă'), ('bate', 'bată'),
    ('trage', 'tragă'), ('pune', 'pună'), ('împinge', 'împingă'),
    ('înghite', 'înghită'), ('strânge', 'strângă'), ('atinge', 'atingă'),
    ('prinde', 'prindă'), ('scoate', 'scoată'), ('înfinge', 'înfingă'),
]

# Optional clitic group between "să" and verb:
#   hyphenated (să-mi, să-l, să-și) or space-separated (să mă, să te, să mi-o)
_RO_CLITIC_OPT = (
    r'(?:'
    r'(?:-(?:mi|ți|i|l|o|ne|vă|le|și|mă|te))'
    r'|(?:\s+(?:mă|te|se|îl|o|îi|ne|vă|le|îmi|îți))'
    r'|(?:\s+(?:mi|ți|i|ni|vi|li)-(?:l|o|i))'
    r')?'
)

_RO_SUBJ_COMPILED: list = []


def apply_ro_subjunctive(translation: str) -> str:
    """Fix Romanian class-III indicative forms after 'să' to subjunctive.
    Handles clitics: 'să mă fute' → 'să mă fută', 'să-ți suge' → 'să-ți sugă'.
    """
    import re
    if not _RO_SUBJ_COMPILED:
        for ind, subj in _RO_SUBJ_PAIRS:
            pat = re.compile(
                rf'\b(?:să{_RO_CLITIC_OPT}|s-[oi])\s+{re.escape(ind)}\b',
                re.IGNORECASE
            )
            _RO_SUBJ_COMPILED.append((pat, len(ind), subj))
    for pat, ind_len, subj in _RO_SUBJ_COMPILED:
        translation = pat.sub(
            lambda m, n=ind_len, s=subj: m.group(0)[:-n] + s,
            translation
        )
    return translation


_BACK_MAP_CACHE: Dict[str, list] = {}


def back_map_for(target_language: str, project_root: Optional[Path] = None) -> list:
    """
    Return the source-conditioned replacement list for a target language name.
    Lazy-loaded from the corrections YAML; cached per language.
    Returns [] when no list is configured for that language.
    """
    name_to_code = {v: k for k, v in get_language_code_map().items()}
    lang_code = name_to_code.get(target_language)
    if not lang_code:
        return []
    if lang_code in _BACK_MAP_CACHE:
        return _BACK_MAP_CACHE[lang_code]
    corrections = load_corrections(lang_code, project_root) or {}
    replacements = corrections.get("source_conditioned_replacements") or []
    _BACK_MAP_CACHE[lang_code] = replacements
    return replacements


def probe_device() -> str:
    """
    Return 'cuda' if CUDA is available and kernels actually work, else 'cpu'.
    Some systems report CUDA as available but fail at kernel execution
    (driver/toolkit version mismatch). A cheap probe catches this early so
    models are loaded on the right device from the start.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        t = torch.zeros(2, 2, dtype=torch.float32, device="cuda")
        _ = torch.matmul(t, t)
        return "cuda"
    except RuntimeError:
        print("  CUDA available but kernels failed probe -- falling back to CPU")
        return "cpu"
