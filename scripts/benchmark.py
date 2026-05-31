"""
Translation Quality Benchmark using BLEU Scores

Benchmarks translation quality by comparing model outputs to reference translations
using BLEU (Bilingual Evaluation Understudy) scores.

Benchmark data format (YAML):
- source: English text to translate
  target: Reference translation
  alt_targets:                            # optional: equally-valid alternates
    - "Another acceptable translation"
    - "Yet another"
  context: Optional previous dialogue for context
- source: Another text...
  target: Another translation...

Usage:
    python benchmark.py data/ro_benchmark.yaml [--glossary data/ro_glossary.yaml]
    python benchmark.py data/de_benchmark.yaml --glossary data/de_glossary.yaml
"""

import sys
import os
import contextlib
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import time
import re

# Try to import BLEU from nltk
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("WARNING: nltk not installed. Install with: pip install nltk")
    print("Falling back to simple word-match accuracy")

# Try to import chrF from sacrebleu
try:
    from sacrebleu.metrics import CHRF as _CHRF
    _chrf_metric = _CHRF(word_order=2)  # chrF++
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

# Try to import METEOR from nltk (requires WordNet corpus)
try:
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    import nltk as _nltk_lib
    _nltk_lib.data.find('corpora/wordnet')
    METEOR_AVAILABLE = True
except (ImportError, LookupError):
    METEOR_AVAILABLE = False

# COMET lazy loader — checks HF cache before loading; never triggers a download
_comet_model = None
_comet_load_attempted = False

# Mirror of compare.py's MODEL_KEY_OVERRIDES — keep both in sync.
_KEY_OVERRIDES: dict[str, str] = {
    'ayaExpanse8b': 'ae',
    'euroLLM9b':    'eu',
    'euroLLM22b':   'el',
    'nllb1300':     'nb',
    'opusTCBig':    'tc',
    'seamlessm96':  'se',
}


def _gpu_count() -> int:
    try:
        import torch
        return 1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def _build_model_name_map(project_root: Path) -> dict[str, str]:
    """Return {short_key: model_name} for every model in models_config.yaml."""
    cfg_path = project_root / 'models' / 'models_config.yaml'
    if not cfg_path.exists():
        return {}
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    result = {}
    for full_key, info in cfg.get('available_models', {}).items():
        short_key = _KEY_OVERRIDES.get(full_key, full_key[:2].lower())
        result[short_key] = info.get('name', full_key)
    return result


@contextlib.contextmanager
def _quiet():
    """Redirect stdout+stderr to null at the fd level, silencing all third-party noise."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fds = [os.dup(1), os.dup(2)]
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    old_streams = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_streams
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in saved_fds:
            os.close(fd)
        os.close(devnull_fd)


def _get_comet_model():
    global _comet_model, _comet_load_attempted
    if _comet_load_attempted:
        return _comet_model
    _comet_load_attempted = True

    # Silence HF progress bars before download_model is called
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        with _quiet():
            from comet import download_model, load_from_checkpoint
    except ImportError:
        print("INFO: unbabel-comet not installed — COMET scoring disabled.")
        return None

    # COMET lives in the default HF cache (populated by 0-setup.ps1)
    default_cache = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        / "hub" / "models--Unbabel--wmt22-comet-da"
    )
    if not default_cache.exists():
        print("INFO: COMET model not cached — run 0-setup.ps1 to download it.")
        return None

    try:
        with _quiet():
            model_path = download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(model_path)
        return _comet_model
    except Exception as e:
        print(f"WARNING: COMET model failed to load: {e}")
        return None


# Fix Windows PATH for CUDA DLLs
if sys.platform == "win32":
    import os
    torch_lib = str(Path(__file__).parent.parent / "venv" / "Lib" / "site-packages" / "torch" / "lib")
    if os.path.exists(torch_lib) and torch_lib not in os.environ["PATH"]:
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def tokenize(text: str) -> List[str]:
    """Simple tokenization: split on whitespace and punctuation"""
    # Remove Ren'Py tags and variables for fair comparison
    text = re.sub(r'\{[^}]+\}', '', text)  # Remove {color=...}, etc.
    text = re.sub(r'\[[^\]]+\]', '', text)  # Remove [name], etc.

    # Tokenize
    return text.lower().split()


def calculate_bleu(references, hypothesis: str) -> float:
    """
    Calculate BLEU score between hypothesis and one or more references.

    `references` may be a single string or a list of strings; in the list case
    this is standard multi-reference BLEU (max n-gram overlap per position),
    so adding alternates can only raise the score.

    Returns score between 0.0 (worst) and 1.0 (perfect match).
    """
    if isinstance(references, str):
        references = [references]

    if not NLTK_AVAILABLE:
        # Fallback: max single-reference word overlap
        hyp_words = set(tokenize(hypothesis))
        best = 0.0
        for ref in references:
            ref_words = set(tokenize(ref))
            if not ref_words:
                continue
            best = max(best, len(ref_words & hyp_words) / len(ref_words))
        return best

    # NLTK BLEU with smoothing (for short sentences)
    ref_token_lists = [tokenize(r) for r in references]
    hyp_tokens = tokenize(hypothesis)

    smoothing = SmoothingFunction().method1
    return sentence_bleu(ref_token_lists, hyp_tokens, smoothing_function=smoothing)


def compound_score(bleu: float, chrf: float, comet: float | None, meteor: float | None = None) -> float | None:
    """
    0.60*COMET + 0.25*chrF++ + 0.10*METEOR + 0.05*BLEU.
    Returns None if any metric is unavailable — Score column is omitted rather than computed with reduced weights.
    """
    if comet is None or meteor is None:
        return None
    return 0.60 * comet + 0.25 * chrf + 0.10 * meteor + 0.05 * bleu


def calculate_chrf(references, hypothesis: str) -> float:
    """
    Calculate chrF score (character n-gram F-score, the FLORES metric).
    Returns score between 0.0 (worst) and 1.0 (perfect match).
    Falls back to 0.0 if sacrebleu is not installed.
    """
    if isinstance(references, str):
        references = [references]
    if not SACREBLEU_AVAILABLE:
        return 0.0
    return _chrf_metric.sentence_score(hypothesis, references).score / 100.0


def calculate_meteor(references, hypothesis: str) -> float | None:
    """
    Calculate METEOR score with stemming and synonym matching via WordNet.
    Returns None if nltk or WordNet data is unavailable.
    """
    if not METEOR_AVAILABLE:
        return None
    if isinstance(references, str):
        references = [references]
    hyp_tokens = tokenize(hypothesis)
    ref_token_lists = [tokenize(r) for r in references]
    try:
        return _nltk_meteor(ref_token_lists, hyp_tokens)
    except LookupError:
        return None


def load_benchmark_data(data_path: Path) -> List[Dict]:
    """Load benchmark data from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Benchmark data must be a JSON array")

    # Validate format
    for i, item in enumerate(data):
        if 'source' not in item or 'target' not in item:
            raise ValueError(f"Item {i} missing 'source' or 'target' field")
        if 'alt_targets' in item:
            alts = item['alt_targets']
            if not isinstance(alts, list) or not all(isinstance(t, str) for t in alts):
                raise ValueError(f"Item {i}: 'alt_targets' must be a list of strings")

    return data


def load_glossary(glossary_path: Path) -> Dict:
    """Load glossary from JSON file"""
    if not glossary_path.exists():
        return {}

    with open(glossary_path, 'r', encoding='utf-8') as f:
        glossary = yaml.safe_load(f)

    # Filter out comment entries (starting with _)
    return {k: v for k, v in glossary.items() if not k.startswith('_')}


def detect_language_from_filename(filename: str) -> str:
    """
    Detect language from filename (e.g., 'ro_benchmark.yaml' → 'Romanian')
    """
    lang_map = {
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
    }

    # Extract language code from filename
    for code, lang in lang_map.items():
        if filename.startswith(f'{code}_') or f'_{code}_' in filename:
            return lang

    # Default to Romanian
    return 'Romanian'


def _load_profile_params(project_root: Path, model_key: str) -> dict:
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


def detect_lang_code_from_filename(filename: str) -> str:
    for code in ['ro', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'tr', 'cs', 'pl',
                 'uk', 'bg', 'zh', 'ja', 'ko', 'vi', 'th', 'id', 'ar', 'he',
                 'fa', 'hi', 'bn', 'nl', 'sv', 'da', 'fi', 'el', 'hu']:
        if filename.startswith(f'{code}_') or f'_{code}_' in filename:
            return code
    return 'ro'


def run_benchmark(data_path: Path, glossary_path: Path = None, model_key: str = "aya23") -> Dict:
    """
    Run translation quality benchmark

    Returns:
        Statistics dict with scores and examples
    """
    print("=" * 70)
    print("Translation Quality Benchmark")
    print("=" * 70)

    # Detect target language
    target_language = detect_language_from_filename(data_path.name)
    print(f"\nTarget language: {target_language}")
    print(f"Benchmark data: {data_path}")

    # Load data
    print("\nLoading benchmark data...")
    benchmark_data = load_benchmark_data(data_path)
    print(f"  Loaded {len(benchmark_data)} test cases")

    # Load glossary
    glossary = {}
    if glossary_path and glossary_path.exists():
        print(f"\nLoading glossary: {glossary_path}")
        glossary = load_glossary(glossary_path)
        print(f"  Loaded {len(glossary)} terms")

    # Load model configuration
    project_root = Path(__file__).parent.parent
    models_config_path = project_root / "models" / "models_config.yaml"

    with open(models_config_path, 'r', encoding='utf-8') as f:
        models_config = yaml.safe_load(f)

    model_info = models_config['available_models'].get(model_key)
    if not model_info:
        print(f"ERROR: Model '{model_key}' not found in models_config.yaml")
        sys.exit(1)

    from hardware import resolve_model_path

    print(f"\nModel: {model_info['name']}")
    print(f"Initializing translator...")

    lang_code = detect_lang_code_from_filename(data_path.name)

    # Create translator based on model type. Glossary is passed to the constructor
    # (none of the translate() methods accept it as a kwarg).
    if model_key == "aya23":
        from translators.aya23_translator import Aya23Translator
        model_path = resolve_model_path(model_key)
        translator = Aya23Translator(str(model_path), target_language=target_language, glossary=glossary)
    elif model_key in ("helsinkyRo", "helsinkiRo", "opusTCBig"):
        from translators.helsinkyRo_translator import QuickMTTranslator
        model_path = resolve_model_path(model_key)
        translator = QuickMTTranslator(model_path=str(model_path), target_language=target_language, glossary=glossary)
    elif model_key == "madlad400":
        from translators.madlad400_translator import MADLAD400Translator
        translator = MADLAD400Translator(target_language=target_language, glossary=glossary)
    elif model_key == "mbartRo":
        from translators.mbartRo_translator import MBARTTranslator
        model_path = resolve_model_path(model_key)
        translator = MBARTTranslator(model_path=str(model_path), target_language=target_language, glossary=glossary)
    elif model_key in ("nllb200", "nllb1300"):
        from translators.nllb200_translator import NLLB200Translator
        model_path = resolve_model_path(model_key)
        translator = NLLB200Translator(model_path=str(model_path), target_language=target_language, lang_code=lang_code, glossary=glossary)
    elif model_key in ("seamlessm96", "seamless96"):
        from translators.seamless96_translator import SeamlessM4Tv2Translator
        model_path = resolve_model_path(model_key)
        translator = SeamlessM4Tv2Translator(model_name=str(model_path), target_language=target_language, glossary=glossary)
    elif model_key in ("ayaExpanse8b", "euroLLM9b"):
        from translators.llama_cpp_translator import LlamaCppTranslator
        model_path = resolve_model_path(model_key)
        profile_params = _load_profile_params(project_root, model_key)
        translator = LlamaCppTranslator(model_path=str(model_path), target_language=target_language, glossary=glossary, **profile_params)
    else:
        print(f"ERROR: Model '{model_key}' not supported for benchmarking")
        sys.exit(1)

    # Run translations and calculate scores
    print("\n" + "=" * 70)
    print("Running translations...")
    print("=" * 70)

    bleu_scores = []
    chrf_scores = []
    meteor_scores_per_item = []
    comet_inputs = []
    results = []
    t_start = time.time()

    for i, item in enumerate(benchmark_data, 1):
        source = item['source']
        reference = item['target']
        alt_targets = item.get('alt_targets') or []
        references = [reference] + alt_targets
        context = item.get('context', None)

        print(f"\n[{i}/{len(benchmark_data)}]")
        print(f"  Source: {source}")

        # Parse context (can be string or list)
        context_list = None
        if context:
            if isinstance(context, str):
                context_list = [context]
            elif isinstance(context, list):
                context_list = context

        # Translate (glossary was already passed to the translator constructor)
        hypothesis = translator.translate(
            source,
            context=context_list
        )

        bleu   = calculate_bleu(references, hypothesis)
        chrf   = calculate_chrf(references, hypothesis)
        meteor = calculate_meteor(references, hypothesis)
        bleu_scores.append(bleu)
        chrf_scores.append(chrf)
        if meteor is not None:
            meteor_scores_per_item.append(meteor)
        comet_inputs.append({'src': source, 'mt': hypothesis, 'ref': reference})

        print(f"  Reference:  {reference}")
        for alt in alt_targets:
            print(f"  Alt:        {alt}")
        print(f"  Hypothesis: {hypothesis}")
        meteor_str = f"  METEOR: {meteor:.4f}" if meteor is not None else ""
        print(f"  BLEU: {bleu:.4f}  chrF: {chrf:.4f}{meteor_str}")

        results.append({
            'source': source,
            'reference': reference,
            'hypothesis': hypothesis,
            'score': bleu,
            'chrf': chrf,
            'meteor': meteor,
        })

    # Batch COMET scoring
    comet_scores = []
    comet_model = _get_comet_model()
    if comet_model is not None:
        print("\nRunning COMET predictions...")
        try:
            with _quiet():
                output = comet_model.predict(
                    comet_inputs, batch_size=16, gpus=_gpu_count(), progress_bar=False
                )
            comet_scores = output.scores
            for res, cs in zip(results, comet_scores):
                res['comet'] = cs
        except Exception as e:
            print(f"WARNING: COMET prediction failed: {e}")

    duration_s = round(time.time() - t_start, 2)

    avg_bleu   = sum(bleu_scores)  / len(bleu_scores)  if bleu_scores  else 0.0
    min_bleu   = min(bleu_scores)                       if bleu_scores  else 0.0
    max_bleu   = max(bleu_scores)                       if bleu_scores  else 0.0
    avg_chrf   = sum(chrf_scores)  / len(chrf_scores)   if chrf_scores  else 0.0
    avg_meteor = sum(meteor_scores_per_item) / len(meteor_scores_per_item) if meteor_scores_per_item else None
    avg_comet  = sum(comet_scores) / len(comet_scores)  if comet_scores else None
    avg_cmpd   = compound_score(avg_bleu, avg_chrf, avg_comet, avg_meteor)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal test cases: {len(benchmark_data)}")
    print(f"Average BLEU:     {avg_bleu:.4f}")
    print(f"Average chrF++:   {avg_chrf:.4f}")
    if avg_meteor is not None:
        print(f"Average METEOR:   {avg_meteor:.4f}")
    if avg_comet is not None:
        print(f"Average COMET:    {avg_comet:.4f}")
    if avg_cmpd is not None:
        print(f"Average Score:    {avg_cmpd:.4f}")
    print(f"Duration:         {duration_s}s")

    # Show best and worst examples
    sorted_results = sorted(results, key=lambda x: x['score'])

    print("\n" + "-" * 70)
    print("WORST TRANSLATION:")
    worst = sorted_results[0]
    print(f"  Source:     {worst['source']}")
    print(f"  Reference:  {worst['reference']}")
    print(f"  Hypothesis: {worst['hypothesis']}")
    print(f"  BLEU: {worst['score']:.4f}  chrF: {worst['chrf']:.4f}")

    print("\n" + "-" * 70)
    print("BEST TRANSLATION:")
    best = sorted_results[-1]
    print(f"  Source:     {best['source']}")
    print(f"  Reference:  {best['reference']}")
    print(f"  Hypothesis: {best['hypothesis']}")
    print(f"  BLEU: {best['score']:.4f}  chrF: {best['chrf']:.4f}")

    print("\n" + "=" * 70)

    stats = {
        'total':        len(benchmark_data),
        'average_bleu': avg_bleu,
        'min_bleu':     min_bleu,
        'max_bleu':     max_bleu,
        'average_chrf': avg_chrf,
        'duration_s':   duration_s,
        'results':      results,
    }
    if avg_cmpd is not None:
        stats['average_score'] = avg_cmpd
    if avg_meteor is not None:
        stats['average_meteor'] = avg_meteor
    if avg_comet is not None:
        stats['average_comet'] = avg_comet
        stats['min_comet']     = min(comet_scores)
        stats['max_comet']     = max(comet_scores)
    return stats


def _save_benchmark_result(project_root: Path, model_key: str, data_path: Path, stats: dict) -> Path:
    """Append this run's summary to models/benchmarks.yaml and return the file path."""
    out_path = project_root / "models" / "benchmarks.yaml"

    existing = []
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, list):
                existing = loaded

    name_map = _build_model_name_map(project_root)
    short_key = _KEY_OVERRIDES.get(model_key, model_key[:2].lower())

    record: dict = {
        'model':    short_key,
        'name':     name_map.get(short_key, model_key),
        'lines':    stats['total'],
        'avg_bleu': round(stats['average_bleu'], 4),
        'avg_chrf': round(stats.get('average_chrf', 0.0), 4),
    }
    if 'average_meteor' in stats:
        record['avg_meteor'] = round(stats['average_meteor'], 4)
    if 'average_comet' in stats:
        record['avg_comet'] = round(stats['average_comet'], 4)
    if 'average_score' in stats:
        record['avg_score'] = round(stats['average_score'], 4)

    existing.append(record)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return out_path


def run_score_parsed(parsed_path: Path, benchmark_path: Path, project_root: Path) -> None:
    """
    Score all model columns in a parsed YAML against benchmark references.
    No model loading — uses existing translations only.
    Saves one record per model to models/benchmarks.yaml.
    """
    print("=" * 70)
    print("Score from parsed YAML (no re-translation)")
    print("=" * 70)
    print(f"\nParsed file: {parsed_path}")
    print(f"References:  {benchmark_path}")

    # Load reference data: source text → {target, alt_targets}
    benchmark_data = load_benchmark_data(benchmark_path)
    ref_map = {item['source']: item for item in benchmark_data}
    print(f"\nLoaded {len(ref_map)} reference items")

    # Load parsed YAML
    with open(parsed_path, 'r', encoding='utf-8') as f:
        parsed = yaml.safe_load(f)

    # Collect all model keys present across blocks (anything that isn't 'en')
    model_keys = set()
    for block in parsed.values():
        if isinstance(block, dict):
            model_keys.update(k for k in block if k != 'en')
    model_keys = sorted(model_keys)
    print(f"Model columns found: {', '.join(model_keys)}")

    # Score each block where en matches a benchmark reference
    model_item_bleu:   dict[str, list] = {k: [] for k in model_keys}
    model_item_chrf:   dict[str, list] = {k: [] for k in model_keys}
    model_item_meteor: dict[str, list] = {k: [] for k in model_keys}
    model_comet_data:  dict[str, list] = {k: [] for k in model_keys}
    model_worst: dict[str, dict] = {}
    model_best:  dict[str, dict] = {}
    matched_blocks = 0

    for block_id, block in parsed.items():
        if not isinstance(block, dict):
            continue
        en = block.get('en', '').strip()
        if en not in ref_map:
            continue
        matched_blocks += 1
        ref_item = ref_map[en]
        references = [ref_item['target']] + (ref_item.get('alt_targets') or [])

        for key in model_keys:
            hyp = block.get(key, '')
            if not hyp:
                continue
            hyp_str = str(hyp)
            bleu   = calculate_bleu(references, hyp_str)
            chrf   = calculate_chrf(references, hyp_str)
            meteor = calculate_meteor(references, hyp_str)
            entry = {'block': block_id, 'source': en,
                     'reference': ref_item['target'], 'hypothesis': hyp_str, 'score': bleu}
            model_item_bleu[key].append(bleu)
            model_item_chrf[key].append(chrf)
            if meteor is not None:
                model_item_meteor[key].append(meteor)
            model_comet_data[key].append({'src': en, 'mt': hyp_str, 'ref': ref_item['target']})
            if key not in model_worst or bleu < model_worst[key]['score']:
                model_worst[key] = entry
            if key not in model_best or bleu > model_best[key]['score']:
                model_best[key] = entry

    # Batch COMET predictions per model (skipped if model not cached)
    comet_model = _get_comet_model()
    model_item_comet: dict[str, list] = {}
    if comet_model is not None:
        gpus = _gpu_count()
        print("Running COMET predictions...")
        for key in model_keys:
            data = model_comet_data[key]
            if not data:
                continue
            try:
                with _quiet():
                    output = comet_model.predict(data, batch_size=16, gpus=gpus, progress_bar=False)
                model_item_comet[key] = output.scores
            except Exception as e:
                print(f"  WARNING: COMET failed for '{key}': {e}")

    print(f"\nMatched {matched_blocks} blocks against references\n")

    use_comet  = bool(model_item_comet)
    use_meteor = any(model_item_meteor.get(k) for k in model_keys)
    name_map = _build_model_name_map(project_root)

    records = []
    for key in model_keys:
        bleus   = model_item_bleu[key]
        chrfs   = model_item_chrf[key]
        meteors = model_item_meteor.get(key) or []
        comets  = model_item_comet.get(key)
        if not bleus:
            continue
        avg_bleu   = round(sum(bleus)   / len(bleus),   4)
        avg_chrf   = round(sum(chrfs)   / len(chrfs),   4) if chrfs   else 0.0
        avg_meteor = round(sum(meteors) / len(meteors), 4) if meteors else None
        avg_comet  = round(sum(comets)  / len(comets),  4) if comets  else None
        avg_cmpd   = compound_score(avg_bleu, avg_chrf, avg_comet, avg_meteor)

        record: dict = {
            'model':    key,
            'name':     name_map.get(key, key),
            'lines':    len(bleus),
            'avg_bleu': avg_bleu,
            'avg_chrf': avg_chrf,
        }
        if avg_meteor is not None:
            record['avg_meteor'] = avg_meteor
        if avg_comet is not None:
            record['avg_comet'] = avg_comet
        if avg_cmpd is not None:
            record['avg_score'] = round(avg_cmpd, 4)
        records.append(record)

    records.sort(key=lambda r: r.get('avg_score', r.get('avg_comet', r.get('avg_meteor', r['avg_chrf']))), reverse=True)

    table_lines = []
    if use_comet and use_meteor:
        table_lines.append(f"{'Model':<8}  {'BLEU':>7}  {'chrF++':>7}  {'METEOR':>7}  {'COMET':>7}  {'Score':>7}  {'Name'}")
        table_lines.append("-" * 70)
        for r in records:
            score_str = f"{r['avg_score']:>7.4f}" if 'avg_score' in r else f"{'N/A':>7}"
            table_lines.append(f"{r['model']:<8}  {r['avg_bleu']:>7.4f}  {r['avg_chrf']:>7.4f}  {r['avg_meteor']:>7.4f}  {r['avg_comet']:>7.4f}  {score_str}  {r['name']}")
    elif use_comet:
        table_lines.append(f"{'Model':<8}  {'BLEU':>7}  {'chrF++':>7}  {'COMET':>7}  {'Name'}")
        table_lines.append("-" * 50)
        for r in records:
            table_lines.append(f"{r['model']:<8}  {r['avg_bleu']:>7.4f}  {r['avg_chrf']:>7.4f}  {r['avg_comet']:>7.4f}  {r['name']}")
    elif use_meteor:
        table_lines.append(f"{'Model':<8}  {'BLEU':>7}  {'chrF++':>7}  {'METEOR':>7}  {'Name'}")
        table_lines.append("-" * 50)
        for r in records:
            table_lines.append(f"{r['model']:<8}  {r['avg_bleu']:>7.4f}  {r['avg_chrf']:>7.4f}  {r['avg_meteor']:>7.4f}  {r['name']}")
    else:
        table_lines.append(f"{'Model':<8}  {'BLEU':>7}  {'chrF++':>7}  {'Name'}")
        table_lines.append("-" * 36)
        for r in records:
            table_lines.append(f"{r['model']:<8}  {r['avg_bleu']:>7.4f}  {r['avg_chrf']:>7.4f}  {r['name']}")

    for line in table_lines:
        print(line)
    print()

    # Save visual table to benchmark_scores.yaml
    scores_path = project_root / "models" / "benchmark_scores.yaml"
    with open(scores_path, 'w', encoding='utf-8') as f:
        f.write('table: |\n')
        for line in table_lines:
            f.write('  ' + line + '\n')

    # Append all records to benchmarks.yaml
    out_path = project_root / "models" / "benchmarks.yaml"
    existing = []
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, list):
                existing = loaded
    existing.extend(records)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"Results saved to:      {out_path}")
    print(f"Scores table saved to: {scores_path}")


def _auto_detect_glossary(data_path: Path) -> Path:
    lang_code = data_path.stem.split('_')[0]
    uncensored = data_path.parent / f"{lang_code}_uncensored_glossary.yaml"
    regular = data_path.parent / f"{lang_code}_glossary.yaml"
    if uncensored.exists():
        print(f"Auto-detected glossary: {uncensored}")
        return uncensored
    if regular.exists():
        print(f"Auto-detected glossary: {regular}")
        return regular
    return None


def _select_model_interactive(installed_models: list, available_models: dict) -> str:
    print()
    print("=" * 65)
    print("       Step 1: Select Model to Benchmark")
    print("=" * 65)
    print()
    for i, key in enumerate(installed_models, 1):
        info = available_models.get(key, {})
        name = info.get('name', key)
        params = info.get('params', '?')
        size = info.get('size', '?')
        print(f"  [{i}] {name} ({params}, {size})")
    print("  [Q] Quit")
    print()
    while True:
        choice = input(f"Select a model (1-{len(installed_models)} or Q): ").strip()
        if choice.lower() == 'q':
            print("Cancelled by user.")
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(installed_models):
                return installed_models[idx]
        except ValueError:
            pass
        print(f"Invalid selection. Please enter a number between 1 and {len(installed_models)}.")


def run_orchestrate():
    """Interactive orchestrator: load YAML config, prompt for model, run benchmark."""
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark a translation model with BLEU scoring')
    parser.add_argument('_', help='orchestrate command')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Benchmark YAML file (auto-detected if not specified: '
                             'data/<lang>_uncensored_benchmark.yaml -> data/<lang>_benchmark.yaml)')
    parser.add_argument('--lang', type=str, default='ro',
                        help='Language code for benchmark auto-detection (default: ro)')
    parser.add_argument('--glossary', type=str, default=None,
                        help='Glossary YAML file (auto-detected if not specified)')
    parser.add_argument('--model-key', type=str, default=None,
                        help='Model key (e.g., aya23); prompts interactively if not specified')
    parser.add_argument('--model-number', type=int, default=0,
                        help='Model number (1-based index in installed_models)')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Resolve benchmark file: explicit --benchmark wins; otherwise auto-detect
    # uncensored first, then fall back to regular.
    if args.benchmark:
        data_path = Path(args.benchmark)
        if not data_path.is_absolute():
            data_path = project_root / data_path
        if not data_path.exists():
            print(f"ERROR: Benchmark data not found: {data_path}")
            sys.exit(1)
    else:
        uncensored = project_root / "data" / f"{args.lang}_uncensored_benchmark.yaml"
        regular = project_root / "data" / f"{args.lang}_benchmark.yaml"
        if uncensored.exists():
            data_path = uncensored
            print(f"Auto-detected benchmark: {data_path}")
        elif regular.exists():
            data_path = regular
            print(f"Auto-detected benchmark: {data_path}")
        else:
            print(f"ERROR: No benchmark file found for language '{args.lang}'.")
            print(f"  Looked for: {uncensored}")
            print(f"  Looked for: {regular}")
            sys.exit(1)

    # Load models config (catalog of available models)
    models_config_path = project_root / "models" / "models_config.yaml"
    if not models_config_path.exists():
        print(f"ERROR: Models configuration not found at {models_config_path}")
        print("Please run 0-setup.ps1 first to install models.")
        sys.exit(1)
    with open(models_config_path, 'r', encoding='utf-8') as f:
        models_config = yaml.safe_load(f)
    available_models = models_config.get('available_models', {})

    current_config_path = project_root / "models" / "current_config.yaml"
    installed_models = []
    if current_config_path.exists():
        with open(current_config_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f) or {}
        installed_models = current_config.get('installed_models', [])
    if not installed_models:
        print("ERROR: No models are installed!")
        sys.exit(1)

    # Resolve model key
    if args.model_key:
        if args.model_key not in installed_models:
            print(f"ERROR: Model '{args.model_key}' is not installed. Available: {', '.join(installed_models)}")
            sys.exit(1)
        model_key = args.model_key
    elif args.model_number > 0:
        if args.model_number > len(installed_models):
            print(f"ERROR: Invalid model number: {args.model_number}. Available: 1-{len(installed_models)}")
            sys.exit(1)
        model_key = installed_models[args.model_number - 1]
    else:
        model_key = _select_model_interactive(installed_models, available_models)

    # Resolve glossary
    if args.glossary:
        glossary_path = Path(args.glossary)
        if not glossary_path.is_absolute():
            glossary_path = project_root / glossary_path
    else:
        glossary_path = _auto_detect_glossary(data_path)

    # Confirmation
    model_info = available_models.get(model_key, {})
    print()
    print("=" * 65)
    print("       Benchmark Summary")
    print("=" * 65)
    print(f"  Model:     {model_info.get('name', model_key)} "
          f"({model_info.get('params', '?')}, {model_info.get('size', '?')})")
    print(f"  Benchmark: {data_path}")
    print(f"  Glossary:  {glossary_path if glossary_path else 'None'}")
    print("=" * 65)
    print()

    if not args.yes:
        confirm = input("Proceed with benchmark? (Y/N): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("Cancelled by user.")
            sys.exit(0)

    stats = run_benchmark(data_path, glossary_path, model_key)

    saved_path = _save_benchmark_result(project_root, model_key, data_path, stats)
    print(f"\nResults saved to: {saved_path}")


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python benchmark.py data/ro_benchmark.yaml")
        print("  python benchmark.py data/ro_benchmark.yaml --model aya23")
        print("  python benchmark.py data/ro_benchmark.yaml --model madlad400 --glossary data/ro_glossary.yaml")
        print("  python benchmark.py orchestrate [--benchmark FILE] [--model-key KEY] [-y]")
        sys.exit(1)

    # Parse arguments
    data_path = Path(sys.argv[1])
    glossary_path = None
    model_key = "aya23"  # Default model

    # Check for --glossary and --model parameters
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--glossary' and i + 1 < len(sys.argv):
            glossary_path = Path(sys.argv[i + 1])
            i += 2
        elif arg == '--model' and i + 1 < len(sys.argv):
            model_key = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if not data_path.exists():
        print(f"ERROR: Benchmark data not found: {data_path}")
        sys.exit(1)

    # Auto-detect glossary if not specified
    if glossary_path is None:
        glossary_path = _auto_detect_glossary(data_path)

    # Run benchmark
    run_benchmark(data_path, glossary_path, model_key)


def _lang_code_from_path(path: Path) -> str:
    """Infer ISO lang code from a tl/<lang_name>/ segment in the path."""
    lang_dir_map = {
        'romanian': 'ro', 'spanish': 'es', 'french': 'fr', 'german': 'de',
        'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'turkish': 'tr',
        'czech': 'cs', 'polish': 'pl', 'ukrainian': 'uk', 'bulgarian': 'bg',
        'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko', 'vietnamese': 'vi',
        'thai': 'th', 'indonesian': 'id', 'arabic': 'ar', 'hebrew': 'he',
        'persian': 'fa', 'hindi': 'hi', 'bengali': 'bn', 'dutch': 'nl',
        'swedish': 'sv', 'norwegian': 'no', 'danish': 'da', 'finnish': 'fi',
        'greek': 'el', 'hungarian': 'hu',
    }
    for part in path.parts:
        code = lang_dir_map.get(part.lower())
        if code:
            return code
    return 'ro'


def _select_parsed_yaml_interactive(project_root: Path) -> Path:
    """Use the shared game selector to pick a game, then select a parsed YAML from it."""
    sys.path.insert(0, str(Path(__file__).parent))
    from config_selector import select_item

    # Use the game already configured by 1-config.ps1 if available
    game_path = None
    current_config_path = project_root / 'models' / 'current_config.yaml'
    if current_config_path.exists():
        with open(current_config_path, 'r', encoding='utf-8') as f:
            current_cfg = yaml.safe_load(f) or {}
        current_game = current_cfg.get('current_game')
        if current_game:
            game_info = current_cfg.get('games', {}).get(current_game, {})
            path_str = game_info.get('path')
            if path_str and Path(path_str).exists():
                game_path = Path(path_str)
                print(f"\nUsing configured game: {current_game}")

    if game_path is None:
        from config import select_game
        _game_name, game_path = select_game()

    tl_root = game_path / "game" / "tl"
    parsed_files = sorted(
        f for f in tl_root.rglob("*.parsed.yaml") if '.translated.' not in f.name
    ) if tl_root.exists() else []

    if not parsed_files:
        print(f"ERROR: No .parsed.yaml files found under {tl_root}")
        sys.exit(1)

    if len(parsed_files) == 1:
        print(f"\nAuto-selecting: {parsed_files[0].relative_to(project_root)}")
        return parsed_files[0]

    items = [{"name": f.relative_to(project_root).as_posix(), "path": f} for f in parsed_files]
    selected = select_item(
        title="Select parsed YAML to score",
        items=items,
        item_formatter_func=lambda f, i: f"  [{i}] {f['name']}",
        item_type_name="file",
    )
    return selected["path"]


def run_score_parsed_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Score all model columns in a parsed YAML against benchmark references')
    parser.add_argument('_', help='score-parsed command')
    parser.add_argument('--parsed', default=None,
                        help='Path to .parsed.yaml file (interactive game selector if omitted)')
    parser.add_argument('--benchmark', default=None,
                        help='Benchmark reference YAML (auto-detected from language if omitted)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.parsed:
        parsed_path = Path(args.parsed)
        if not parsed_path.is_absolute():
            parsed_path = project_root / parsed_path
        if not parsed_path.exists():
            print(f"ERROR: Parsed file not found: {parsed_path}")
            sys.exit(1)
    else:
        parsed_path = _select_parsed_yaml_interactive(project_root)

    if args.benchmark:
        benchmark_path = Path(args.benchmark)
        if not benchmark_path.is_absolute():
            benchmark_path = project_root / benchmark_path
    else:
        lang = _lang_code_from_path(parsed_path)
        benchmark_path = project_root / "data" / f"{lang}_uncensored_benchmark.yaml"
        if not benchmark_path.exists():
            benchmark_path = project_root / "data" / f"{lang}_benchmark.yaml"
    if not benchmark_path.exists():
        print(f"ERROR: Benchmark reference file not found: {benchmark_path}")
        sys.exit(1)

    run_score_parsed(parsed_path, benchmark_path, project_root)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "orchestrate":
        run_orchestrate()
    elif len(sys.argv) > 1 and sys.argv[1] == "score-parsed":
        run_score_parsed_cli()
    else:
        main()
