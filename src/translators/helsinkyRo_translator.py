"""
QuickMT-En-Ro Translator Implementation

Uses Hugging Face transformers for fast English to Romanian translation.
Lightweight model optimized for speed.
"""

import warnings
from pathlib import Path
from translators.translator_utils import (
    probe_device, safe_generate, apply_glossary, apply_source_conditioned, back_map_for,
    apply_ro_subjunctive, from_pretrained_cached,
)

# Try to import transformers dependencies
try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Define dummy classes to avoid NameError
    MarianMTModel = None
    MarianTokenizer = None
    torch = None


class QuickMTTranslator:
    """
    QuickMT translator using Hugging Face transformers

    Lightweight Marian MT model for fast English-Romanian translation.
    """

    def __init__(self, model_path: str = None, target_language: str = "Romanian",
                 lang_code: str = "ro", device: str = None, glossary: dict = None):
        """
        Initialize QuickMT translator

        Args:
            model_path: Path to local model or HuggingFace model ID
            target_language: Target language name
            lang_code: Language code (default: "ro" for Romanian)
            device: Device to use ('cuda' or 'cpu'). If None, auto-detected.
            glossary: Optional dict of EN->target language term mappings
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                f"QuickMTTranslator requires transformers and torch packages.\n"
                f"Original error: {IMPORT_ERROR}"
            )

        self._target_language = target_language
        self.lang_code = lang_code
        self.glossary = glossary or {}

        # Auto-detect device
        if device is None:
            device = probe_device()
        self.device = device

        if model_path is None:
            model_path = "Helsinki-NLP/opus-mt-en-ro"

        self.model_path = str(model_path)

        print(f"Initializing QuickMT Translation (EN->{target_language})...")
        print(f"  Language code: {lang_code}")
        print(f"  Device: {device}")
        print(f"  Model: {model_path}")
        print(f"  Loading model... This may take 10-30 seconds...")

        # Suppress sacremoses warning (it's optional and not needed for basic translation)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*sacremoses.*")
            # Load tokenizer and model from HF cache (offline-first, fetch if missing)
            self.tokenizer = from_pretrained_cached(MarianTokenizer, str(model_path))

        # Use memory-efficient loading
        self.model = from_pretrained_cached(
            MarianMTModel,
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.model = self.model.to(device)

        self.model.eval()
        print("Model loaded successfully!")

    @property
    def target_language(self) -> str:
        """Return the target language name"""
        return self._target_language

    def translate(self, text: str, max_length: int = 256, num_beams: int = 4,
                  context: list = None, speaker: str = None) -> str:
        """
        Translate text using QuickMT

        Args:
            text: English text to translate
            max_length: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search (default 4)
            context: Optional list of previous dialogue lines (for consistency)
            speaker: Optional character name/identifier

        Returns:
            Translated text
        """
        return self.translate_batch([text], max_length=max_length, num_beams=num_beams)[0]

    def translate_batch(self, texts: list, max_length: int = 256,
                        num_beams: int = 4) -> list:
        if not texts:
            return []

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        def _generate(inputs_dict):
            return self.model.generate(
                **inputs_dict,
                max_new_tokens=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        generated_tokens, self.model, self.device = safe_generate(self.model, inputs, self.device, _generate)

        translations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        results = []
        back_map = back_map_for(self.target_language)
        for src, t in zip(texts, translations):
            t = apply_glossary(src, t, self.glossary)
            t = apply_source_conditioned(src, t, back_map)
            t = apply_ro_subjunctive(t)
            results.append(t.strip())
        return results


if __name__ == "__main__":
    """
    CLI entry point for standalone translation script usage.

    Usage:
        python quickmt_translator.py <input_file> --language ro

    Example:
        python quickmt_translator.py script.rpy --language ro
    """
    import sys
    from pathlib import Path
    from translator_utils import setup_sys_path

    # Add parent directory to path for imports
    setup_sys_path()

    from translation_pipeline import RenpyTranslationPipeline

    if len(sys.argv) < 3:
        print("Usage: python quickmt_translator.py <input_file> --language ro")
        sys.exit(1)

    # Parse arguments
    input_file = Path(sys.argv[1])
    lang_code = None

    # Check for --language parameter
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--language' and i + 1 < len(sys.argv):
            lang_code = sys.argv[i + 1]
            break

    if lang_code != 'ro':
        print("Error: QuickMT-En-Ro only supports Romanian translation (--language ro)")
        sys.exit(1)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Load glossary using shared utility
    from translator_utils import get_project_root, load_glossary
    project_root = get_project_root()
    glossary = load_glossary("ro", project_root)

    # Initialize translator
    translator = QuickMTTranslator(
        target_language="Romanian",
        lang_code="ro",
        glossary=glossary
    )

    # Initialize pipeline
    pipeline = RenpyTranslationPipeline(translator)

    # Translate file
    try:
        pipeline.translate_file(input_file, output_path=None)
        sys.exit(0)
    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
