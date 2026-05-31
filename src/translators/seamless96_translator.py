"""
SeamlessM4T-v2 Translator Implementation

Uses Hugging Face transformers for translation with Meta's latest multimodal model.
Optimized for high-quality text translation with nearly 100 languages supported.
"""

import warnings
from pathlib import Path
from contextlib import contextmanager
from translators.translator_utils import (
    probe_device, safe_generate, apply_glossary, apply_source_conditioned, back_map_for,
    apply_ro_subjunctive, from_pretrained_cached,
)

# Suppress known non-critical warnings for this module
warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*swigvarlink.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*layer_idx.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*were not initialized.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TRAIN this model.*", category=UserWarning)

# Try to import transformers dependencies
try:
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    from transformers import logging as transformers_logging

    # Set transformers logging to error level to suppress warnings
    transformers_logging.set_verbosity_error()

    TRANSFORMERS_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Define dummy classes to avoid NameError
    AutoProcessor = None
    SeamlessM4Tv2Model = None
    torch = None
    transformers_logging = None


class SeamlessM4Tv2Translator:
    """
    SeamlessM4T-v2 translator using Hugging Face transformers

    Supports nearly 100 languages with state-of-art translation quality.
    Uses Meta's 2024 multimodal model (text + speech, though we only use text).
    """

    # Mapping of common language codes to SeamlessM4T language codes
    # SeamlessM4T uses 3-letter ISO codes (e.g., "ron" for Romanian)
    LANGUAGE_CODE_MAP = {
        'ro': 'ron',  # Romanian
        'es': 'spa',  # Spanish
        'fr': 'fra',  # French
        'de': 'deu',  # German
        'it': 'ita',  # Italian
        'pt': 'por',  # Portuguese
        'ru': 'rus',  # Russian
        'tr': 'tur',  # Turkish
        'cs': 'ces',  # Czech
        'pl': 'pol',  # Polish
        'uk': 'ukr',  # Ukrainian
        'bg': 'bul',  # Bulgarian
        'zh': 'cmn',  # Chinese (Mandarin)
        'ja': 'jpn',  # Japanese
        'ko': 'kor',  # Korean
        'vi': 'vie',  # Vietnamese
        'th': 'tha',  # Thai
        'id': 'ind',  # Indonesian
        'ar': 'arb',  # Arabic (Modern Standard)
        'he': 'heb',  # Hebrew
        'fa': 'pes',  # Persian/Farsi
        'hi': 'hin',  # Hindi
        'bn': 'ben',  # Bengali
        'nl': 'nld',  # Dutch
        'sv': 'swe',  # Swedish
        'no': 'nor',  # Norwegian
        'da': 'dan',  # Danish
        'fi': 'fin',  # Finnish
        'el': 'ell',  # Greek
        'hu': 'hun',  # Hungarian
        'en': 'eng',  # English (source)
    }

    def __init__(self, model_path: str = None, target_language: str = "Romanian", lang_code: str = None,
                 device: str = None, glossary: dict = None, model_name: str = None):
        """
        Initialize SeamlessM4T-v2 translator

        Args:
            target_language: Target language name (e.g., "Romanian", "Spanish", "Japanese")
            lang_code: 2-letter language code (e.g., "ro", "es", "ja"). Auto-converted to 3-letter.
            device: Device to use ('cuda' or 'cpu'). If None, auto-detected.
            glossary: Optional dict of EN->target language term mappings
            model_name: Model variant to use. Default: "facebook/seamless-m4t-v2-large"
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                f"SeamlessM4Tv2Translator requires transformers and torch packages.\n"
                f"Original error: {IMPORT_ERROR}\n"
                f"This is likely due to triton/torch version incompatibility.\n"
                f"See: https://github.com/pytorch/ao/issues/2919"
            )

        self._target_language = target_language
        self.glossary = glossary or {}

        # Auto-detect language code if not provided
        if lang_code is None:
            # Try to guess from language name
            lang_name_lower = target_language.lower()
            # Try exact matches first
            name_to_code = {
                'romanian': 'ro', 'spanish': 'es', 'french': 'fr',
                'german': 'de', 'italian': 'it', 'portuguese': 'pt',
                'russian': 'ru', 'turkish': 'tr', 'czech': 'cs',
                'polish': 'pl', 'ukrainian': 'uk', 'bulgarian': 'bg',
                'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko',
                'vietnamese': 'vi', 'thai': 'th', 'indonesian': 'id',
                'arabic': 'ar', 'hebrew': 'he', 'persian': 'fa',
                'farsi': 'fa', 'hindi': 'hi', 'bengali': 'bn',
                'dutch': 'nl', 'swedish': 'sv', 'norwegian': 'no',
                'danish': 'da', 'finnish': 'fi', 'greek': 'el',
                'hungarian': 'hu'
            }
            lang_code = name_to_code.get(lang_name_lower, target_language[:2].lower())

        # Convert 2-letter code to 3-letter code for SeamlessM4T
        self.lang_code_3letter = self.LANGUAGE_CODE_MAP.get(lang_code, lang_code)
        self.lang_code = lang_code

        # Auto-detect device
        if device is None:
            device = probe_device()
        self.device = device

        if model_path is None:
            if model_name is None:
                model_path = "facebook/seamless-m4t-v2-large"
            else:
                model_path = model_name

        self.model_name = str(model_path)

        print(f"Initializing SeamlessM4T-v2 Translation (EN->{target_language})...")
        print(f"  Language code: {lang_code} ({self.lang_code_3letter})")
        print(f"  Device: {device}")
        print(f"  Model: {model_path}")
        print(f"  Loading model... This may take 60-90 seconds...")

        # Load processor and model from HF cache (offline-first, fetch if missing)
        self.processor = from_pretrained_cached(AutoProcessor, str(model_path))

        # Use memory-efficient loading to avoid paging file errors
        self.model = from_pretrained_cached(
            SeamlessM4Tv2Model,
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.model = self.model.to(device)

        self.model.eval()

        print("Model loaded successfully!")
        print("  Note: SeamlessM4T-v2 is a large model (2.3GB+)")

    @property
    def target_language(self) -> str:
        """Return the target language name"""
        return self._target_language

    def _apply_glossary(self, text: str, translation: str) -> str:
        translation = apply_glossary(text, translation, self.glossary)
        translation = apply_source_conditioned(text, translation, back_map_for(self.target_language))
        translation = apply_ro_subjunctive(translation)
        return translation

    def translate(self, text: str, max_length: int = 256, num_beams: int = 5,
                  context: list = None, speaker: str = None) -> str:
        """
        Translate text using SeamlessM4T-v2

        Args:
            text: English text to translate
            max_length: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search (default 5 for quality)
            context: Optional list of previous dialogue lines (for consistency)
            speaker: Optional character name/identifier

        Returns:
            Translated text
        """
        return self.translate_batch([text], max_length=max_length, num_beams=num_beams)[0]

    def translate_batch(self, texts: list, max_length: int = 256,
                        num_beams: int = 5) -> list:
        if not texts:
            return []

        text_inputs = self.processor(
            text=texts,
            src_lang="eng",
            return_tensors="pt",
            padding=True,
        )

        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        if 'input_ids' in text_inputs:
            text_inputs['input_ids'] = text_inputs['input_ids'].to(torch.long)
        if 'attention_mask' in text_inputs:
            text_inputs['attention_mask'] = text_inputs['attention_mask'].to(torch.long)

        def _generate(inputs_dict):
            return self.model.generate(
                **inputs_dict,
                tgt_lang=self.lang_code_3letter,
                max_new_tokens=max_length,
                num_beams=num_beams,
                early_stopping=True,
                generate_speech=False,
                return_intermediate_token_ids=True,
            )

        output_tokens, self.model, self.device = safe_generate(self.model, text_inputs, self.device, _generate)

        generated_sequences = output_tokens.sequences
        if generated_sequences.dtype != torch.long:
            generated_sequences = generated_sequences.to(torch.long)

        results = []
        for i, src in enumerate(texts):
            translation = self.processor.tokenizer.decode(
                generated_sequences[i], skip_special_tokens=True
            )
            if self.lang_code == 'ro':
                translation = translation.replace('ş', 'ș').replace('ţ', 'ț')
            translation = self._apply_glossary(src, translation)
            results.append(translation.strip())
        return results


if __name__ == "__main__":
    """
    CLI entry point for standalone translation script usage.

    Usage:
        python seamlessm4t_translator.py <input_file> --language <language_code>

    Example:
        python seamlessm4t_translator.py script.rpy --language ro
    """
    import sys
    from pathlib import Path
    from translator_utils import (
        get_project_root, load_glossary, parse_cli_language_arg,
        setup_sys_path
    )

    # Add parent directory to path for imports
    setup_sys_path()
    from translation_pipeline import RenpyTranslationPipeline

    if len(sys.argv) < 3:
        print("Usage: python seamlessm4t_translator.py <input_file> --language <lang_code>")
        print("Example: python seamlessm4t_translator.py script.rpy --language ro")
        sys.exit(1)

    # Parse arguments
    input_file = Path(sys.argv[1])
    target_language, lang_code = parse_cli_language_arg()

    if not target_language or not lang_code:
        print("Error: --language parameter is required")
        sys.exit(1)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Load glossary using shared utility
    project_root = get_project_root()
    glossary = load_glossary(lang_code, project_root)

    # Initialize translator
    translator = SeamlessM4Tv2Translator(
        target_language=target_language,
        lang_code=lang_code,
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
