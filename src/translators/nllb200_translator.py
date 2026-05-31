"""
NLLB-200-Distilled-600M translator.
Meta's 200-language seq2seq model — better quality than mBART/M2M-100.
Language codes use NLLB format: ron_Latn, eng_Latn, spa_Latn, etc.
"""

from pathlib import Path
from translators.translator_utils import (
    probe_device, safe_generate, apply_glossary, apply_source_conditioned, back_map_for,
    apply_ro_subjunctive, from_pretrained_cached,
)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    torch = None


# ISO 639-1 → NLLB language code
_NLLB_CODES = {
    'ro': 'ron_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'tr': 'tur_Latn',
    'cs': 'ces_Latn',
    'pl': 'pol_Latn',
    'uk': 'ukr_Cyrl',
    'bg': 'bul_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'vi': 'vie_Latn',
    'th': 'tha_Thai',
    'id': 'ind_Latn',
    'ar': 'arb_Arab',
    'he': 'heb_Hebr',
    'fa': 'pes_Arab',
    'hi': 'hin_Deva',
    'bn': 'ben_Beng',
    'nl': 'nld_Latn',
    'sv': 'swe_Latn',
    'no': 'nob_Latn',
    'da': 'dan_Latn',
    'fi': 'fin_Latn',
    'el': 'ell_Grek',
    'hu': 'hun_Latn',
}


class NLLB200Translator:
    """
    NLLB-200-distilled-600M translator via HuggingFace transformers.
    Supports 200 languages; no context window or instruction-following.
    """

    def __init__(
        self,
        model_path: str = None,
        target_language: str = "Romanian",
        lang_code: str = "ro",
        device: str = None,
        glossary: dict = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                f"NLLB200Translator requires transformers and torch.\n"
                f"Original error: {IMPORT_ERROR}"
            )

        self._target_language = target_language
        self.lang_code = lang_code
        self.glossary = glossary or {}

        if device is None:
            device = probe_device()
        self.device = device

        if model_path is None:
            model_path = "facebook/nllb-200-distilled-600M"
        self.model_path = str(model_path)

        nllb_target = _NLLB_CODES.get(lang_code)
        if nllb_target is None:
            raise ValueError(
                f"Language code '{lang_code}' has no NLLB mapping. "
                f"Supported codes: {sorted(_NLLB_CODES)}"
            )
        self.nllb_src = "eng_Latn"
        self.nllb_tgt = nllb_target

        print(f"Initializing NLLB-200 (EN -> {target_language})...")
        print(f"  NLLB target code : {nllb_target}")
        print(f"  Device           : {device}")
        print(f"  Model path       : {self.model_path}")
        print("  Loading model... This may take 30-60 seconds...")

        self.tokenizer = from_pretrained_cached(
            AutoTokenizer,
            self.model_path,
            src_lang=self.nllb_src,
        )
        self.model = from_pretrained_cached(
            AutoModelForSeq2SeqLM,
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.model = self.model.to(device)
        self.model.eval()

        print("Model loaded successfully!")

    @property
    def target_language(self) -> str:
        return self._target_language

    def translate(
        self,
        text: str,
        context: list = None,
        speaker: str = None,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> str:
        return self.translate_batch([text], max_new_tokens=max_new_tokens, num_beams=num_beams)[0]

    def translate_batch(
        self,
        texts: list,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> list:
        if not texts:
            return []

        self.tokenizer.src_lang = self.nllb_src
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        forced_bos = self.tokenizer.convert_tokens_to_ids(self.nllb_tgt)

        def _generate(inputs_dict):
            return self.model.generate(
                **inputs_dict,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        tokens, self.model, self.device = safe_generate(self.model, inputs, self.device, _generate)

        translations = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        results = []
        back_map = back_map_for(self.target_language)
        for src, t in zip(texts, translations):
            t = apply_glossary(src, t, self.glossary)
            t = apply_source_conditioned(src, t, back_map)
            t = apply_ro_subjunctive(t)
            results.append(t.strip())
        return results
