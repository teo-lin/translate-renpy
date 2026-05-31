"""
Integration tests for LlamaCppTranslator with real GGUF models.
All tests are skipped automatically when no GGUF model file is found.

To run these tests, at least one model must be downloaded via 0-setup.ps1.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent

# Discover the first available GGUF model in the HF cache for smoke-testing
from hardware import is_model_available, resolve_model_path

_CANDIDATE_KEYS = ["aya23", "ayaExpanse8b", "euroLLM9b", "euroLLM22b"]


def _find_model():
    """Return (model_key, model_ref) for the first cached GGUF, or (None, None)."""
    for key in _CANDIDATE_KEYS:
        if is_model_available(key):
            return key, resolve_model_path(key)
    return None, None


_MODEL_KEY, _MODEL_PATH = _find_model()
_SKIP_REASON = (
    "No GGUF model found in models/. Run 0-setup.ps1 to download one."
    if _MODEL_PATH is None
    else ""
)


@unittest.skipIf(_MODEL_PATH is None, _SKIP_REASON)
class TestLlamaCppTranslatorIntegration(unittest.TestCase):
    """
    Smoke-test: load the first available GGUF model and run a few translations.
    Assertions are loose — we only check that output is a non-empty string.
    """

    @classmethod
    def setUpClass(cls):
        from translators.llama_cpp_translator import LlamaCppTranslator

        print(f"\nIntegration test: loading {_MODEL_KEY} from {_MODEL_PATH}")
        cls.translator = LlamaCppTranslator(
            model_path=str(_MODEL_PATH),
            target_language="Romanian",
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=128,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.translator
        cls.translator = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def test_translate_returns_string(self):
        result = self.translator.translate("Hello!")
        self.assertIsInstance(result, str)

    def test_translate_returns_nonempty_string(self):
        result = self.translator.translate("Hello!")
        self.assertGreater(len(result.strip()), 0)

    def test_translate_with_context_does_not_crash(self):
        result = self.translator.translate(
            "How are you?",
            context=["Alice: Hello!", "Bob: Hi there!"],
            speaker="Alice",
        )
        self.assertIsInstance(result, str)

    def test_translate_short_phrase(self):
        result = self.translator.translate("Good morning.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_target_language_property(self):
        self.assertEqual(self.translator.target_language, "Romanian")

    def test_translate_does_not_echo_prompt(self):
        text = "A completely unique test phrase for echo detection xyz123"
        result = self.translator.translate(text)
        # Output should not just be the input echoed back
        self.assertNotEqual(result.strip(), text)

    def test_translate_with_glossary(self):
        from translators.llama_cpp_translator import LlamaCppTranslator

        # Load this second instance on CPU (n_gpu_layers=0): the class already
        # holds a full GPU copy, and two ~5GB GGUF models won't fit in 8GB VRAM.
        t = LlamaCppTranslator(
            model_path=str(_MODEL_PATH),
            target_language="Romanian",
            n_gpu_layers=0,
            n_ctx=8192,
            n_batch=128,
            glossary={"protagonist": "protagonist"},
        )
        result = t.translate("The protagonist walks forward.")
        self.assertIsInstance(result, str)
        del t


@unittest.skipIf(not is_model_available("ayaExpanse8b"), "ayaExpanse8b model not in HF cache")
class TestAyaExpanse8bIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from translators.llama_cpp_translator import LlamaCppTranslator
        cls.translator = LlamaCppTranslator(
            model_path=resolve_model_path("ayaExpanse8b"),
            target_language="Romanian",
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=128,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.translator

    def test_translate_returns_nonempty_string(self):
        result = self.translator.translate("Hello!")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_translate_with_context(self):
        result = self.translator.translate(
            "How are you?",
            context=["Alice: Hello!"],
            speaker="Alice",
        )
        self.assertIsInstance(result, str)

    def test_target_language_property(self):
        self.assertEqual(self.translator.target_language, "Romanian")


@unittest.skipIf(not is_model_available("euroLLM9b"), "euroLLM9b model not in HF cache")
class TestEuroLLM9bIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from translators.llama_cpp_translator import LlamaCppTranslator
        cls.translator = LlamaCppTranslator(
            model_path=resolve_model_path("euroLLM9b"),
            target_language="Romanian",
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=128,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.translator

    def test_translate_returns_nonempty_string(self):
        result = self.translator.translate("Hello!")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_translate_with_context(self):
        result = self.translator.translate(
            "How are you?",
            context=["Alice: Hello!"],
            speaker="Alice",
        )
        self.assertIsInstance(result, str)

    def test_target_language_property(self):
        self.assertEqual(self.translator.target_language, "Romanian")


@unittest.skipIf(_MODEL_PATH is None, _SKIP_REASON)
class TestAya23TranslatorIntegration(unittest.TestCase):
    """
    Integration test for Aya23Translator using the aya23 model specifically.
    Only runs when aya23 is the discovered model.
    """

    @classmethod
    def setUpClass(cls):
        if not is_model_available("aya23"):
            raise unittest.SkipTest("aya23 model not in HF cache; skipping aya23-specific test")

        from translators.aya23_translator import Aya23Translator

        print("\nIntegration test: loading Aya23Translator (aya23) from HF cache")
        cls.translator = Aya23Translator(
            n_ctx=8192,
            n_batch=128,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "translator"):
            del cls.translator

    def test_translate_returns_string(self):
        result = self.translator.translate("Hello!")
        self.assertIsInstance(result, str)

    def test_target_language_is_romanian(self):
        self.assertEqual(self.translator.target_language, "Romanian")
