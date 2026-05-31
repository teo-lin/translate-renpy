"""
Integration Test: QuickMT-En-Ro Model Translation

This test verifies that the QuickMTTranslator can successfully translate a simple phrase
from English to Romanian. It's a lightweight check to ensure the model loads
and produces the expected output.
"""

import sys
from pathlib import Path

# Add project root and src/translators to sys.path for module discovery
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "translators"))

import unittest

# Import the base test class and utility functions
from tests.utils import (BaseTranslatorIntegrationTest, TranslateBatchTestMixin,
                         get_test_device, skip_if_transformers_unavailable,
                         safe_init_translator)

# Import the specific translator and its related flags
from helsinkyRo_translator import QuickMTTranslator, TRANSFORMERS_AVAILABLE, IMPORT_ERROR

from hardware import is_model_available, resolve_model_path

_KEY = "helsinkiRo"
_AVAILABLE = is_model_available(_KEY)
_MODEL_REF = resolve_model_path(_KEY) if _AVAILABLE else None


@unittest.skipIf(not _AVAILABLE, "helsinkiRo model not in HF cache (run 0-setup.ps1)")
class TestQuickMTIntegration(TranslateBatchTestMixin, BaseTranslatorIntegrationTest):

    @classmethod
    def setUpClass(cls):
        """Set up the translator instance once for all tests in this class."""
        super().setUpClass()  # Call base class setup

        skip_if_transformers_unavailable(TRANSFORMERS_AVAILABLE, IMPORT_ERROR, "QuickMTTranslator")

        cls.translator = safe_init_translator(
            translator_class=QuickMTTranslator,
            translator_name="QuickMTTranslator",
            init_kwargs={
                'model_path': _MODEL_REF,
                'target_language': 'Romanian',
                'lang_code': 'ro',
                'device': get_test_device()
            }
        )

    def test_translate_hello_world(self):
        english_text = "The quick brown fox jumps over the lazy dog."
        translation = self.translator.translate(english_text)
        self.assertIsNotNone(translation)
        self.assertGreater(len(translation.strip()), 0)
        latin_chars = sum(1 for c in translation if c.isalpha() and ord(c) < 0x0500)
        self.assertGreater(latin_chars, 0, f"Output appears non-Latin: {translation!r}")

if __name__ == '__main__':
    unittest.main()
