"""
Integration Test: Aya-23-8B Model Translation

This test verifies that the Aya23Translator can successfully translate a simple phrase
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
from tests.utils import BaseTranslatorIntegrationTest

# Import the specific translator
from aya23_translator import Aya23Translator


# Resolve the model from the HF cache
from hardware import is_model_available, resolve_model_path

_KEY = "aya23"
_AVAILABLE = is_model_available(_KEY)


class TestAya23Integration(BaseTranslatorIntegrationTest):
    @classmethod
    def setUpClass(cls):
        """Set up the translator instance once for all tests in this class."""
        super().setUpClass()  # Call base class setup

        if not _AVAILABLE:
            raise unittest.SkipTest("Aya-23 model not in HF cache (run 0-setup.ps1)")

        print("Setting up Aya23Translator for integration test...")
        cls.translator = Aya23Translator(
            model_path=resolve_model_path(_KEY),
            target_language='Romanian',
            n_gpu_layers=0  # Use CPU for this simple test to avoid GPU memory issues
        )
        print("Translator setup complete.")

    def test_translate_hello_world(self):
        english_text = "The quick brown fox jumps over the lazy dog."
        translation = self.translator.translate(english_text)
        self.assertIsNotNone(translation)
        self.assertGreater(len(translation.strip()), 0)
        latin_chars = sum(1 for c in translation if c.isalpha() and ord(c) < 0x0500)
        self.assertGreater(latin_chars, 0, f"Output appears non-Latin: {translation!r}")

if __name__ == '__main__':
    unittest.main()
