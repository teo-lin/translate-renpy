"""
Unit Test: Model Comparison Script (compare.py)

Tests the BenchmarkTranslator class that translates ALL blocks
and saves translations under numbered keys (r0, r1, r2) for model comparison.

This test uses a mock translator to avoid requiring real models.
"""

import sys
import yaml
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

# Import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from models import ParsedBlock, is_separator_block
from compare import BenchmarkTranslator


class MockTranslator:
    """Mock translator for testing without loading a real model"""

    def __init__(self, model_name='Mock'):
        self.model_name = model_name
        self.translations_called = []

    def translate(self, text: str, context: list = None, speaker: str = None, **kwargs) -> str:
        """Mock translation - just adds [MODEL_NAME] prefix and records the call"""
        # Record what was called
        self.translations_called.append({
            'text': text,
            'context': context,
            'speaker': speaker
        })

        # Simple mock translation
        return f"[{self.model_name}] {text}"


def test_benchmark_translator_translates_all_blocks():
    """Test that BenchmarkTranslator translates ALL blocks, not just empty ones"""

    print("\n" + "=" * 70)
    print("TEST 1: BenchmarkTranslator translates ALL blocks")
    print("=" * 70)

    # Create sample parsed blocks (some already translated)
    parsed_blocks = {
        'dialogue-1-Amelia': {
            'en': 'Hello there!',
            'ro': 'Salut!'  # Already translated
        },
        'dialogue-2-MainCharacter': {
            'en': 'Hi Amelia!',
            'ro': ''  # Empty
        },
        'dialogue-3-Amelia': {
            'en': 'How are you?',
            'ro': 'Ce mai faci?'  # Already translated
        },
        'separator-1': {
            'type': 'separator'
        }
    }

    # Create tags file
    tags_file = {
        'metadata': {
            'source_language': 'english',
            'target_language': 'romanian'
        },
        'structure': {
            'block_order': list(parsed_blocks.keys())
        },
        'blocks': {}
    }

    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        parsed_yaml = tmpdir_path / "test.parsed.yaml"
        tags_yaml = tmpdir_path / "test.tags.json"

        # Write test files
        with open(parsed_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(parsed_blocks, f, allow_unicode=True)

        with open(tags_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Create BenchmarkTranslator with mock translator
        mock_translator = MockTranslator(model_name='TestModel')
        benchmark = BenchmarkTranslator(
            translator=mock_translator,
            characters={},
            save_key='ay'
        )

        # Run translation
        stats = benchmark.translate_file(parsed_yaml, tags_yaml)

        # Verify stats
        assert stats['total'] == 3, f"Expected 3 total blocks, got {stats['total']}"
        assert stats['translated'] == 3, f"Expected 3 translated, got {stats['translated']}"
        assert stats['failed'] == 0, f"Expected 0 failed, got {stats['failed']}"

        print(f"  [OK] Translated {stats['translated']}/{stats['total']} blocks")

        # Verify mock translator was called 3 times (not just for empty translations)
        assert len(mock_translator.translations_called) == 3, \
            f"Expected 3 translation calls, got {len(mock_translator.translations_called)}"

        print(f"  [OK] Mock translator called {len(mock_translator.translations_called)} times")

        # Load result and check
        with open(parsed_yaml, 'r', encoding='utf-8') as f:
            result = yaml.safe_load(f)

        # Check that ALL blocks have the new key
        for block_id in ['dialogue-1-Amelia', 'dialogue-2-MainCharacter', 'dialogue-3-Amelia']:
            assert 'ay' in result[block_id], f"Block {block_id} missing 'ay' key"
            assert result[block_id]['ay'].startswith('[TestModel]'), \
                f"Block {block_id} has incorrect translation: {result[block_id]['ay']}"

        print(f"  [OK] All blocks have 'ay' key with translations")



    print("\n  [PASS] All assertions passed!")



def test_numbered_key_storage():
    """Test that translations are stored under correct numbered keys"""

    print("\n" + "=" * 70)
    print("TEST 2: Two-letter key storage (ay, he, ma, etc.)")
    print("=" * 70)

    # Create sample data
    parsed_blocks = {
        'dialogue-1-Test': {
            'en': 'Test sentence.',
            'ro': ''
        }
    }

    tags_file = {
        'metadata': {
            'source_language': 'english',
            'target_language': 'romanian'
        },
        'structure': {
            'block_order': list(parsed_blocks.keys())
        },
        'blocks': {}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        parsed_yaml = tmpdir_path / "test.parsed.yaml"
        tags_yaml = tmpdir_path / "test.tags.json"

        # Write test files
        with open(parsed_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(parsed_blocks, f, allow_unicode=True)

        with open(tags_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Test different keys
        test_keys = ['ay', 'he', 'ma', 'mb'] # Changed from r0, r1, r2, r3

        for key in test_keys:
            mock_translator = MockTranslator(model_name=f'Model{key}')
            benchmark = BenchmarkTranslator(
                translator=mock_translator,
                characters={},
                save_key=key
            )

            benchmark.translate_file(parsed_yaml, tags_yaml)

            # Check the key was added
            with open(parsed_yaml, 'r', encoding='utf-8') as f:
                result = yaml.safe_load(f)

            assert key in result['dialogue-1-Test'], f"Missing key '{key}'"
            assert result['dialogue-1-Test'][key].startswith(f'[Model{key}]'), \
                f"Incorrect translation for key '{key}'"

            print(f"  [OK] Key '{key}' correctly stored with translation")

    print("\n  [PASS] All two-letter keys work correctly!")



def test_context_extraction():
    """Test that context is extracted correctly for dialogue"""

    print("\n" + "=" * 70)
    print("TEST 3: Context extraction for dialogue")
    print("=" * 70)

    # Create conversation with context
    parsed_blocks = {
        'dialogue-1-Alice': {'en': 'Line 1', 'ro': ''},
        'dialogue-2-Bob': {'en': 'Line 2', 'ro': ''},
        'dialogue-3-Alice': {'en': 'Line 3', 'ro': ''},  # Target - should have 2 lines before, 1 after
        'dialogue-4-Bob': {'en': 'Line 4', 'ro': ''},
        'separator-1': {'type': 'separator'}
    }

    tags_file = {
        'metadata': {'source_language': 'english', 'target_language': 'romanian'},
        'structure': {'block_order': list(parsed_blocks.keys())},
        'blocks': {}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        parsed_yaml = tmpdir_path / "test.parsed.yaml"
        tags_yaml = tmpdir_path / "test.tags.json"

        with open(parsed_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(parsed_blocks, f, allow_unicode=True)

        with open(tags_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        mock_translator = MockTranslator()
        benchmark = BenchmarkTranslator(
            translator=mock_translator,
            characters={},
            save_key='ay',
            context_before=2,  # 2 lines before
            context_after=1   # 1 line after
        )

        benchmark.translate_file(parsed_yaml, tags_yaml)

        # Find the call for 'Line 3' (dialogue-3-Alice)
        line3_call = None
        for call in mock_translator.translations_called:
            if call['text'] == 'Line 3':
                line3_call = call
                break

        assert line3_call is not None, "Translation call for 'Line 3' not found"

        # Check context
        context = line3_call['context']
        assert context is not None, "Context is None"
        assert len(context) == 3, f"Expected 3 context lines, got {len(context)}"
        assert 'Line 1' in context[0], "Missing Line 1 in context"
        assert 'Line 2' in context[1], "Missing Line 2 in context"
        assert 'Line 4' in context[2], "Missing Line 4 in context"

        print(f"  [OK] Context correctly extracted: {len(context)} lines")
        print(f"    Before: {context[:2]}")
        print(f"    After:  {context[2:]}")

    print("\n  [PASS] Context extraction works correctly!")




