"""
Test the modular translation pipeline

Tests the new 3-translate.ps1 workflow:
- Loading .parsed.yaml and .tags.yaml files
- Using YAML configuration (current_config.yaml, characters.yaml)
- Context extraction (DIALOGUE: 3 before + 1 after, CHOICE: no context)
- Translation with glossary and prompt fallback
- Writing translations back to .parsed.yaml

This test uses a mock translator to avoid requiring a full model.
"""

import sys
import yaml
from pathlib import Path
import tempfile
import argparse
from unittest.mock import MagicMock

# Import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from models import ParsedBlock, is_separator_block, parse_block_id
from translate import ModularBatchTranslator


class MockTranslator:
    """Mock translator for testing without loading a real model"""

    def __init__(self, target_language='Romanian', prompt_template=None, glossary=None):
        self._target_language = target_language
        self.prompt_template = prompt_template
        self.glossary = glossary or {}
        self.translations_called = []

    @property
    def target_language(self):
        return self._target_language

    def translate(self, text: str, context: list = None, speaker: str = None, **kwargs) -> str:
        """Mock translation - just adds [TRANSLATED] prefix and records the call"""
        # Record what was called
        self.translations_called.append({
            'text': text,
            'context': context,
            'speaker': speaker
        })

        # Simple mock translation
        translation = f"[TRANSLATED] {text}"

        # Apply glossary if terms are present
        for en_term, ro_term in self.glossary.items():
            if not en_term.startswith("_comment"):
                translation = translation.replace(en_term, ro_term)

        return translation


def test_context_extraction():
    """Test context extraction for DIALOGUE and CHOICE blocks"""

    print("\n" + "=" * 70)
    print("TEST 1: Context Extraction")
    print("=" * 70)

    # Create sample parsed blocks
    parsed_blocks = {
        '1-Amelia': {'en': 'Hello there!', 'ro': 'Salut!'},
        '2-MainCharacter': {'en': 'Hi Amelia!', 'ro': 'Bună Amelia!'},
        '3-Amelia': {'en': 'How are you?', 'ro': ''},  # Untranslated
        '4-MainCharacter': {'en': 'I am fine, thanks.', 'ro': 'Sunt bine, mulțumesc.'},
        'separator-1': {'type': 'separator'},
        '5-Choice': {'en': 'Help her', 'ro': ''},  # Untranslated CHOICE
        '6-Choice': {'en': 'Ignore her', 'ro': ''},  # Untranslated CHOICE
        '7-Amelia': {'en': 'Thank you!', 'ro': ''},  # Untranslated after choices
    }

    block_order = ['1-Amelia', '2-MainCharacter', '3-Amelia', '4-MainCharacter',
                   'separator-1', '5-Choice', '6-Choice', '7-Amelia']

    # Create mock translator
    mock_translator = MockTranslator(target_language='Romanian')
    characters = {
        'am': {'name': 'Amelia', 'gender': 'female'},
        'u': {'name': 'MainCharacter', 'gender': 'male'}
    }

    # Create batch translator
    batch_translator = ModularBatchTranslator(
        translator=mock_translator,
        characters=characters,
        target_lang_code='ro',
        context_before=3,
        context_after=1
    )

    print("\n[Testing] Context extraction...")

    # Test 1: DIALOGUE block should have context
    print("\n  Test 1a: DIALOGUE block context")
    untranslated = ['3-Amelia']
    contexts = batch_translator._extract_contexts(untranslated, parsed_blocks, block_order)

    assert len(contexts) == 1, f"Expected 1 context, got {len(contexts)}"
    context = contexts[0]

    print(f"    Block ID: {context['block_id']}")
    print(f"    Character: {context['character_name']}")
    print(f"    Is Choice: {context['is_choice']}")
    print(f"    Context lines: {len(context['context'])}")

    assert context['block_id'] == '3-Amelia', "Wrong block ID"
    assert context['character_name'] == 'Amelia', "Wrong character name"
    assert context['is_choice'] is False, "Should not be a choice"
    assert len(context['context']) > 0, "DIALOGUE should have context"

    # Should have 2 before (1-Amelia, 2-MainCharacter) and 1 after (4-MainCharacter)
    expected_context_count = 3
    assert len(context['context']) == expected_context_count, \
        f"Expected {expected_context_count} context lines, got {len(context['context'])}"

    print(f"    [OK] Context: {context['context']}")

    # Test 2: CHOICE block should have NO context
    print("\n  Test 1b: CHOICE block context")
    untranslated = ['5-Choice']
    contexts = batch_translator._extract_contexts(untranslated, parsed_blocks, block_order)

    assert len(contexts) == 1, f"Expected 1 context, got {len(contexts)}"
    context = contexts[0]

    print(f"    Block ID: {context['block_id']}")
    print(f"    Character: {context['character_name']}")
    print(f"    Is Choice: {context['is_choice']}")
    print(f"    Context lines: {len(context['context'])}")

    assert context['block_id'] == '5-Choice', "Wrong block ID"
    assert context['character_name'] == 'Choice', "Wrong character name"
    assert context['is_choice'] is True, "Should be a choice"
    assert len(context['context']) == 0, "CHOICE should have NO context"

    print("    [OK] No context (as expected for CHOICE)")

    print("\n[OK] Context extraction test passed!")



def test_translation_workflow():
    """Test full translation workflow with .parsed.yaml and .tags.yaml files"""

    print("\n" + "=" * 70)
    print("TEST 2: Translation Workflow")
    print("=" * 70)

    # Create sample parsed blocks
    parsed_blocks = {
        '1-Amelia': {'en': 'Hello!', 'ro': 'Salut!'},
        '2-MainCharacter': {'en': 'Hi!', 'ro': ''},  # Untranslated
        '3-Choice': {'en': 'Talk to her', 'ro': ''},  # Untranslated CHOICE
        '4-Choice': {'en': 'Walk away', 'ro': ''},  # Untranslated CHOICE
        'separator-1': {'type': 'separator'},
        '5-Amelia': {'en': 'See you!', 'ro': ''},  # Untranslated
    }

    # Create tags file
    tags_file = {
        'metadata': {
            'file_structure_type': 'DIALOGUE_AND_STRINGS',
            'source_file': 'test.rpy',
            'target_language': 'romanian',
            'source_language': 'english',
            'extracted_at': '2025-12-28 10:00:00',
            'total_blocks': 5,
            'untranslated_blocks': 4
        },
        'structure': {
            'block_order': ['1-Amelia', '2-MainCharacter', '3-Choice', '4-Choice', 'separator-1', '5-Amelia'],
            'separator_positions': [4]
        },
        'blocks': {
            '1-Amelia': {
                'type': 'DIALOGUE',
                'character_var': 'am',
                'tags': [],
                'template': '{character_var} "{text}"'
            },
            '2-MainCharacter': {
                'type': 'DIALOGUE',
                'character_var': 'u',
                'tags': [],
                'template': '{character_var} "{text}"'
            },
            '3-Choice': {
                'type': 'CHOICE',
                'tags': [],
                'template': '"{text}"'
            },
            '4-Choice': {
                'type': 'CHOICE',
                'tags': [],
                'template': '"{text}"'
            },
            '5-Amelia': {
                'type': 'DIALOGUE',
                'character_var': 'am',
                'tags': [],
                'template': '{character_var} "{text}"'
            }
        },
        'character_map': {
            'am': 'Amelia',
            'u': 'MainCharacter',
            '': 'Narrator'
        }
    }

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parsed.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(parsed_blocks, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        temp_parsed_yaml = Path(f.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tags.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        temp_tags_yaml = Path(f.name)

    try:
        print(f"\n[Created] Test files:")
        print(f"    Parsed YAML: {temp_parsed_yaml.name}")
        print(f"    Tags YAML: {temp_tags_yaml.name}")

        # Create mock translator with glossary
        glossary = {
            'Hello': 'Bună ziua',
            'Hi': 'Salut'
        }
        mock_translator = MockTranslator(
            target_language='Romanian',
            glossary=glossary
        )

        characters = tags_file['character_map']

        # Create batch translator
        batch_translator = ModularBatchTranslator(
            translator=mock_translator,
            characters=characters,
            target_lang_code='ro',
            context_before=3,
            context_after=1
        )

        # Translate
        print("\n[Running] Translation...")
        stats = batch_translator.translate_file(
            parsed_yaml_path=temp_parsed_yaml,
            tags_yaml_path=temp_tags_yaml,
            output_yaml_path=None  # Overwrite in place
        )

        # Verify statistics
        print(f"\n[Statistics]:")
        print(f"    Total blocks: {stats['total']}")
        print(f"    Translated: {stats['translated']}")
        print(f"    Skipped: {stats['skipped']}")
        print(f"    Failed: {stats['failed']}")

        assert stats['total'] == 5, f"Expected 5 total blocks, got {stats['total']}"
        assert stats['translated'] == 4, f"Expected 4 translated blocks, got {stats['translated']}"
        assert stats['skipped'] == 1, f"Expected 1 skipped block, got {stats['skipped']}"
        assert stats['failed'] == 0, f"Expected 0 failed blocks, got {stats['failed']}"

        # Verify translations were written to file
        print("\n[Verifying] Output file...")
        with open(temp_parsed_yaml, 'r', encoding='utf-8') as f:
            updated_blocks = yaml.safe_load(f)

        # Check that untranslated blocks are now translated
        assert updated_blocks['2-MainCharacter']['ro'] != '', "Block 2 should be translated"
        assert updated_blocks['3-Choice']['ro'] != '', "Block 3 should be translated"
        assert updated_blocks['4-Choice']['ro'] != '', "Block 4 should be translated"
        assert updated_blocks['5-Amelia']['ro'] != '', "Block 5 should be translated"

        print(f"    [OK] Block 2: {updated_blocks['2-MainCharacter']['ro']}")
        print(f"    [OK] Block 3: {updated_blocks['3-Choice']['ro']}")
        print(f"    [OK] Block 4: {updated_blocks['4-Choice']['ro']}")
        print(f"    [OK] Block 5: {updated_blocks['5-Amelia']['ro']}")

        # Verify context was used for DIALOGUE but not for CHOICE
        print("\n[Verifying] Context usage...")
        calls = mock_translator.translations_called

        # Find the DIALOGUE translation (block 2: "Hi!")
        dialogue_calls = [c for c in calls if c['text'] == 'Hi!']
        assert len(dialogue_calls) == 1, f"Expected 1 DIALOGUE call for 'Hi!', got {len(dialogue_calls)}"
        dialogue_call = dialogue_calls[0]
        assert dialogue_call['context'] is not None and len(dialogue_call['context']) > 0, \
            "DIALOGUE should have context"
        print(f"    [OK] DIALOGUE block had {len(dialogue_call['context'])} context line(s)")
        for ctx_line in dialogue_call['context']:
            print(f"        - {ctx_line}")

        # Find a CHOICE translation (block 3 or 4: "Talk to her" or "Walk away")
        choice_calls = [c for c in calls if c['text'] in ['Talk to her', 'Walk away']]
        assert len(choice_calls) >= 1, "Should have at least one CHOICE translation"
        choice_call = choice_calls[0]
        # CHOICE blocks should have empty context list (not None, but empty)
        # The context is passed as empty list [] for CHOICE blocks
        actual_context = choice_call['context'] if choice_call['context'] is not None else []
        assert len(actual_context) == 0, \
            f"CHOICE should have empty context, got {len(actual_context)} items: {actual_context}"
        print(f"    [OK] CHOICE block '{choice_call['text']}' had no context (as expected)")

        # Verify glossary was used
        print("\n[Verifying] Glossary usage...")
        hi_translation = updated_blocks['2-MainCharacter']['ro']
        assert 'Salut' in hi_translation or 'Bună ziua' in hi_translation, \
            "Glossary should have been applied"
        print(f"    [OK] Glossary applied: 'Hi' -> contains Romanian greeting")

        print("\n[OK] Translation workflow test passed!")
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Translation workflow test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Translation workflow test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        temp_parsed_yaml.unlink(missing_ok=True)
        temp_tags_yaml.unlink(missing_ok=True)


def test_untranslated_identification():
    """Test identification of untranslated blocks"""

    print("\n" + "=" * 70)
    print("TEST 3: Untranslated Block Identification")
    print("=" * 70)

    # Create sample blocks with various states
    parsed_blocks = {
        '1-Amelia': {'en': 'Hello!', 'ro': 'Salut!'},  # Translated
        '2-MainCharacter': {'en': 'Hi!', 'ro': ''},  # Empty string
        '3-Narrator': {'en': 'Text', 'ro': '   '},  # Whitespace only
        '4-Choice': {'en': 'Choice', 'ro': 'Alegere'},  # Translated
        'separator-1': {'type': 'separator'},  # Separator (should be skipped)
        '5-Amelia': {'en': 'Bye!'},  # Missing 'ro' field entirely
    }

    # Create mock translator
    mock_translator = MockTranslator()
    characters = {}

    # Create batch translator
    batch_translator = ModularBatchTranslator(
        translator=mock_translator,
        characters=characters,
        target_lang_code='ro',
        context_before=3,
        context_after=1
    )

    print("\n[Identifying] Untranslated blocks...")
    untranslated = batch_translator._identify_untranslated(parsed_blocks, 'ro')

    print(f"\n  Found {len(untranslated)} untranslated blocks:")
    for block_id in untranslated:
        print(f"    - {block_id}")

    # Should find blocks 2, 3, and 5 (empty, whitespace, missing field)
    expected_untranslated = {'2-MainCharacter', '3-Narrator', '5-Amelia'}
    actual_untranslated = set(untranslated)

    assert actual_untranslated == expected_untranslated, \
        f"Expected {expected_untranslated}, got {actual_untranslated}"

    # Should NOT include separator or already translated blocks
    assert 'separator-1' not in untranslated, "Separator should be skipped"
    assert '1-Amelia' not in untranslated, "Translated block should be skipped"
    assert '4-Choice' not in untranslated, "Translated choice should be skipped"

    print("\n[OK] Untranslated identification test passed!")



def test_language_agnostic():
    """Test that the translator works with different languages (not just Romanian)"""

    print("\n" + "=" * 70)
    print("TEST 4: Language-Agnostic Translation")
    print("=" * 70)

    # Create sample blocks
    parsed_blocks = {
        '1-Amelia': {'en': 'Hello!', 'es': ''},  # Spanish
        '2-MainCharacter': {'en': 'Hi!', 'es': ''},
    }

    tags_file = {
        'metadata': {
            'file_structure_type': 'DIALOGUE_ONLY',
            'source_file': 'test.rpy',
            'target_language': 'spanish',
            'source_language': 'english',
            'extracted_at': '2025-12-28 10:00:00',
            'total_blocks': 2,
            'untranslated_blocks': 2
        },
        'structure': {
            'block_order': ['1-Amelia', '2-MainCharacter'],
            'separator_positions': []
        },
        'blocks': {},
        'character_map': {'am': 'Amelia', 'u': 'MainCharacter'}
    }

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parsed.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(parsed_blocks, f, allow_unicode=True)
        temp_parsed_yaml = Path(f.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tags.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        temp_tags_yaml = Path(f.name)

    try:
        print(f"\n[Testing] Spanish translation...")

        # Create mock translator for Spanish
        mock_translator = MockTranslator(target_language='Spanish')

        # Create batch translator with 'es' language code
        batch_translator = ModularBatchTranslator(
            translator=mock_translator,
            characters=tags_file['character_map'],
            target_lang_code='es',  # Spanish
            context_before=3,
            context_after=1
        )

        # Translate
        stats = batch_translator.translate_file(
            parsed_yaml_path=temp_parsed_yaml,
            tags_yaml_path=temp_tags_yaml,
            output_yaml_path=None
        )

        print(f"\n[Statistics]:")
        print(f"    Total: {stats['total']}, Translated: {stats['translated']}")

        assert stats['translated'] == 2, f"Expected 2 translations, got {stats['translated']}"

        # Verify Spanish translations were written
        with open(temp_parsed_yaml, 'r', encoding='utf-8') as f:
            updated_blocks = yaml.safe_load(f)

        assert updated_blocks['1-Amelia']['es'] != '', "Spanish translation missing"
        assert updated_blocks['2-MainCharacter']['es'] != '', "Spanish translation missing"

        print(f"    [OK] Spanish block 1: {updated_blocks['1-Amelia']['es']}")
        print(f"    [OK] Spanish block 2: {updated_blocks['2-MainCharacter']['es']}")

        print("\n[OK] Language-agnostic test passed!")
        return True

    finally:
        temp_parsed_yaml.unlink(missing_ok=True)
        temp_tags_yaml.unlink(missing_ok=True)


# ── helpers for Stage 2 batch-path tests ──────────────────────────────────────

class MockBatchTranslator(MockTranslator):
    """MockTranslator that also exposes translate_batch(), tracking which path was used."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_calls = []   # texts lists passed to translate_batch
        self.single_calls = []  # individual texts passed to translate (via super)

    def translate(self, text: str, context=None, speaker=None, **kwargs) -> str:
        self.single_calls.append(text)
        return super().translate(text, context=context, speaker=speaker, **kwargs)

    def translate_batch(self, texts: list) -> list:
        self.batch_calls.append(list(texts))
        return [f"[BATCH] {t}" for t in texts]


def _make_simple_fixture(n_blocks=4):
    """Return (parsed_blocks, tags_file) with n_blocks untranslated dialogue lines."""
    parsed_blocks = {f"{i}-Char": {"en": f"Line {i}", "ro": ""} for i in range(1, n_blocks + 1)}
    tags_file = {
        "metadata": {
            "file_structure_type": "DIALOGUE_ONLY",
            "source_file": "test.rpy",
            "target_language": "romanian",
            "source_language": "english",
            "extracted_at": "2026-01-01 00:00:00",
            "total_blocks": n_blocks,
            "untranslated_blocks": n_blocks,
        },
        "structure": {
            "block_order": [f"{i}-Char" for i in range(1, n_blocks + 1)],
            "separator_positions": [],
        },
        "blocks": {},
        "character_map": {"c": "Char"},
    }
    return parsed_blocks, tags_file


def _write_fixtures(parsed_blocks, tags_file):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".parsed.yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(parsed_blocks, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        parsed_path = Path(f.name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tags.yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tags_path = Path(f.name)
    return parsed_path, tags_path


# ── Stage 2 tests ──────────────────────────────────────────────────────────────

def test_batch_mode_uses_translate_batch_not_translate():
    """translate_file uses translate_batch() when hf_batch_size > 1 and the method exists."""
    parsed_blocks, tags_file = _make_simple_fixture(n_blocks=4)
    parsed_path, tags_path = _write_fixtures(parsed_blocks, tags_file)
    try:
        translator = MockBatchTranslator(target_language="Romanian")
        bt = ModularBatchTranslator(
            translator=translator,
            characters={},
            target_lang_code="ro",
            hf_batch_size=2,
        )
        stats = bt.translate_file(parsed_path, tags_path)

        assert stats["translated"] == 4
        assert stats["failed"] == 0
        # translate_batch received all texts split into batches of 2
        assert translator.batch_calls == [["Line 1", "Line 2"], ["Line 3", "Line 4"]]
        # translate() must not have been called directly
        assert translator.single_calls == []
    finally:
        parsed_path.unlink(missing_ok=True)
        tags_path.unlink(missing_ok=True)


def test_hf_batch_size_1_uses_single_translate_even_when_translate_batch_exists():
    """hf_batch_size=1 forces single-item translate() even if translate_batch is present."""
    parsed_blocks, tags_file = _make_simple_fixture(n_blocks=3)
    parsed_path, tags_path = _write_fixtures(parsed_blocks, tags_file)
    try:
        translator = MockBatchTranslator(target_language="Romanian")
        bt = ModularBatchTranslator(
            translator=translator,
            characters={},
            target_lang_code="ro",
            hf_batch_size=1,
        )
        stats = bt.translate_file(parsed_path, tags_path)

        assert stats["translated"] == 3
        assert translator.batch_calls == []          # batch path never entered
        assert len(translator.single_calls) == 3    # one call per block
    finally:
        parsed_path.unlink(missing_ok=True)
        tags_path.unlink(missing_ok=True)

