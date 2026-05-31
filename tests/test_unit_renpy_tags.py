"""
Test Ren'Py tag extraction and restoration logic
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
import yaml
from renpy_utils import RenpyTagExtractor


def test_tag_extraction():
    """Test tag extraction from various Ren'Py formatted strings"""

    test_cases = [
        # (original, expected_clean, expected_tags_count)
        (
            "Hello {color=#fff}world{/color}!",
            "Hello world!",
            2
        ),
        (
            "My name is [name]",
            "My name is",
            1
        ),
        (
            "{size=20}{color=#797979}02/2023{/color}{/size}",
            "02/2023",
            4
        ),
        (
            "See you later [name]!",
            "See you later!",
            1
        ),
        (
            "{color=#ff0000}Red text{/color} and {color=#00ff00}green text{/color}",
            "Red text and green text",
            4
        ),
        (
            "Plain text without tags",
            "Plain text without tags",
            0
        ),
        (
            "{b}Bold{/b} and {i}italic{/i} with [variable]",
            "Bold and italic with",
            5
        ),
    ]

    print("Testing Tag Extraction")
    print("=" * 70)

    all_passed = True

    for i, (original, expected_clean, expected_tag_count) in enumerate(test_cases, 1):
        clean_text, tags = RenpyTagExtractor.extract_tags(original)

        passed = (
            clean_text == expected_clean and
            len(tags) == expected_tag_count
        )

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\nTest {i}: {status}")
        print(f"  Original:  {original}")
        print(f"  Clean:     {clean_text}")
        print(f"  Expected:  {expected_clean}")
        print(f"  Tags:      {len(tags)} (expected {expected_tag_count})")

        if tags:
            print(f"  Tag list:  {[tag for _, tag in tags]}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] All tag extraction tests passed!")
    else:
        print("[FAIL] Some tag extraction tests failed")

    return all_passed


def test_tag_restoration():
    """Test tag restoration after translation simulation"""

    test_cases = [
        # (original, simulated_translation, should_contain_tags)
        (
            "Hello {color=#fff}world{/color}!",
            "Salut lume!",
            ["{color=#fff}", "{/color}"]
        ),
        (
            "My name is [name]",
            "Numele meu este",
            ["[name]"]
        ),
        (
            "{size=20}Big text{/size}",
            "Text mare",
            ["{size=20}", "{/size}"]
        ),
    ]

    print("\n\nTesting Tag Restoration")
    print("=" * 70)

    all_passed = True

    for i, (original, translation, expected_tags) in enumerate(test_cases, 1):
        # Extract tags
        clean_original, tags = RenpyTagExtractor.extract_tags(original)

        # Restore tags
        restored = RenpyTagExtractor.restore_tags(translation, tags, clean_original)

        # Check if all expected tags are present
        tags_present = all(tag in restored for tag in expected_tags)

        passed = tags_present
        status = "[PASS]" if passed else "[FAIL]"

        print(f"\nTest {i}: {status}")
        print(f"  Original:     {original}")
        print(f"  Translation:  {translation}")
        print(f"  Restored:     {restored}")
        print(f"  Expected tags: {expected_tags}")
        print(f"  All present:  {tags_present}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] All tag restoration tests passed!")
    else:
        print("[FAIL] Some tag restoration tests failed")

    return all_passed


if __name__ == "__main__":
    results = []

    results.append(test_tag_extraction())
    results.append(test_tag_restoration())

    print("\n\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)

    if all(results):
        print("[OK] All tests passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed")
        sys.exit(1)
