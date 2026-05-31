"""
Ren'Py Translation Utilities

Common utilities for parsing and processing Ren'Py translation files.
Shared across all translation backends (Aya-23-8B, MADLAD-400-3B, etc.)
"""

import re
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, TypedDict


# Define a TypedDict for a Ren'Py translation block
class RenpyBlock(TypedDict):
    type: str # 'dialogue' or 'string'
    label: str
    original: str
    character_var: Optional[str] # For dialogue blocks
    current_translation: str
    start_pos: int
    end_pos: int
    full_match: str
    source_language: str # Language identifier from the 'translate' statement

_PROGRESS_DOTS_PER_LINE = 50


def show_progress(current, total, start_time, prefix=""):
    """Append one dot per tick; emit a status line every 50 ticks (and at the end).
    Avoids \\r-style overwriting since some terminals/pipes treat it as a newline."""
    if total <= 0:
        return
    if (current - 1) % _PROGRESS_DOTS_PER_LINE == 0:
        print(prefix, end="", flush=True)
    print(".", end="", flush=True)
    if current % _PROGRESS_DOTS_PER_LINE == 0 or current == total:
        elapsed = time.time() - start_time
        rate = current / elapsed if (current > 0 and elapsed > 0) else 0
        remaining = (total - current) / rate if rate > 0 else 0
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
        remaining_str = f"{int(remaining // 60)}m {int(remaining % 60)}s" if remaining >= 60 else f"{int(remaining)}s"
        percentage = (current / total) * 100
        print(f" {current}/{total} ({percentage:.0f}%) | {elapsed_str} elapsed | ETA {remaining_str}",
              flush=True)


# Canonical language registry (path name -> (proper name, code)).
# Single source of truth for path detection and code/name lookups across the project.
LANGUAGE_MAP = {
    'romanian': ('Romanian', 'ro'),
    'spanish': ('Spanish', 'es'),
    'french': ('French', 'fr'),
    'german': ('German', 'de'),
    'italian': ('Italian', 'it'),
    'portuguese': ('Portuguese', 'pt'),
    'russian': ('Russian', 'ru'),
    'turkish': ('Turkish', 'tr'),
    'czech': ('Czech', 'cs'),
    'polish': ('Polish', 'pl'),
    'ukrainian': ('Ukrainian', 'uk'),
    'bulgarian': ('Bulgarian', 'bg'),
    'chinese': ('Chinese', 'zh'),
    'japanese': ('Japanese', 'ja'),
    'korean': ('Korean', 'ko'),
    'vietnamese': ('Vietnamese', 'vi'),
    'thai': ('Thai', 'th'),
    'indonesian': ('Indonesian', 'id'),
    'arabic': ('Arabic', 'ar'),
    'hebrew': ('Hebrew', 'he'),
    'persian': ('Persian', 'fa'),
    'hindi': ('Hindi', 'hi'),
    'bengali': ('Bengali', 'bn'),
    'dutch': ('Dutch', 'nl'),
    'swedish': ('Swedish', 'sv'),
    'norwegian': ('Norwegian', 'no'),
    'danish': ('Danish', 'da'),
    'finnish': ('Finnish', 'fi'),
    'greek': ('Greek', 'el'),
    'hungarian': ('Hungarian', 'hu'),
}

# code -> proper name, derived from LANGUAGE_MAP
_CODE_TO_NAME = {code: name for name, code in LANGUAGE_MAP.values()}


def detect_language_from_path(path: Path) -> Tuple[str, str]:
    """
    Auto-detect target language from path (e.g., "game/tl/romanian" → ("Romanian", "ro"))

    Returns:
        Tuple of (language_name, language_code)
    """
    path_str = str(path).lower().replace('\\', '/')

    # Check each language in path
    for path_lang, (proper_lang, code) in LANGUAGE_MAP.items():
        if f'/{path_lang}/' in path_str or path_str.endswith(f'/{path_lang}') or path_str.endswith(f'{path_lang}'):
            return proper_lang, code

    # Default to Romanian if not detected
    return 'Romanian', 'ro'


def language_name_from_code(code: str) -> str:
    """
    Map a language code (e.g. "ro") to its proper name (e.g. "Romanian").

    Falls back to the capitalized code when unknown.
    """
    return _CODE_TO_NAME.get(code, code.capitalize())


class RenpyTagExtractor:
    """Extract and restore Ren'Py tags and variables"""

    # Patterns for Ren'Py formatting
    TAG_PATTERN = re.compile(r'\{[^}]+\}')  # {color=#fff}, {/color}, etc.
    VAR_PATTERN = re.compile(r'\[[^\]]+\]')  # [name], [variable]

    @classmethod
    def extract_tags(cls, text: str) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract tags and variables from text, return clean text and tag positions

        Returns:
            (clean_text, [(position, tag), ...])
        """
        tags = []
        clean_text = text

        # Find all tags and variables
        all_matches = []
        for match in cls.TAG_PATTERN.finditer(text):
            all_matches.append((match.start(), match.group()))
        for match in cls.VAR_PATTERN.finditer(text):
            all_matches.append((match.start(), match.group()))

        # Sort by position (reverse order for removal)
        all_matches.sort(key=lambda x: x[0], reverse=True)

        # Remove tags from text and store positions
        for pos, tag in all_matches:
            # Calculate position in words/chars for restoration
            before_tag = text[:pos]
            tags.insert(0, (len(before_tag), tag))
            clean_text = clean_text[:pos] + clean_text[pos + len(tag):]

        # Clean up extra spaces left after tag removal
        # Remove multiple consecutive spaces
        clean_text = re.sub(r' +', ' ', clean_text)
        # Remove spaces before punctuation
        clean_text = re.sub(r' +([.,!?;:])', r'\1', clean_text)

        return clean_text.strip(), tags

    @classmethod
    def restore_tags(cls, translated_text: str, tags: List[Tuple[int, str]], original_text: str) -> str:
        """
        Restore tags into translated text based on relative positions

        Strategy:
        - If text length is similar, use proportional positions
        - Place tags at word boundaries when possible
        - Never insert tags inside other tags
        """
        if not tags:
            return translated_text

        result = translated_text
        original_len = len(original_text)
        translated_len = len(translated_text)

        # Sort tags by position for insertion
        sorted_tags = sorted(tags, key=lambda x: x[0], reverse=True)

        for orig_pos, tag in sorted_tags:
            # Calculate proportional position
            if original_len > 0:
                ratio = orig_pos / original_len
                new_pos = int(ratio * translated_len)
            else:
                new_pos = 0

            # Clamp position to text bounds
            new_pos = max(0, min(new_pos, len(result)))

            # CRITICAL FIX: Ensure we don't insert inside another tag
            # Check if we're inside a tag (between { and } or [ and ])
            safe_pos = cls._find_safe_insertion_point(result, new_pos)

            # Insert tag
            result = result[:safe_pos] + tag + result[safe_pos:]

        return result

    @staticmethod
    def _find_safe_insertion_point(text: str, target_pos: int) -> int:
        """
        Find a safe position to insert a tag, ensuring we don't break existing tags

        Args:
            text: The text to insert into
            target_pos: The desired insertion position

        Returns:
            A safe position that won't break existing tags
        """
        # Clamp to text bounds
        target_pos = max(0, min(target_pos, len(text)))

        # Check if we're inside a tag at target_pos
        # Count unclosed braces/brackets before this position
        before_text = text[:target_pos]

        # Count { and } before target
        open_braces = before_text.count('{')
        close_braces = before_text.count('}')

        # Count [ and ] before target
        open_brackets = before_text.count('[')
        close_brackets = before_text.count(']')

        # If we're inside a tag, find the end of it
        if open_braces > close_braces:
            # We're inside {}, find the next }
            next_close = text.find('}', target_pos)
            if next_close != -1:
                return next_close + 1

        if open_brackets > close_brackets:
            # We're inside [], find the next ]
            next_close = text.find(']', target_pos)
            if next_close != -1:
                return next_close + 1

        # Position is safe
        return target_pos

