"""
Data models for the Ren'Py translation pipeline.

This module defines TypedDict classes and enums for type-safe data handling
across the extraction, translation, and merge phases.
"""

from typing import TypedDict, Literal, Optional, List, Dict
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class BlockType(str, Enum):
    """Types of blocks in a Ren'Py translation file."""
    DIALOGUE = "dialogue"      # Character dialogue: character_var "text"
    NARRATOR = "narrator"      # Narration with no character variable
    STRING = "string"          # Menu/UI strings (old/new format)
    SEPARATOR = "separator"    # Separator lines (# ----)


class FileStructureType(str, Enum):
    """Overall structure type of a translation file."""
    DIALOGUE_AND_STRINGS = "dialogue_and_strings"  # Has dialogue blocks + strings section
    STRINGS_ONLY = "strings_only"                  # Only has strings section


class CharacterType(str, Enum):
    """Character types for classification."""
    MAIN = "main"
    PROTAGONIST = "protagonist"
    SUPPORTING = "supporting"
    SYSTEM = "system"
    NARRATOR = "narrator"


# ============================================================================
# TYPED DICTIONARIES - BLOCKS
# ============================================================================

class RenpyBlock(TypedDict, total=False):
    """
    Represents a single translation block from a .rpy file.

    Used during parsing and extraction phase.
    """
    type: Literal["dialogue", "narrator", "string", "separator"]
    label: str                      # Translation label (e.g., "QuestChap2_Jasmine01_18_3deb0746")
    location: str                   # Source location (e.g., "game/Cell01_JM.rpy:6256")
    character_var: Optional[str]    # Character variable (e.g., "jm", "u") - None for narrator/strings
    original: str                   # Original English text (with tags)
    current_translation: str        # Current translation (empty if untranslated)
    full_match: str                 # Full regex match from file
    start_pos: int                  # Character position in file (for legacy support)
    end_pos: int                    # Character position in file (for legacy support)


class ParsedBlock(TypedDict, total=False):
    """
    Represents a block after extraction (clean text, no tags).

    Used in YAML format for human editing.
    """
    en: str                         # Clean English text (tags removed)
    ro: str                         # Clean Romanian translation (tags removed)
    type: Optional[str]             # Only present for separators


class TaggedBlock(TypedDict):
    """
    Represents tag information for a single block.

    Stored in tags.json file.
    """
    type: Literal["dialogue", "narrator", "string", "separator"]
    label: Optional[str]            # Translation label
    location: Optional[str]         # Source location comment
    char_var: Optional[str]         # Character variable
    char_name: Optional[str]        # Character display name (from characters.json)
    tags: List['TagInfo']           # List of tags with positions
    template: str                   # Template for reconstruction
    separator_content: Optional[str]  # Content if type is separator


class TagInfo(TypedDict):
    """Information about a single tag in text."""
    pos: int                        # Position in clean text (before tag insertion)
    tag: str                        # The tag itself (e.g., "{color=#fff}", "[name]")
    type: Optional[str]             # Type hint: "color", "variable", "size", etc.


# ============================================================================
# TYPED DICTIONARIES - FILE STRUCTURES
# ============================================================================

class ParsedFileMetadata(TypedDict):
    """Metadata about a parsed translation file."""
    source_file: str                # Original file path
    target_language: str            # Target language code (e.g., "romanian")
    source_language: str            # Source language code (e.g., "english")
    extracted_at: str               # ISO timestamp
    file_structure_type: Literal["dialogue_and_strings", "strings_only"]
    has_separator_lines: bool       # Whether file contains separator lines
    total_blocks: int               # Total number of blocks
    untranslated_blocks: int        # Number of untranslated blocks


class FileStructure(TypedDict):
    """Structure information for file reconstruction."""
    block_order: List[str]          # Ordered list of block IDs (e.g., ["1-Jasmine", "2-Narrator"])
    string_section_start: Optional[int]  # Block index where strings section starts
    string_section_header: Optional[str]  # Header line (e.g., "translate romanian strings:")


class TagsFileContent(TypedDict):
    """Complete content of a .tags.json file."""
    metadata: ParsedFileMetadata
    structure: FileStructure
    blocks: Dict[str, TaggedBlock]  # Block ID -> TaggedBlock
    character_map: Dict[str, str]   # Character variable -> Display name


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_character_display_name(
    char_var: Optional[str],
    character_map: Dict[str, str]
) -> str:
    """
    Get display name for a character variable.

    Args:
        char_var: Character variable (e.g., "jm", None for narrator)
        character_map: Mapping of variables to display names

    Returns:
        Display name (e.g., "Jasmine", "Narrator", "Choice")
    """
    if char_var is None or char_var == "":
        return "Narrator"
    return character_map.get(char_var, char_var.upper())


def create_block_id(index: int, char_name: str) -> str:
    """
    Create a composite block ID.

    Args:
        index: Block index (1-based)
        char_name: Character display name

    Returns:
        Composite ID like "1-Jasmine" or "42-Choice"
    """
    return f"{index}-{char_name}"


def parse_block_id(block_id: str) -> tuple[int, str]:
    """
    Parse a composite block ID.

    Args:
        block_id: Composite ID like "1-Jasmine"

    Returns:
        Tuple of (index, character_name)
    """
    parts = block_id.split('-', 1)
    if len(parts) == 2:
        try:
            return int(parts[0]), parts[1]
        except ValueError:
            pass
    # Fallback for invalid format
    return 0, block_id


def is_separator_block(block_id: str, block_data: ParsedBlock) -> bool:
    """
    Check if a block is a separator.

    Args:
        block_id: Block ID
        block_data: Parsed block data

    Returns:
        True if block is a separator
    """
    return (
        block_id.startswith("separator-") or
        block_data.get("type") == "separator"
    )
