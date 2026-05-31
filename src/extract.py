"""
Ren'Py Translation File Extractor

Extracts translation blocks from .rpy files and separates clean text from tags.
Outputs two files:
- .parsed.yaml: Human-readable clean text for translation
- .tags.yaml: Machine-readable metadata and tags for reconstruction
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

try:
    # Try relative import (when used as package)
    from .models import (
        BlockType, FileStructureType, CharacterType,
        RenpyBlock, ParsedBlock, TaggedBlock, TagInfo,
        ParsedFileMetadata, FileStructure, TagsFileContent,
        get_character_display_name, create_block_id
    )
    from .renpy_utils import RenpyTagExtractor
except ImportError:
    # Fall back to absolute import (when run standalone or from tests)
    from models import (
        BlockType, FileStructureType, CharacterType,
        RenpyBlock, ParsedBlock, TaggedBlock, TagInfo,
        ParsedFileMetadata, FileStructure, TagsFileContent,
        get_character_display_name, create_block_id
    )
    from renpy_utils import RenpyTagExtractor


class RenpyExtractor:
    """Extract and parse Ren'Py translation files."""

    # ========================================================================
    # REGEX PATTERNS
    # ========================================================================

    # Dialogue block: has character variable
    # # game/Cell01_JM.rpy:6256
    # translate romanian QuestChap2_Jasmine01_18_3deb0746:
    #
    #     # jm "See you later [name]. Bye!"
    #     jm "Translation here"
    DIALOGUE_PATTERN = re.compile(
        r'# (game/[^\n]+)\n'  # Location comment
        r'translate (\w+) (\w+):\s*\n'  # translate <language> <label>:
        r'\s*\n'  # Empty line
        r'\s*# (\w+) "([^"\\]*(?:\\.[^"\\]*)*)"\s*\n'  # Original with char_var
        r'\s*(\w+) "([^"\\]*(?:\\.[^"\\]*)*)"'  # Translation with char_var
    , re.MULTILINE | re.DOTALL)

    # Narrator block: NO character variable (just quoted text)
    # # game/Room_01.rpy:106
    # translate romanian room01_mail01_92caf0b8:
    #
    #     # "You don't have any mail!"
    #     ""
    NARRATOR_PATTERN = re.compile(
        r'# (game/[^\n]+)\n'  # Location comment
        r'translate (\w+) (\w+):\s*\n'  # translate <language> <label>:
        r'\s*\n'  # Empty line
        r'\s*# "([^"\\]*(?:\\.[^"\\]*)*)"\s*\n'  # Original (no char_var)
        r'\s*"([^"\\]*(?:\\.[^"\\]*)*)"'  # Translation (no char_var)
    , re.MULTILINE | re.DOTALL)

    # String block (inside "translate <lang> strings:" section)
    # # game/Cell01_JM.rpy:184
    # old "{color=#3ad8ff}Text here{/color}"
    # new "Translation"
    STRING_PATTERN = re.compile(
        r'# (game/[^\n]+)\n'  # Location comment
        r'\s*old "([^"\\]*(?:\\.[^"\\]*)*)"\s*\n'  # old "original"
        r'\s*new "([^"\\]*(?:\\.[^"\\]*)*)"'  # new "translation"
    , re.MULTILINE | re.DOTALL)

    # Separator line
    SEPARATOR_PATTERN = re.compile(
        r'^(# -+)$',
        re.MULTILINE
    )

    # Strings section header
    STRINGS_SECTION_PATTERN = re.compile(
        r'translate (\w+) strings:',
        re.MULTILINE
    )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self, character_map: Optional[Dict[str, str]] = None):
        """
        Initialize extractor.

        Args:
            character_map: Optional mapping of char_var -> display name
        """
        self.character_map = character_map or {}
        self.tag_extractor = RenpyTagExtractor()

    # ========================================================================
    # MAIN EXTRACTION METHOD
    # ========================================================================

    def extract_file(
        self,
        rpy_file_path: Path,
        target_language: str = "romanian",
        source_language: str = "english"
    ) -> Tuple[Dict[str, ParsedBlock], TagsFileContent]:
        """
        Extract translation blocks from a .rpy file.

        Args:
            rpy_file_path: Path to the .rpy file
            target_language: Target language code (e.g., "romanian")
            source_language: Source language code (e.g., "english")

        Returns:
            Tuple of (parsed_blocks, tags_file_content)
            - parsed_blocks: Dict[block_id, ParsedBlock] for YAML output
            - tags_file_content: Complete tags.yaml structure
        """
        print(f"Reading file: {rpy_file_path}")

        # Read file content
        with open(rpy_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Detect file structure
        file_structure_type = self._detect_file_structure(content)
        print(f"File structure: {file_structure_type.value}")

        # Parse blocks
        blocks = self._parse_blocks(content, target_language)
        print(f"Found {len(blocks)} blocks")

        # Detect separator lines
        has_separators = bool(self.SEPARATOR_PATTERN.search(content))

        # Extract separators and add to blocks
        if has_separators:
            blocks.extend(self._extract_separators(content))
            print(f"[Separators] Found {len([b for b in blocks if b['type'] == 'separator'])} separator lines")

        # Sort blocks by position to preserve order
        blocks.sort(key=lambda b: b.get('start_pos', 0))

        # Build parsed blocks and tags
        parsed_blocks: Dict[str, ParsedBlock] = {}
        tagged_blocks: Dict[str, TaggedBlock] = {}
        block_order: List[str] = []
        string_section_start: Optional[int] = None

        for idx, block in enumerate(blocks, start=1):
            block_type = block['type']

            if block_type == 'separator':
                # Separator block - add to tags_file but NOT to parsed_blocks (YAML)
                block_id = f"separator-{idx}"
                tagged_blocks[block_id] = {
                    "type": BlockType.SEPARATOR.value,
                    "label": None,
                    "location": None,
                    "char_var": None,
                    "char_name": None,
                    "tags": [],
                    "template": "",
                    "separator_content": block['content']
                }
                block_order.append(block_id)
                continue

            # Get character info
            char_var = block.get('character_var')
            char_name = get_character_display_name(char_var, self.character_map)

            # Check if this is the start of strings section
            if block_type == 'string' and string_section_start is None:
                string_section_start = idx - 1  # 0-based index

            # Extract tags from original text
            original_text = block['original']
            clean_en, tags = self.tag_extractor.extract_tags(original_text)

            # Extract tags from current translation (if exists)
            current_translation = block['current_translation']
            clean_ro = ""
            if current_translation and current_translation.strip():
                clean_ro, _ = self.tag_extractor.extract_tags(current_translation)

            # Create block ID
            block_id = create_block_id(idx, char_name)
            block_order.append(block_id)

            # Add to parsed blocks (YAML)
            parsed_blocks[block_id] = {
                "en": clean_en,
                "ro": clean_ro
            }

            # Convert tags to TagInfo format
            tag_list: List[TagInfo] = [
                {"pos": pos, "tag": tag, "type": self._classify_tag(tag)}
                for pos, tag in tags
            ]

            # Build reconstruction template
            template = self._build_template(block)

            # Add to tagged blocks (YAML)
            tagged_blocks[block_id] = {
                "type": BlockType(block_type).value,
                "label": block.get('label'),
                "location": block.get('location'),
                "char_var": char_var,
                "char_name": char_name,
                "tags": tag_list,
                "template": template,
                "separator_content": None
            }

        # Build metadata (excluding separators from counts)
        non_separator_blocks = [b for b in blocks if b['type'] != 'separator']
        metadata: ParsedFileMetadata = {
            "source_file": str(rpy_file_path),
            "target_language": target_language,
            "source_language": source_language,
            "extracted_at": datetime.now().isoformat(),
            "file_structure_type": file_structure_type.value,
            "has_separator_lines": has_separators,
            "total_blocks": len(non_separator_blocks),
            "untranslated_blocks": len([b for b in non_separator_blocks if not b.get('current_translation', '').strip()])
        }

        # Build structure
        structure: FileStructure = {
            "block_order": block_order,
            "string_section_start": string_section_start,
            "string_section_header": f"translate {target_language} strings:" if string_section_start is not None else None
        }

        # Build tags file content
        tags_file: TagsFileContent = {
            "metadata": metadata,
            "structure": structure,
            "blocks": tagged_blocks,
            "character_map": self.character_map
        }

        return parsed_blocks, tags_file

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _detect_file_structure(self, content: str) -> FileStructureType:
        """Detect if file is strings-only or has dialogue blocks."""
        # Check if file starts with "translate <lang> strings:"
        first_translate = re.search(r'translate \w+ (\w+)', content)
        if first_translate and first_translate.group(1) == 'strings':
            return FileStructureType.STRINGS_ONLY

        # Check for dialogue blocks
        if self.DIALOGUE_PATTERN.search(content) or self.NARRATOR_PATTERN.search(content):
            return FileStructureType.DIALOGUE_AND_STRINGS

        # Default to strings only
        return FileStructureType.STRINGS_ONLY

    def _parse_blocks(self, content: str, target_language: str) -> List[RenpyBlock]:
        """Parse all translation blocks from file content."""
        blocks: List[RenpyBlock] = []

        # Parse dialogue blocks
        for match in self.DIALOGUE_PATTERN.finditer(content):
            location, lang, label, char_var_orig, original, char_var_trans, translation = match.groups()
            blocks.append({
                'type': 'dialogue',
                'label': label,
                'location': location,
                'character_var': char_var_trans,
                'original': original,
                'current_translation': translation,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'full_match': match.group(0)
            })

        # Parse narrator blocks
        for match in self.NARRATOR_PATTERN.finditer(content):
            location, lang, label, original, translation = match.groups()
            blocks.append({
                'type': 'narrator',
                'label': label,
                'location': location,
                'character_var': None,
                'original': original,
                'current_translation': translation,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'full_match': match.group(0)
            })

        # Parse string blocks
        for match in self.STRING_PATTERN.finditer(content):
            location, original, translation = match.groups()
            blocks.append({
                'type': 'string',
                'label': 'strings',
                'location': location,
                'character_var': None,
                'original': original,
                'current_translation': translation,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'full_match': match.group(0)
            })

        return blocks

    def _extract_separators(self, content: str) -> List[RenpyBlock]:
        """Extract separator lines from content."""
        separators: List[RenpyBlock] = []
        for match in self.SEPARATOR_PATTERN.finditer(content):
            separators.append({
                'type': 'separator',
                'content': match.group(1),
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        return separators

    def _classify_tag(self, tag: str) -> Optional[str]:
        """Classify tag type for easier debugging."""
        if tag.startswith('{color'):
            return 'color'
        elif tag.startswith('{size'):
            return 'size'
        elif tag.startswith('{font'):
            return 'font'
        elif tag.startswith('{cps'):
            return 'cps'
        elif tag.startswith('{/'):
            return 'close'
        elif tag.startswith('{image'):
            return 'image'
        elif tag.startswith('[') and tag.endswith(']'):
            return 'variable'
        else:
            return 'other'

    def _build_template(self, block: RenpyBlock) -> str:
        """Build reconstruction template for a block."""
        block_type = block['type']

        if block_type == 'dialogue':
            return (
                "# {location}\n"
                "translate {language} {label}:\n"
                "\n"
                "    # {char_var} \"{original}\"\n"
                "    {char_var} \"{translation}\""
            )
        elif block_type == 'narrator':
            return (
                "# {location}\n"
                "translate {language} {label}:\n"
                "\n"
                "    # \"{original}\"\n"
                "    \"{translation}\""
            )
        elif block_type == 'string':
            return (
                "    # {location}\n"
                "    old \"{original}\"\n"
                "    new \"{translation}\""
            )
        else:
            return ""

    # ========================================================================
    # SAVE METHODS
    # ========================================================================

    def save_parsed_yaml(self, parsed_blocks: Dict[str, ParsedBlock], output_path: Path):
        """Save parsed blocks to YAML file."""
        print(f"Saving parsed YAML to: {output_path}")

        # Add header comment
        header = (
            f"# {output_path.stem} - Parsed Translations\n"
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "\n"
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            yaml.dump(parsed_blocks, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def save_tags_yaml(self, tags_file: TagsFileContent, output_path: Path):
        """Save tags file to YAML."""
        print(f"Saving tags YAML to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(tags_file, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def show_banner():
    """Display the extraction banner"""
    print()
    print("=" * 70)
    print("               Translation File Extraction                 ")
    print("=" * 70)
    print()


def load_game_config(game_name: str) -> Dict[str, any]:
    """Load game configuration from current_config.yaml"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "models" / "current_config.yaml"

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please run 1-config.ps1 first to configure a game.")
        exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Get the game configuration
    games = config.get('games', {})
    if game_name not in games:
        print(f"Error: Game '{game_name}' not found in configuration.")
        print(f"Available games: {', '.join(games.keys())}")
        exit(1)

    return games[game_name]


def load_character_map(game_path: Path, language_dir: str) -> Dict[str, str]:
    """Load character map from characters.yaml"""
    characters_path = game_path / "game" / "tl" / language_dir / "characters.yaml"

    if not characters_path.exists():
        print(f"Warning: characters.yaml not found at {characters_path}")
        return {}

    with open(characters_path, 'r', encoding='utf-8') as f:
        chars = yaml.safe_load(f)

    # Convert to simple char_var -> name mapping
    return {k: v.get('name', k.upper()) for k, v in chars.items()}


def extract_single_file(
    file_path: Path,
    game_path: Path,
    language_dir: str,
    character_map: Dict[str, str]
):
    """Extract a single .rpy file"""
    print(f"\nExtracting: {file_path.name}")

    # Prepare output paths
    base_name = file_path.stem
    output_dir = file_path.parent
    yaml_path = output_dir / f"{base_name}.parsed.yaml"
    tags_yaml_path = output_dir / f"{base_name}.tags.yaml"

    # Extract
    extractor = RenpyExtractor(character_map)
    parsed_blocks, tags_file = extractor.extract_file(
        file_path,
        target_language=language_dir,
        source_language='english'
    )

    # Save files
    extractor.save_parsed_yaml(parsed_blocks, yaml_path)
    extractor.save_tags_yaml(tags_file, tags_yaml_path)

    print(f"\nExtraction complete!")
    print(f"  Parsed YAML: {yaml_path}")
    print(f"  Tags YAML: {tags_yaml_path}")


def find_rpy_files(tl_path: Path) -> List[Path]:
    """Find source .rpy files, excluding generated merge outputs (*.translated.rpy)."""
    all_files = list(tl_path.rglob("*.rpy"))
    return [
        f for f in all_files
        if '.translated.' not in f.name
        and not f.name.endswith('.parsed.rpy')
        and not f.name.endswith('.tags.rpy')
    ]


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract translation blocks from Ren\'Py files')
    parser.add_argument('--game-name', type=str, default='', help='Game name from configuration')
    parser.add_argument('--source', type=str, default='', help='Specific .rpy file to extract')
    parser.add_argument('--all', action='store_true', help='Extract all .rpy files')

    args = parser.parse_args()

    show_banner()

    # Load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "models" / "current_config.yaml"

    if not config_path.exists():
        print("Error: Configuration file not found.")
        print("Please run 1-config.ps1 first to configure a game.")
        exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Determine which game to use
    if args.game_name:
        game_name = args.game_name
    else:
        game_name = config.get('current_game')
        if not game_name:
            print("Error: No game configured.")
            print("Please run 1-config.ps1 first.")
            exit(1)

    # Get game config
    game_config = load_game_config(game_name)
    game_path = Path(game_config['path'])
    language_info = game_config['target_language']
    language_code = language_info.get('code', language_info.get('name', 'unknown').lower())
    language_dir = language_info['name'].lower()  # Directory name is lowercase language name

    print(f"Game: {game_config['name']}")
    print(f"Language: {language_info['name']} ({language_code})")
    print(f"Path: {game_path}")
    print()

    # Load character map
    character_map = load_character_map(game_path, language_dir)

    # Get translation directory
    tl_path = game_path / "game" / "tl" / language_dir

    if not tl_path.exists():
        print(f"Error: Translation directory not found: {tl_path}")
        exit(1)

    # Determine what to extract
    if args.all:
        # Extract all files
        print("Finding all .rpy files...")
        rpy_files = find_rpy_files(tl_path)

        if not rpy_files:
            print("No .rpy files found!")
            exit(1)

        print(f"Found {len(rpy_files)} files\n")

        for file_path in rpy_files:
            extract_single_file(file_path, game_path, language_dir, character_map)
            print()

    elif args.source:
        # Extract specific file
        if args.source.endswith('.rpy'):
            file_path = tl_path / args.source
        else:
            file_path = tl_path / f"{args.source}.rpy"

        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            exit(1)

        extract_single_file(file_path, game_path, language_dir, character_map)

    else:
        # Interactive mode - ask user
        print("Extract Options:")
        print("  [1] Extract all .rpy files")
        print("  [2] Extract specific file")
        print()

        choice = input("Select option (1-2): ").strip()

        if choice == "1":
            # Extract all
            print("\nFinding all .rpy files...")
            rpy_files = find_rpy_files(tl_path)

            if not rpy_files:
                print("No .rpy files found!")
                exit(1)

            print(f"Found {len(rpy_files)} files\n")

            for file_path in rpy_files:
                extract_single_file(file_path, game_path, language_dir, character_map)
                print()

        elif choice == "2":
            # List and select file
            print("\nAvailable .rpy files:")
            print()

            rpy_files = find_rpy_files(tl_path)

            if not rpy_files:
                print("No .rpy files found!")
                exit(1)

            for i, file_path in enumerate(rpy_files, 1):
                print(f"  [{i}] {file_path.name}")

            print()
            selection = input(f"Select file (1-{len(rpy_files)}): ").strip()

            try:
                index = int(selection) - 1
                if 0 <= index < len(rpy_files):
                    selected_file = rpy_files[index]
                    print()
                    extract_single_file(selected_file, game_path, language_dir, character_map)
                else:
                    print("Invalid selection!")
                    exit(1)
            except ValueError:
                print("Invalid input!")
                exit(1)

        else:
            print("Invalid option!")
            exit(1)

    print()
    print("Extraction complete!")
    print()
    print("Next steps:")
    print("  1. Review the .parsed.yaml files for any issues")
    print("  2. Run '3-translate.ps1' to translate untranslated blocks")
    print()


if __name__ == "__main__":
    main()
