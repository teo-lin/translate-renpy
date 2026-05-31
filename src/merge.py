"""
Ren'Py Translation File Merger

Merges translated YAML files back into .rpy format using tags metadata.
Performs integrity validation to ensure syntax correctness.
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    # Try relative import (when used as package)
    from .models import (
        ParsedBlock, TaggedBlock, TagInfo, TagsFileContent,
        FileStructureType, is_separator_block
    )
    from .renpy_utils import RenpyTagExtractor
    from .config_utils import load_game_config
except ImportError:
    # Fall back to absolute import (when run standalone or from tests)
    from models import (
        ParsedBlock, TaggedBlock, TagInfo, TagsFileContent,
        FileStructureType, is_separator_block
    )
    from renpy_utils import RenpyTagExtractor
    from config_utils import load_game_config


@dataclass
class ValidationError:
    """Represents a validation error found during integrity check."""
    block_id: str
    error_type: str
    message: str
    line_number: Optional[int] = None


class RenpyMerger:
    """Merge translated YAML files back into .rpy format."""

    def __init__(self):
        """Initialize merger."""
        self.tag_extractor = RenpyTagExtractor()
        self.validation_errors: List[ValidationError] = []

    # ========================================================================
    # MAIN MERGE METHOD
    # ========================================================================

    def merge_file(
        self,
        parsed_yaml_path: Path,
        tags_yaml_path: Path,
        output_rpy_path: Path,
        validate: bool = True
    ) -> bool:
        """
        Merge YAML + JSON back into .rpy file.

        Args:
            parsed_yaml_path: Path to .parsed.yaml file
            tags_yaml_path: Path to .tags.yaml file
            output_rpy_path: Path to output .rpy file
            validate: Whether to run integrity validation

        Returns:
            True if successful, False if validation errors found
        """
        print(f"Loading files for merge...")

        # Load files
        with open(parsed_yaml_path, 'r', encoding='utf-8') as f:
            parsed_blocks: Dict[str, ParsedBlock] = yaml.safe_load(f)

        with open(tags_yaml_path, 'r', encoding='utf-8') as f:
            tags_file: TagsFileContent = yaml.safe_load(f)

        metadata = tags_file['metadata']
        structure = tags_file['structure']
        tagged_blocks = tags_file['blocks']

        print(f"Loaded {len(parsed_blocks)} blocks")
        print(f"File structure: {metadata['file_structure_type']}")

        # Build output content
        output_lines: List[str] = []

        # Add header comment
        output_lines.append(f"# TODO: Translation updated at {metadata['extracted_at'][:10]}")
        output_lines.append("")

        # Determine structure type
        file_structure = FileStructureType(metadata['file_structure_type'])

        if file_structure == FileStructureType.STRINGS_ONLY:
            # File starts directly with strings section
            # Only append header if it exists
            if structure['string_section_header']:
                output_lines.append(structure['string_section_header'])
                output_lines.append("")

        # Process blocks in order
        block_order = structure['block_order']
        in_strings_section = file_structure == FileStructureType.STRINGS_ONLY
        string_section_start = structure.get('string_section_start')

        for idx, block_id in enumerate(block_order):
            # Check if we've reached strings section
            if not in_strings_section and string_section_start is not None and idx == string_section_start:
                in_strings_section = True
                output_lines.append("")
                output_lines.append(structure['string_section_header'])
                output_lines.append("")

            # Get block data
            parsed_block = parsed_blocks.get(block_id)
            tagged_block = tagged_blocks.get(block_id)

            if not parsed_block or not tagged_block:
                print(f"Warning: Missing data for block {block_id}")
                continue

            # Handle separator blocks
            if is_separator_block(block_id, parsed_block):
                separator_content = tagged_block.get('separator_content', '# -----------')
                output_lines.append("")
                output_lines.append(separator_content)
                output_lines.append("")
                continue

            # Get clean translations
            clean_en = parsed_block.get('en', '')
            clean_ro = parsed_block.get('ro', '')

            # Restore tags to both original and translation
            tags = tagged_block['tags']
            tag_list = [(tag['pos'], tag['tag']) for tag in tags]

            original_with_tags = self.tag_extractor.restore_tags(clean_en, tag_list, clean_en)
            translation_with_tags = self.tag_extractor.restore_tags(clean_ro, tag_list, clean_en)

            # Build block using template
            template = tagged_block['template']
            block_content = self._fill_template(
                template,
                language=metadata['target_language'],
                label=tagged_block.get('label', ''),
                location=tagged_block.get('location', ''),
                char_var=tagged_block.get('char_var', ''),
                original=original_with_tags,
                translation=translation_with_tags
            )

            output_lines.append(block_content)
            output_lines.append("")

        # Join output
        output_content = '\n'.join(output_lines)

        # Validate if requested
        if validate:
            print("Running integrity validation...")
            self.validation_errors = self.validate_content(output_content, block_order, parsed_blocks, tagged_blocks)

            if self.validation_errors:
                print(f"Found {len(self.validation_errors)} validation errors:")
                for error in self.validation_errors[:10]:  # Show first 10
                    print(f"   - {error.block_id}: {error.message}")
                print(f"\nFile will be saved, but please review errors!")
            else:
                print("Validation passed!")

        # Write output file
        print(f"Writing to: {output_rpy_path}")
        with open(output_rpy_path, 'w', encoding='utf-8') as f:
            f.write(output_content)

        return len(self.validation_errors) == 0

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _fill_template(
        self,
        template: str,
        language: str,
        label: str,
        location: str,
        char_var: str,
        original: str,
        translation: str
    ) -> str:
        """Fill a block template with values."""
        return template.format(
            language=language,
            label=label,
            location=location,
            char_var=char_var,
            original=original,
            translation=translation
        )

    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================

    def validate_content(
        self,
        content: str,
        block_order: List[str],
        parsed_blocks: Dict[str, ParsedBlock],
        tagged_blocks: Dict[str, TaggedBlock]
    ) -> List[ValidationError]:
        """
        Validate merged content for syntax errors.

        Checks for:
        - Unmatched quotes
        - Unmatched braces/brackets
        - Missing character variables
        - Missing variables from original in translation
        """
        errors: List[ValidationError] = []

        # Split content into lines for line-based errors
        lines = content.split('\n')

        # 1. Check for unmatched quotes in each line
        for line_num, line in enumerate(lines, start=1):
            # Skip comment lines and labels
            if line.strip().startswith('#') or line.strip().endswith(':'):
                continue

            # Count quotes (excluding escaped quotes)
            quote_count = len(re.findall(r'(?<!\\)"', line))
            if quote_count % 2 != 0:
                errors.append(ValidationError(
                    block_id="unknown",
                    error_type="unmatched_quote",
                    message=f"Unmatched quote on line {line_num}: {line[:50]}",
                    line_number=line_num
                ))

        # 2. Check for unmatched braces/brackets in entire file
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            errors.append(ValidationError(
                block_id="file",
                error_type="unmatched_brace",
                message=f"Unmatched braces: {open_braces} open, {close_braces} close"
            ))

        open_brackets = content.count('[')
        close_brackets = content.count(']')
        if open_brackets != close_brackets:
            errors.append(ValidationError(
                block_id="file",
                error_type="unmatched_bracket",
                message=f"Unmatched brackets: {open_brackets} open, {close_brackets} close"
            ))

        # 3. Check each block for missing variables
        for block_id in block_order:
            if is_separator_block(block_id, parsed_blocks.get(block_id, {})):
                continue

            parsed = parsed_blocks.get(block_id)
            tagged = tagged_blocks.get(block_id)

            if not parsed or not tagged:
                continue

            # Find variables in original and translation
            original_vars = set(re.findall(r'\[(\w+)\]', parsed.get('en', '')))
            trans_vars = set(re.findall(r'\[(\w+)\]', parsed.get('ro', '')))

            # Check if translation is missing variables from original
            missing_vars = original_vars - trans_vars
            if missing_vars and parsed.get('ro', '').strip():  # Only check if translation exists
                errors.append(ValidationError(
                    block_id=block_id,
                    error_type="missing_variable",
                    message=f"Translation missing variables: {', '.join(missing_vars)}"
                ))

        # 4. Check for missing character variables in dialogue blocks
        for block_id in block_order:
            tagged = tagged_blocks.get(block_id)
            if not tagged:
                continue

            if tagged['type'] == 'dialogue' and not tagged.get('char_var'):
                errors.append(ValidationError(
                    block_id=block_id,
                    error_type="missing_char_var",
                    message="Dialogue block missing character variable"
                ))

        return errors

    def get_validation_report(self) -> str:
        """Get a formatted validation error report."""
        if not self.validation_errors:
            return "No validation errors found!"

        report = [f"Found {len(self.validation_errors)} validation errors:\n"]

        # Group errors by type
        errors_by_type: Dict[str, List[ValidationError]] = {}
        for error in self.validation_errors:
            errors_by_type.setdefault(error.error_type, []).append(error)

        for error_type, errors in errors_by_type.items():
            report.append(f"\n{error_type.upper().replace('_', ' ')} ({len(errors)} errors):")
            for error in errors[:5]:  # Show first 5 of each type
                report.append(f"  - {error.block_id}: {error.message}")
            if len(errors) > 5:
                report.append(f"  ... and {len(errors) - 5} more")

        return '\n'.join(report)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def show_banner():
    """Display the merge banner"""
    print()
    print("=" * 70)
    print("               Translation File Merge                      ")
    print("=" * 70)
    print()


def merge_single_file(
    file_path: Path,
    skip_validation: bool = False
):
    """Merge a single .parsed.yaml file"""
    print(f"\nMerging: {file_path.name}")

    # Prepare paths
    base_name = file_path.stem.replace('.parsed', '')
    output_dir = file_path.parent
    tags_yaml_path = output_dir / f"{base_name}.tags.yaml"
    output_rpy_path = output_dir / f"{base_name}.translated.rpy"

    # Check if tags file exists
    if not tags_yaml_path.exists():
        print(f"Error: Tags YAML file not found: {tags_yaml_path}")
        exit(1)

    # Merge
    merger = RenpyMerger()
    success = merger.merge_file(
        parsed_yaml_path=file_path,
        tags_yaml_path=tags_yaml_path,
        output_rpy_path=output_rpy_path,
        validate=not skip_validation
    )

    if not success:
        print('\nValidation found issues. Please review the output file.')
        print(merger.get_validation_report())

    print(f'\nMerge complete!')
    print(f'  Output: {output_rpy_path}')


def find_parsed_yaml_files(tl_path: Path) -> List[Path]:
    """Find source .parsed.yaml files, excluding re-extracted *.translated.parsed.yaml."""
    return [
        f for f in tl_path.rglob("*.parsed.yaml")
        if '.translated.' not in f.name
    ]


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Merge translated YAML files back into .rpy format')
    parser.add_argument('--game-name', type=str, default='', help='Game name from configuration')
    parser.add_argument('--source', type=str, default='', help='Specific .parsed.yaml file to merge')
    parser.add_argument('--all', action='store_true', help='Merge all .parsed.yaml files')
    parser.add_argument('--skip-validation', action='store_true', help='Skip integrity validation')

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
    language_dir = language_info['name'].lower()

    print(f"Game: {game_config['name']}")
    print(f"Language: {language_info['name']}")
    print(f"Path: {game_path}")
    print()

    # Get translation directory
    tl_path = game_path / "game" / "tl" / language_dir

    if not tl_path.exists():
        print(f"Error: Translation directory not found: {tl_path}")
        exit(1)

    # Determine what to merge
    if args.all:
        # Merge all files
        print("Finding all .parsed.yaml files...")
        parsed_files = find_parsed_yaml_files(tl_path)

        if not parsed_files:
            print("No .parsed.yaml files found!")
            print("Please run 2-extract.ps1 first.")
            exit(1)

        print(f"Found {len(parsed_files)} files\n")

        for file_path in parsed_files:
            merge_single_file(file_path, args.skip_validation)
            print()

    elif args.source:
        # Merge specific file
        if args.source.endswith('.parsed.yaml'):
            file_path = tl_path / args.source
        elif args.source.endswith('.yaml'):
            file_path = tl_path / args.source
        else:
            file_path = tl_path / f"{args.source}.parsed.yaml"

        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            exit(1)

        merge_single_file(file_path, args.skip_validation)

    else:
        # Interactive mode
        print("Merge Options:")
        print("  [1] Merge all .parsed.yaml files")
        print("  [2] Merge specific file")
        print()

        choice = input("Select option (1-2): ").strip()

        if choice == "1":
            # Merge all
            print("\nFinding all .parsed.yaml files...")
            parsed_files = find_parsed_yaml_files(tl_path)

            if not parsed_files:
                print("No .parsed.yaml files found!")
                print("Please run 2-extract.ps1 first.")
                exit(1)

            print(f"Found {len(parsed_files)} files\n")

            for file_path in parsed_files:
                merge_single_file(file_path, args.skip_validation)
                print()

        elif choice == "2":
            # List and select file
            print("\nAvailable .parsed.yaml files:")
            print()

            parsed_files = find_parsed_yaml_files(tl_path)

            if not parsed_files:
                print("No .parsed.yaml files found!")
                print("Please run 2-extract.ps1 first.")
                exit(1)

            for i, file_path in enumerate(parsed_files, 1):
                print(f"  [{i}] {file_path.name}")

            print()
            selection = input(f"Select file (1-{len(parsed_files)}): ").strip()

            try:
                index = int(selection) - 1
                if 0 <= index < len(parsed_files):
                    selected_file = parsed_files[index]
                    print()
                    merge_single_file(selected_file, args.skip_validation)
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
    print("Merge complete!")
    print()
    print("Next steps:")
    print("  1. Review the .translated.rpy files")
    print("  2. Test the translations in the game")
    print("  3. Replace the original .rpy files if satisfied")
    print()


if __name__ == "__main__":
    main()
