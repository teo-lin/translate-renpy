"""
Common test utilities shared across test files

Provides reusable functions for:
- Character discovery
- File operations (backup, restore, cleanup)
- Translation counting
- Validation
- Integration test helpers
"""

import cmd
import re
import shutil
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, TypeVar, Type, Callable, Any


# --- Existing Utility Functions (unchanged) ---

def discover_characters(tl_path: Path, game_path: Path) -> Dict[str, dict]:
    """
    Discover characters from .rpy files by scanning for dialogue patterns
    and extracting character names from script.rpy definitions.

    Args:
        tl_path: Path to translation directory (e.g., game/tl/romanian)
        game_path: Path to game root directory

    Returns:
        Dictionary mapping character variables to character info
    """
    character_vars = {}
    character_files = {}  # Track which files each character appears in

    # Step 1: Scan translation .rpy files for dialogue patterns
    rpy_files = list(tl_path.glob("*.rpy"))
    # Exclude backup and generated files
    rpy_files = [f for f in rpy_files if not f.name.endswith('.backup')
                 and not f.name.endswith('.translated.rpy')]

    for rpy_file in rpy_files:
        content = rpy_file.read_text(encoding='utf-8')
        file_name = rpy_file.stem

        # Find dialogue patterns: character_var "text"
        pattern = r'^\s*(\w+)\s+"[^"\\]*(?:\\.[^"\\]*)*"'
        matches = re.finditer(pattern, content, re.MULTILINE)

        for match in matches:
            char_var = match.group(1)

            # Skip keywords
            if char_var in ('translate', 'old', 'new'):
                continue

            # Add character if not seen before
            if char_var not in character_vars:
                character_vars[char_var] = {
                    'name': char_var.upper(),
                    'gender': 'neutral',
                    'type': 'supporting',
                    'description': ''
                }
                character_files[char_var] = []

            # Track this file for the character
            if file_name not in character_files[char_var]:
                character_files[char_var].append(file_name)

    # Step 2: Extract character names from script.rpy define statements
    script_files = list(game_path.rglob("script*.rpy"))
    script_files = [f for f in script_files if 'tl' not in f.parts]

    for script_file in script_files:
        content = script_file.read_text(encoding='utf-8')

        # Match: define var = Character('Name', ...) or Character(None)
        pattern = r'define\s+(\w+)\s*=\s*Character\((?:["\\](.+?)["\\]|None)\s*[,)]'
        matches = re.finditer(pattern, content)

        for match in matches:
            char_var = match.group(1)
            char_name = match.group(2) if match.group(2) else ''

            # Skip if we haven't seen this character in translations
            if char_var not in character_vars:
                continue

            # Handle special cases
            if char_var == 'narrator' or char_name == '':
                character_vars[char_var]['name'] = 'Narrator'
                character_vars[char_var]['type'] = 'narrator'
            # Detect protagonist (common patterns: mc, u, player)
            elif re.match(r'^(mc|u|player)$', char_var) or re.match(r'^\\[.*name.*\\]$', char_name):
                # Use proper name if not a placeholder
                if not re.match(r'^\\[.*\\]$', char_name) and char_name:
                    character_vars[char_var]['name'] = char_name
                else:
                    character_vars[char_var]['name'] = 'MainCharacter'
                character_vars[char_var]['type'] = 'protagonist'
            # Regular characters
            elif not re.match(r'^\?+$|\\[.*\\]$', char_name) and char_name:
                character_vars[char_var]['name'] = char_name
                character_vars[char_var]['type'] = 'main'

    # Step 3: Generate descriptions based on file appearances
    for char_var, files in character_files.items():
        if not files:
            continue

        file_types = []

        # Categorize file types
        cell_files = [f for f in files if f.startswith('Cell')]
        room_files = [f for f in files if f.startswith('Room')]
        exped_files = [f for f in files if f.startswith('Exped')]
        chara_files = [f for f in files if f.startswith('Chara')]

        if cell_files:
            file_types.append('Cell character')
        if room_files:
            file_types.append('Room character')
        if exped_files:
            file_types.append('Expedition character')
        if chara_files:
            file_types.append('Character definition')

        if file_types:
            description = f"{', '.join(file_types)} (appears in {len(files)} files)"
        else:
            description = f"Appears in: {', '.join(files[:3])}"

        character_vars[char_var]['description'] = description

    # Step 4: Add special narrator character (empty string)
    character_vars[''] = {
        'name': 'Narrator',
        'gender': 'neutral',
        'type': 'narrator',
        'description': 'Narration without character'
    }

    return character_vars


def count_translations(file_path: Path) -> int:
    """
    Count how many non-empty translations exist in a .rpy file.

    Args:
        file_path: Path to .rpy file

    Returns:
        Number of non-empty translations found
    """
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    count = 0
    for line in lines:
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith('#'):
            continue

        # Skip "old" lines (source text, not translations)
        if stripped.startswith('old '):
            continue

        # Count character dialogue lines with non-empty translations
        # Format: character_var "translation text"
        if '"' in stripped and '""' not in stripped:
            # Check if it's a character line (starts with identifier) or "new" line
            if stripped.startswith('new ') or (not stripped.startswith('translate') and ' "' in stripped):
                count += 1

    return count


def backup_file(file_path: Path) -> Path:
    """
    Create a backup of a file.

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file
    """
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    shutil.copy2(file_path, backup_path)
    return backup_path


def restore_file(file_path: Path, backup_path: Path):
    """
    Restore a file from its backup.

    Args:
        file_path: Original file path
        backup_path: Backup file path
    """
    if backup_path.exists():
        shutil.copy2(backup_path, file_path)
        backup_path.unlink()


def cleanup_files(files: List[Path]):
    """
    Delete a list of files if they exist.

    Args:
        files: List of file paths to delete
    """
    for file_path in files:
        if file_path.exists():
            file_path.unlink()


def validate_rpy_structure(file_path: Path) -> Dict[str, bool]:
    """
    Validate that a .rpy file has expected structure elements.

    Args:
        file_path: Path to .rpy file to validate

    Returns:
        Dictionary mapping check names to pass/fail status
    """
    content = file_path.read_text(encoding='utf-8')

    checks = {
        'has_translate_statements': 'translate ' in content,
        'has_strings_section': 'translate ' in content and 'strings:' in content,
        'has_color_tags': '{color=' in content or '{/color}' in content,
        'has_variables': '[' in content and ']' in content,
        'has_bold_tags': '{b}' in content or '{/b}' in content,
    }

    return checks


def get_rpy_files(directory: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """
    Get list of .rpy files in a directory, excluding certain patterns.

    Args:
        directory: Directory to search
        exclude_patterns: List of patterns to exclude (e.g., ['.backup', '.translated.rpy'])

    Returns:
        List of .rpy file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ['.backup', '.translated.rpy']

    rpy_files = sorted(directory.glob("*.rpy"))

    # Exclude files matching patterns
    filtered_files = []
    for f in rpy_files:
        should_exclude = False
        for pattern in exclude_patterns:
            if f.name.endswith(pattern):
                should_exclude = True
                break
        if not should_exclude:
            filtered_files.append(f)

    return filtered_files


# --- New Base Test Class ---

_TranslatorType = TypeVar('_TranslatorType')

class BaseTranslatorIntegrationTest(unittest.TestCase):
    """
    Base class for translator integration tests.
    Handles common setup like path configuration and provides a template
    for translator initialization.
    """
    project_root = Path(__file__).parent.parent
    translator: _TranslatorType = None # Type hint for the translator instance
    _test_start_time: float = None  # Track test start time for timing measurements

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment: add project paths to sys.path.
        Derived classes should call super().setUpClass() and then initialize
        their specific translator.
        """
        print(f"\nSetting up test environment for {cls.__name__}...")
        cls._test_start_time = time.time()

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources after all tests in this class have run.
        Derived classes should call super().tearDownClass() and then clean up
        their specific translator if necessary.
        """
        print(f"Tearing down test environment for {cls.__name__}.")
        if cls.translator:
            del cls.translator
            cls.translator = None

        # Free GPU memory if using CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass  # Torch not available or CUDA error, skip cleanup

    def _get_model_path(self, model_subdir: str, model_filename: str) -> Path:
        """
        Helper method to construct model paths.
        """
        return self.project_root / "models" / model_subdir / model_filename

    def _assert_translation(self, english_text: str, expected_romanian: str, **translate_kwargs):
        """
        Helper method for common translation assertion with timing.

        Args:
            english_text: Text to translate
            expected_romanian: Expected translation
            **translate_kwargs: Additional arguments to pass to translate() (e.g., temperature=0)
        """
        print(f"Translating: '{english_text}'")

        # Time the translation
        start_time = time.time()
        translation = self.translator.translate(english_text, **translate_kwargs)
        elapsed_time = time.time() - start_time

        # Handle Unicode characters on Windows console (cp1252 encoding)
        try:
            print(f"Received translation: '{translation}' (took {elapsed_time:.3f}s)")
        except UnicodeEncodeError:
            # Use ASCII representation for characters that can't be printed
            print(f"Received translation: {ascii(translation)} (took {elapsed_time:.3f}s)")

        self.assertEqual(translation, expected_romanian)


# --- translate_batch() Test Mixin ---

class TranslateBatchTestMixin:
    """
    Test mixin for translators that implement translate_batch(texts) -> list.

    Provides three checks usable by any integration test class that exposes
    `self.translator`:
      - empty input returns empty list
      - N inputs return a list of N non-empty strings
      - single-item batch matches translate(text) output

    Override `BATCH_TEXTS` / `SINGLE_TEXT` to use different samples, or override
    `_assert_batched_matches_single()` to relax the equality check for models
    where beam-search padding causes minor divergence (e.g. MADLAD).
    """
    BATCH_TEXTS = ["Hello!", "Good morning.", "Thank you."]
    SINGLE_TEXT = "Hello!"

    def test_translate_batch_empty_input(self):
        self.assertEqual(self.translator.translate_batch([]), [])

    def test_translate_batch_returns_list(self):
        results = self.translator.translate_batch(list(self.BATCH_TEXTS))
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.BATCH_TEXTS))
        for r in results:
            self.assertIsInstance(r, str)
            self.assertGreater(len(r.strip()), 0)

    def test_translate_batch_single_matches_translate(self):
        single = self.translator.translate(self.SINGLE_TEXT)
        batched = self.translator.translate_batch([self.SINGLE_TEXT])
        self.assertEqual(len(batched), 1)
        self._assert_batched_matches_single(batched[0], single)

    def _assert_batched_matches_single(self, batched, single):
        """Default: strict equality. Override for translators where beam search
        diverges between padded batch and unpadded single inputs."""
        self.assertEqual(batched, single)


# --- Integration Test Helper Functions ---

def get_test_device() -> str:
    """
    Auto-detect best available device for testing, using the same probe as the
    translators (verifies CUDA kernels actually work, not just that CUDA is present).
    """
    try:
        from translators.translator_utils import probe_device
        return probe_device()
    except Exception:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def skip_if_transformers_unavailable(transformers_available: bool, import_error: str,
                                     translator_name: str) -> None:
    """
    Skip test if transformers is not available.

    Args:
        transformers_available: Boolean flag indicating if transformers imported successfully
        import_error: Error message from failed import
        translator_name: Name of the translator for error message

    Raises:
        unittest.SkipTest: If transformers is not available
    """
    if not transformers_available:
        raise unittest.SkipTest(
            f"{translator_name} requires transformers and torch packages "
            f"due to: {import_error}"
        )


import io

class OutputCapturer(object):
    """Context manager to capture stdout and stderr."""
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.stdout = io.StringIO()
        sys.stderr = self.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def safe_init_translator(translator_class: Type, translator_name: str,
                        init_kwargs: dict) -> Any:
    """
    Safely initialize a translator. If an exception occurs, it will
    print captured stdout/stderr and re-raise the exception, causing
    the test to fail instead of skip.
    """
    print(f"Setting up {translator_name} for integration test...")
    with OutputCapturer() as capturer:
        try:
            translator = translator_class(**init_kwargs)
            print("Translator setup complete.")
            return translator
        except Exception as e:
            # Print captured output before re-raising the exception
            captured_stdout = capturer.stdout.getvalue()
            captured_stderr = capturer.stderr.getvalue()
            if captured_stdout:
                sys.stdout.write("\n--- Captured STDOUT during translator init ---\n")
                sys.stdout.write(captured_stdout)
                sys.stdout.write("----------------------------------------------\n")
            if captured_stderr:
                sys.stderr.write("\n--- Captured STDERR during translator init ---\n")
                sys.stderr.write(captured_stderr)
                sys.stderr.write("----------------------------------------------\n")
            raise e


# --- Test Runner for Automated Testing ---

import subprocess
import yaml
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Dict, Set


@dataclass
class TestResult:
    """Represents the result of a single test execution."""
    name: str
    passed: bool
    duration: timedelta
    exit_code: int
    device: Optional[str] = None
    n_passed: int = 0
    n_skipped: int = 0

    @property
    def all_skipped(self) -> bool:
        return self.passed and self.n_skipped > 0 and self.n_passed == 0


class TestRunner:
    """
    Automated test runner for the translation system.
    Handles test discovery, filtering, execution, and reporting.
    """

    def __init__(self, project_root: Path):
        """
        Initialize test runner.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.python_exe = project_root / "venv" / "Scripts" / "python.exe"
        self.results: List[TestResult] = []

        # Model-specific E2E tests mapping
        self.model_specific_tests = {
            "test_e2e_aya23.py": "aya23",
            "test_e2e_madlad400.py": "madlad400",
            "test_e2e_mbartRo.py": "mbartRo",
            "test_e2e_helsinkyRo.py": "helsinkiRo",
            "test_e2e_seamlessm96.py": "seamlessm96",
        }

        # Deprecated tests (incompatible with current architecture)
        self.deprecated_tests: Set[str] = set()

        # Tests that need model script arguments
        self.tests_needing_model_script = {
            "test_e2e_translate_aio.py",
            "test_e2e_translate_aio_uncensored.py"
        }

    def setup_environment(self):
        """Set up environment variables for test execution."""
        import os

        # Models live in the default HuggingFace cache (~/.cache/huggingface),
        # shared across projects — do not redirect HF_HOME into the project.

        # Add PyTorch lib directory to PATH for CUDA DLLs
        torch_lib_path = self.project_root / "venv" / "Lib" / "site-packages" / "torch" / "lib"
        if torch_lib_path.exists():
            os.environ['PATH'] = f"{torch_lib_path};{os.environ.get('PATH', '')}"

        # Set UTF-8 encoding for Python
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    def check_python_exe(self) -> bool:
        """
        Check if venv Python executable exists.

        Returns:
            True if Python exists, False otherwise
        """
        return self.python_exe.exists()

    def detect_test_device(self) -> str:
        """
        Detect available device for integration/e2e tests.

        Returns:
            'cuda' if GPU available, otherwise 'cpu'
        """
        try:
            script = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu', end='')"
            result = subprocess.run(
                [str(self.python_exe), "-c", script],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "cpu"

    def load_configurations(self) -> tuple:
        """
        Load models and current configurations from YAML files.

        Returns:
            Tuple of (models_config, current_config)

        Raises:
            FileNotFoundError: If configuration files don't exist
            ValueError: If required configuration values are missing
        """
        models_config_file = self.project_root / "models" / "models_config.yaml"
        current_config_file = self.project_root / "models" / "current_config.yaml"

        if not models_config_file.exists():
            raise FileNotFoundError(
                f"Models configuration not found at: {models_config_file}. "
                "Please run 0-setup.ps1."
            )

        if not current_config_file.exists():
            raise FileNotFoundError(
                f"Current configuration not found at: {current_config_file}. "
                "Please run 1-config.ps1."
            )

        with open(models_config_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        with open(current_config_file, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)

        return models_config, current_config

    def get_installed_models(self, models_config: dict) -> List[dict]:
        """
        Get list of installed models.

        Args:
            models_config: Models configuration dictionary

        Returns:
            List of installed model dictionaries
        """
        installed_models = []

        sys.path.insert(0, str(self.project_root / "src"))
        from hardware import is_model_available

        for model_key, model_config in models_config.get('available_models', {}).items():
            if is_model_available(model_key):
                installed_models.append({
                    'key': model_key,
                    'name': model_config['name'],
                    'description': model_config.get('description', ''),
                    'size': model_config.get('size', ''),
                    'config': model_config
                })

        return installed_models

    def select_model(self, target_model_name: str, installed_models: List[dict]) -> Optional[dict]:
        """
        Select model based on target model name.

        Args:
            target_model_name: Name or key of target model
            installed_models: List of installed models

        Returns:
            Selected model dictionary or None if not found
        """
        # Try exact name match
        for model in installed_models:
            if model['name'] == target_model_name:
                return model

        # Try key match
        for model in installed_models:
            if model['key'] == target_model_name:
                return model

        # Try case-insensitive name match
        for model in installed_models:
            if model['name'].lower() == target_model_name.lower():
                return model

        # Try fuzzy match
        normalized_target = ''.join(c for c in target_model_name if c.isalnum()).lower()
        for model in installed_models:
            normalized_key = ''.join(c for c in model['key'] if c.isalnum()).lower()
            normalized_name = ''.join(c for c in model['name'] if c.isalnum()).lower()

            if (normalized_target.startswith(normalized_key) or
                normalized_target in normalized_key or
                normalized_key.startswith(normalized_target) or
                normalized_key in normalized_target):
                return model

        return None

    def discover_tests(self, category: Optional[str] = None) -> List[Path]:
        """
        Discover test files based on category.

        Args:
            category: Test category ('unit', 'int', 'e2e', or None for all)

        Returns:
            List of test file paths
        """
        all_test_files = sorted(self.test_dir.glob("test_*.py"))

        if category == 'unit':
            return [f for f in all_test_files if f.name.startswith("test_unit_")]
        elif category == 'int':
            return [f for f in all_test_files if f.name.startswith("test_int_")]
        elif category == 'e2e':
            return [f for f in all_test_files if f.name.startswith("test_e2e_")]
        else:
            return all_test_files

    def filter_tests(self, test_files: List[Path], installed_models: List[dict]) -> List[Path]:
        """
        Filter out deprecated tests and model-specific tests for uninstalled models.

        Args:
            test_files: List of test file paths
            installed_models: List of installed models

        Returns:
            Filtered list of test file paths
        """
        filtered = []
        installed_keys = {m['key'] for m in installed_models}

        for test_file in test_files:
            # Skip deprecated tests
            if test_file.name in self.deprecated_tests:
                print(f"Skipping deprecated test {test_file.name} - incompatible with current architecture")
                continue

            # Check model-specific tests
            if test_file.name in self.model_specific_tests:
                required_model_key = self.model_specific_tests[test_file.name]
                if required_model_key in installed_keys:
                    filtered.append(test_file)
                else:
                    print(f"Skipping test {test_file.name} - associated model '{required_model_key}' is not installed.")
            else:
                filtered.append(test_file)

        return filtered

    def run_test(self, test_file: Path, selected_model: Optional[dict],
                 language_code: str, test_device: str) -> TestResult:
        """
        Run a single test file.

        Args:
            test_file: Path to test file
            selected_model: Selected model dictionary (if applicable)
            language_code: Target language code
            test_device: Device to use for testing ('cuda' or 'cpu')

        Returns:
            TestResult object
        """
        import time

        start_time = time.time()

        # Build command — run via pytest so @unittest.skipIf shows as SKIP not PASS
        cmd = [str(self.python_exe), "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header"]

        # Add arguments for tests that need them
        if test_file.name in self.tests_needing_model_script and selected_model:
            script_path = self.project_root / selected_model['config']['script'].replace('/', '\\')
            cmd.extend([
                "--model_script", str(script_path),
                "--language", language_code,
                "--model_key", selected_model['key']
            ])

        # Run test, capture output so we can parse skip/pass counts.
        # Force UTF-8 stdio in the child so prints of Romanian text (ă, ș, ț, ...)
        # don't crash on Windows' cp1252 console.
        import os
        import re as _re
        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        n_passed = 0
        n_skipped = 0
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env)
            # Print output so user sees test details
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
            exit_code = result.returncode
            # Parse pytest summary line, e.g. "2 passed, 3 skipped in 0.05s"
            combined = result.stdout + result.stderr
            if m := _re.search(r'(\d+) passed', combined):
                n_passed = int(m.group(1))
            if m := _re.search(r'(\d+) skipped', combined):
                n_skipped = int(m.group(1))
        except Exception as e:
            print(f"Error running test: {e}")
            exit_code = 1

        duration = timedelta(seconds=time.time() - start_time)

        # Determine if this test needs device info
        needs_device = test_file.name.startswith("test_int_") or test_file.name.startswith("test_e2e_")

        return TestResult(
            name=test_file.name,
            passed=(exit_code == 0),
            duration=duration,
            exit_code=exit_code,
            device=test_device if needs_device else None,
            n_passed=n_passed,
            n_skipped=n_skipped,
        )

    def print_summary(self):
        """Print test execution summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed_count = sum(1 for r in self.results if r.passed and not r.all_skipped)
        skipped_count = sum(1 for r in self.results if r.all_skipped)
        failed_count = sum(1 for r in self.results if not r.passed)
        total_duration = sum((r.duration for r in self.results), timedelta())

        # Print individual results
        print()
        max_name_length = max(len(r.name) for r in self.results) if self.results else 0

        for result in self.results:
            if not result.passed:
                status, color_code = "FAIL", "\033[91m"
            elif result.all_skipped:
                status, color_code = "SKIP", "\033[93m"
            else:
                status, color_code = "PASS", "\033[92m"
            reset_code = "\033[0m"

            # Format duration
            total_seconds = result.duration.total_seconds()
            if total_seconds < 60:
                duration_str = f"{total_seconds:6.2f}s"
            else:
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                duration_str = f"{minutes}m {seconds:.0f}s"

            # Pad name for alignment
            padded_name = result.name.ljust(max_name_length)

            # Print result line
            print(f"  [{color_code}{status}{reset_code}] {padded_name} took \033[96m{duration_str}\033[0m", end="")

            # Show device for integration/e2e tests
            if result.device:
                device_color = "\033[93m" if result.device == "cuda" else "\033[90m"  # Yellow or Gray
                print(f" on {device_color}{result.device.upper()}{reset_code}")
            else:
                print()

        # Print summary stats
        print()
        print("=" * 70)

        # Format total duration
        total_seconds = total_duration.total_seconds()
        if total_seconds < 60:
            total_duration_str = f"{total_seconds:.2f}s"
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            total_duration_str = f"{minutes}m {seconds:.0f}s"
        else:
            hours = int(total_seconds // 3600)
            remaining = total_seconds % 3600
            minutes = int(remaining // 60)
            seconds = remaining % 60
            total_duration_str = f"{hours}h {minutes}m {seconds:.0f}s"

        print(f"Total: {len(self.results)} tests | ", end="")
        print(f"\033[92mPassed: {passed_count}\033[0m | ", end="")
        print(f"\033[93mSkipped: {skipped_count}\033[0m | ", end="")

        if failed_count > 0:
            print(f"\033[91mFailed: {failed_count}\033[0m | ", end="")
        else:
            print(f"Failed: {failed_count} | ", end="")

        print(f"Total Time: \033[96m{total_duration_str}\033[0m")
        print("=" * 70)

        return failed_count

    def run(self, category: Optional[str] = None) -> int:
        """
        Run all tests.

        Args:
            category: Test category to run ('unit', 'int', 'e2e', or None for all)

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("\n" + "=" * 70)
        print("RUNNING ALL STANDALONE TESTS")
        print("=" * 70)

        # Setup environment
        self.setup_environment()

        # Check Python
        if not self.check_python_exe():
            print(f"\033[91mVirtual environment not found at: {self.python_exe}\033[0m")
            print("\033[91mPlease run setup first or ensure venv is created.\033[0m")
            return 1

        # Detect device
        test_device = self.detect_test_device()

        # Load configurations
        try:
            models_config, current_config = self.load_configurations()
        except (FileNotFoundError, ValueError) as e:
            print(f"\033[91m{e}\033[0m")
            return 1

        # Get current game configuration
        current_game_name = current_config.get('current_game')
        if not current_game_name:
            print("\033[91mNo 'current_game' set in current_config.yaml. Please run 1-config.ps1.\033[0m")
            return 1

        current_game_config = current_config.get('games', {}).get(current_game_name)
        if not current_game_config:
            print(f"\033[91mConfiguration for current game '{current_game_name}' not found.\033[0m")
            return 1

        target_language_code = current_game_config.get('target_language')
        target_model_name = current_game_config.get('model')

        if not target_language_code or not target_model_name:
            print("\033[91mNo 'target_language' or 'model' set for current game. Please run 1-config.ps1.\033[0m")
            return 1

        # Get installed models
        installed_models = self.get_installed_models(models_config)
        if not installed_models:
            print("\033[91mERROR: No downloaded models found. Please run 0-setup.ps1.\033[0m")
            return 1

        # Select model
        selected_model = self.select_model(target_model_name, installed_models)
        if not selected_model:
            print(f"\033[91mThe configured model '{target_model_name}' is not installed.\033[0m")
            print("\033[96mAvailable installed models:\033[0m")
            for model in installed_models:
                print(f"  - Key: {model['key']}, Name: {model['name']}")
            return 1

        print(f"\033[96mUsing configured model: {selected_model['name']}\033[0m")
        print(f"\033[96mTest Language: {target_language_code}\033[0m")

        if test_device == "cuda":
            print("\033[96mTest Device: CUDA (GPU acceleration enabled)\033[0m")
        else:
            print("\033[96mTest Device: CPU\033[0m")

        # Discover tests
        test_files = self.discover_tests(category)
        if not test_files:
            print(f"\033[91mNo test files found matching category: {category or 'all'}\033[0m")
            return 1

        category_name = {
            'unit': 'Unit Tests',
            'int': 'Integration Tests',
            'e2e': 'End-to-End Tests',
            None: 'All Tests'
        }.get(category, 'All Tests')

        print()
        print(f"\033[96mRunning: {category_name}\033[0m")
        print(f"\033[96mFound {len(test_files)} test file(s):\033[0m")
        print()
        for file in test_files:
            print(f"  - {file.name}")

        # Filter tests
        test_files = self.filter_tests(test_files, installed_models)

        # Run tests
        for i, test_file in enumerate(test_files, 1):
            print("\n" + "=" * 70)
            print(f"Running test {i} of {len(test_files)}: {test_file.name}")
            print("=" * 70)

            result = self.run_test(test_file, selected_model, target_language_code, test_device)
            self.results.append(result)

            # Print immediate result
            print()
            if result.passed:
                print(f"\033[92mPASSED: {test_file.name} (took {result.duration.total_seconds():.2f}s)\033[0m")
            else:
                print(f"\033[91mFAILED: {test_file.name} (exit code: {result.exit_code}, "
                      f"took {result.duration.total_seconds():.2f}s)\033[0m")

        # Print summary
        failed_count = self.print_summary()

        # Final message
        print()
        if failed_count > 0:
            print("\033[91mSome tests failed!\033[0m")
            return 1
        elif all(r.all_skipped for r in self.results):
            print("\033[93mAll tests skipped (models not installed).\033[0m")
            return 0
        else:
            print("\033[92mAll tests passed!\033[0m")
            return 0