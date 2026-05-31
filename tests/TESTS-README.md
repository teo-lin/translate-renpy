# Test Suite

| Test                                     | Config | Extract   | Translate   | Correct | Merge     | Pipeline      |
| ---------------------------------------- | ------ | --------- | ----------- | ------- | --------- | ------------- |
| `test_e2e_aya23.py`                    | ✅     | ✅        | ✅ Aya-23   |         | ✅        | ✅ Mod        |
| `test_e2e_madlad.py`                   | ✅     | ✅        | ✅ Madlad   |         | ✅        | ✅ Mod        |
| `test_e2e_seamlessm4t.py`              | ✅     | ✅        | ✅ Seamless |         | ✅        | ✅ Mod        |
| `test_e2e_mbart.py`                    | ✅     | ✅        | ✅ MBART    |         | ✅        | ✅ Mod        |
| `test_e2e_quickmt.py`                  | ✅     | ✅        | ✅ QuickMT  |         | ✅        | ✅ Mod        |
| `test_e2e_example_game.py`             | ✅     | ✅        | ✅ Any      |         | ✅        | ⚠️ Mod (PS) |
| `test_e2e_translate_aio.py`            |        |           | ✅ Any      |         |           | ⚠️ AIO      |
| `test_e2e_translate_aio_uncensored.py` |        |           | ✅ Any      |         |           | ⚠️ AIO      |
| `test_unit_extract_merge.py`           |        | ✅        |             |         | ✅        | ✅ Mod        |
| `test_unit_renpy_tags.py`              |        | ⚠️ Unit |             |         | ⚠️ Unit | ✅ Mod        |
| `test_unit_config.py`                  | ✅     |           |             |         |           | ⚠️ Mod (PS) |
| `test_unit_setup.py`                   | ✅     |           |             |         |           | ✅ Python     |
| `test_unit_config_selector.py`         | ✅     |           |             |         |           | ✅ Python     |
| `test_unit_extract.py`                 |        | ✅        |             |         |           | ✅ Mod        |
| `test_unit_translate.py`               |        |           | ⚠️ Mock   |         |           | ✅ Mod        |
| `test_unit_correct.py`                 |        |           |             | ✅      |           | ✅ Mod        |
| `test_unit_merge.py`                   |        |           |             |         | ✅        | ✅ Mod        |

---

## Running & Debugging Tests

### Test Execution Methods

**Two approaches available:**

1. **Standalone Python** (Simple, no dependencies)

   - Run test files directly with Python
   - Matches existing test pattern
   - Works immediately without additional setup
2. **pytest** (Industry standard, already installed in requirements.txt)

   - Modern Python testing framework (~90% of projects use it)
   - Auto-discovers all tests
   - Better failure output, fixtures, parallel execution
   - More powerful features (coverage, markers, parametrization)

**Recommendation**: Use pytest for new development (it's already installed), but standalone execution still works fine.

---

### Run All Tests

```powershell
# Run all tests with model selection prompt
.\2-test.ps1

# Run all tests with specific model (e.g., Aya-23 = model 2)
.\2-test.ps1 -Model 2
```

### Run Individual Tests (Standalone Python)

```powershell
# Run unit tests directly (no model needed)
.\venv\Scripts\python.exe .\tests\test_unit_config.py
.\venv\Scripts\python.exe .\tests\test_unit_setup.py
.\venv\Scripts\python.exe .\tests\test_unit_config_selector.py
.\venv\Scripts\python.exe .\tests\test_unit_extract.py
.\venv\Scripts\python.exe .\tests\test_unit_merge.py
.\venv\Scripts\python.exe .\tests\test_unit_correct.py
.\venv\Scripts\python.exe .\tests\test_unit_renpy_tags.py

# Run modular translation test (requires model, uses mock)
.\venv\Scripts\python.exe .\tests\test_unit_translate.py

# E2E tests (need model files downloaded)
.\venv\Scripts\python.exe .\tests\test_e2e_aya23.py
```

### Run Tests with pytest (Recommended)

```powershell
# Run all tests
.\venv\Scripts\pytest.exe tests/

# Run all unit tests only
.\venv\Scripts\pytest.exe -m unit

# Run specific test file
.\venv\Scripts\pytest.exe tests/test_unit_setup.py -v

# Run tests matching pattern
.\venv\Scripts\pytest.exe -k "setup or config_selector"

# Run with verbose output
.\venv\Scripts\pytest.exe -v

# Run in parallel (faster, requires pytest-xdist)
.\venv\Scripts\pytest.exe -n auto

# Run with coverage report (requires pytest-cov)
.\venv\Scripts\pytest.exe --cov=src --cov=scripts --cov-report=html --cov-report=term
```

**Note**: pytest is already installed via `requirements.txt`. Both execution methods work - standalone tests have `if __name__ == "__main__"` blocks for direct execution, but also work with pytest's test discovery.

### Common Issues

**Import errors with triton/torch/torchao** ✅ **FIXED**:

- **Problem**: E2E tests failed with `ImportError: cannot import name 'AttrsDescriptor'` due to package incompatibility
- **Solution**: Implemented lazy loading in `src/translators/__init__.py` and protected imports in translator modules
- **Result**: Tests now fail gracefully with helpful error messages instead of crashing
- **Remaining**: Tests using transformers (MADLAD, SeamlessM4T) still require compatible package versions for actual use
- See: https://github.com/pytorch/ao/issues/2919 and `tests/FIX_SUMMARY.md`

**Unicode encoding errors** ✅ **FIXED**: All emojis removed from `.py` files (kept only in `.md`).

**Path issues** ✅ **FIXED**: Tests use `from utils import` (not `from tests.utils import`).

**YAML/JSON test data**: Parsed YAML should contain **clean text without tags** (tags are stripped during extraction). Tags are stored separately in the JSON file and restored during merge.

**All translator implementations complete**: All working models (MBART, QuickMT, MADLAD, SeamlessM4T, Aya-23) have been implemented and tested.

**LLMic removed**: The LLMic-3B model was removed due to non-functional translation. See `models/MODELS.md` for details.

  These tests will skip with a helpful message if the translator module is not available.

---

## Shared Utilities

**`utils.py`** - Common functions used across tests:

- `discover_characters()` - Auto-discover characters from .rpy
- `count_translations()` - Count translations in .rpy
- `backup_file()`, `restore_file()`, `cleanup_files()` - File operations
- `validate_rpy_structure()` - Validate .rpy format
- `get_rpy_files()` - Get .rpy files in directory

**`conftest.py`** - Pytest configuration and fixtures:

- `project_root` - Fixture providing project root path
- `mock_models_config` - Fixture with mock model configuration
- `mock_tools_config` - Fixture with mock tools configuration
- Auto-markers for unit/integration/e2e tests

---

## New Python Migration Tests

### `test_unit_setup.py`

Tests for the new `src/setup.py` Python implementation:

- ✅ Configuration loading and language list building
- ✅ Language selection (all, specific, auto-selection)
- ✅ Model selection with language filtering
- ✅ Config save/load from YAML
- ✅ Python package checking (torch, llama-cpp-python, transformers)
- ✅ CUDA availability checking
- ✅ Installation verification
- ✅ Footer output (success/warnings)

**15 tests** covering all major `ProjectSetup` class methods with mocking.

### `test_unit_config_selector.py`

Tests for the new `scripts/config_selector.py` utility functions:

- ✅ Single item selection with auto-selection
- ✅ Multiple item selection
- ✅ Language selection with single-row display
- ✅ User input validation and error handling
- ✅ Quit functionality
- ✅ Duplicate selection handling
- ✅ Empty list error handling

**12 tests** covering all three selection functions (`select_item`, `select_multiple_items`, `select_languages_single_row`).


=====================================================================
=====================================================================

TEST SUMMARY
============

  [PASS] test_e2e_benchmark.py             took   0.50s on CUDA
  [PASS] test_e2e_compare.py                 took 3m 7s on CUDA
  [PASS] test_e2e_example.py                   took  19.77s on CUDA
  [PASS] test_int_aya23.py                 took  12.15s on CUDA
  [PASS] test_int_ayaExpanse8b.py          took   9.33s on CUDA
  [PASS] test_int_helsinkyRo.py            took   7.10s on CUDA
  [PASS] test_int_llama_cpp.py             took 1m 4s on CUDA
  [PASS] test_int_madlad400.py             took  33.73s on CUDA
  [PASS] test_int_mbartRo.py               took  20.68s on CUDA
  [PASS] test_int_nllb1300.py              took  38.04s on CUDA
  [PASS] test_int_nllb200.py               took  34.45s on CUDA
  [PASS] test_int_opusTCBig.py             took  10.52s on CUDA
  [PASS] test_int_seamless96.py            took  46.33s on CUDA
  [PASS] test_unit_compare.py              took   4.77s
  [PASS] test_unit_config.py               took   0.48s
  [PASS] test_unit_config_selector.py      took   0.43s
  [PASS] test_unit_correct.py              took   0.43s
  [PASS] test_unit_extract.py              took   0.66s
  [PASS] test_unit_hardware.py             took   0.58s
  [PASS] test_unit_llama_cpp_translator.py took   0.46s
  [PASS] test_unit_merge.py                took   0.45s
  [PASS] test_unit_renpy_tags.py           took   0.42s
  [PASS] test_unit_setup.py                took   0.49s
  [PASS] test_unit_translate.py            took   0.57s
  [PASS] test_unit_translate_batch.py      took   2.46s
  [PASS] test_unit_translate_new.py        took   0.68s

================
Total: 28 tests | Passed: 28 | Skipped: 0 | Failed: 0 | Total Time: 24m 22s
===========================================================================
