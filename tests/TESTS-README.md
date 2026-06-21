# Test Suite

> **Repo:** `translate-renpy` (full Ren'Py kit). The end-to-end example test uses
> the bundled `games/Example/` demo game. Everything below the "Repo-specific
> notes" section is shared verbatim with the `translate-local` test suite
> (`tests/README.md` there) - keep the two in sync.

## Repo-specific notes (translate-renpy)

- `test_e2e_example.py` runs against `games/Example/game/tl/romanian/` (bundled).
- `test_unit_renpy_tags.py` covers the Ren'Py tag extractor/restorer used by
  `extract.py` / `merge.py`.
- Integration (`test_int_*`) tests download and load real models; they require the
  corresponding model files in the HuggingFace cache and a working CUDA/CPU setup.

---

## Test Map

Layers: **unit** (fast, mocked, no models) | **int** (loads a real model) |
**e2e** (full pipeline / scoring).

| Test                               | Layer | Covers                                  |
| ---------------------------------- | ----- | --------------------------------------- |
| `test_unit_config.py`              | unit  | game configuration                      |
| `test_unit_config_selector.py`     | unit  | interactive selection helpers           |
| `test_unit_setup.py`               | unit  | `ProjectSetup` flow (mocked)            |
| `test_unit_hardware.py`            | unit  | compute-tier detection                  |
| `test_unit_extract.py`             | unit  | `.rpy` -> `.parsed.yaml`/`.tags.yaml`   |
| `test_unit_merge.py`               | unit  | YAML -> `.rpy` merge + validation       |
| `test_unit_renpy_tags.py`          | unit  | tag strip/restore                       |
| `test_unit_correct.py`             | unit  | grammar/pattern correction              |
| `test_unit_translate.py`           | unit  | translate dispatch + batch loop (mock)  |
| `test_unit_translate_batch.py`     | unit  | HF `translate_batch()` parity           |
| `test_unit_translate_new.py`       | unit  | batch translator behavior               |
| `test_unit_benchmark.py`           | unit  | scoring helpers                         |
| `test_unit_compare.py`             | unit  | compare helpers                         |
| `test_unit_llama_cpp_translator.py`| unit  | GGUF/llama.cpp translator               |
| `test_int_aya23.py`                | int   | Aya-23-8B                               |
| `test_int_ayaExpanse8b.py`         | int   | Aya Expanse 8B                          |
| `test_int_llama_cpp.py`            | int   | llama.cpp backend (EuroLLM etc.)        |
| `test_int_madlad400.py`            | int   | MADLAD-400-3B                           |
| `test_int_mbartRo.py`              | int   | MBART-En-Ro                             |
| `test_int_nllb200.py`              | int   | NLLB-200-600M                           |
| `test_int_nllb1300.py`             | int   | NLLB-200-1.3B                           |
| `test_int_opusTCBig.py`            | int   | OPUS-MT-TC-Big                          |
| `test_int_helsinkyRo.py`           | int   | Helsinki OPUS-MT                        |
| `test_int_seamless96.py`           | int   | SeamlessM4T-v2                          |
| `test_e2e_example.py`              | e2e   | full pipeline on the example game       |
| `test_e2e_compare.py`              | e2e   | translate + score                       |
| `test_e2e_benchmark.py`            | e2e   | score existing translations             |

Shared helpers: `conftest.py` (fixtures, auto-markers), `utils.py` (common test
utilities), `__init__.py`.

---

## Running Tests

```powershell
# Interactive runner (prompts for model where needed)
.\7-test.ps1

# All tests via pytest (installed via requirements.txt)
.\venv\Scripts\pytest.exe tests/

# By layer (markers are auto-applied by conftest.py)
.\venv\Scripts\pytest.exe -m unit
.\venv\Scripts\pytest.exe -m "int or e2e"

# A single file, verbose
.\venv\Scripts\pytest.exe tests/test_unit_setup.py -v

# By name pattern
.\venv\Scripts\pytest.exe -k "setup or config_selector"

# In parallel (requires pytest-xdist)
.\venv\Scripts\pytest.exe -n auto

# With coverage (requires pytest-cov)
.\venv\Scripts\pytest.exe --cov=src --cov=scripts --cov-report=term

# Standalone (each test file also has an __main__ block)
.\venv\Scripts\python.exe .\tests\test_unit_config.py
```

## Conventions

- **ASCII only** in test code (no emojis / Unicode arrows) - the Windows console
  uses cp1252; non-ASCII in output raises `UnicodeEncodeError`. See `AGENTS.md`.
- **Parsed YAML holds clean text only**; tags live in the separate `.tags.yaml`
  and are restored on merge.
- Integration/e2e tests **skip with a message** when the required model or game
  files are unavailable, rather than hard-failing.
- Imports use `from utils import ...` (not `from tests.utils import ...`).
