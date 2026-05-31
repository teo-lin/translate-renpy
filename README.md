# Automated Batch Ren'Py Translation

## Goal

Local LLM transtation from English to Romanian of visual ren'py novels, fully context-aware, gramattically correct, following the same tone and style as the original, no censorship/filtering with the best available models (12 tested, 10+ available)

## Features

Translate **any Ren'Py visual novel** into **400+ languages** using state-of-the-art local AI models on consumer grade laptop GPU (tested with RTX 3060, Windows)

- **Automated model download** - interactive setup script
- **Automated virtual environment and dependencies setup** - interactive setup script, run once.
- **Multi Model Support** - Choose between the most fit LLMs available in huggingface for this task
- **Modular Pipeline** - Extract → Translate → Merge workflow for better control and performance
- **Local Translation** - No cloud services, complete privacy
- **Preserves Ren'Py Formatting** - Keeps `{color=...}`, `{size=...}`, `[variables]` intact
- **Glossary Support** - Consistent terminology across your translations
- **Grammar Correction** - Optional post-processing for improved quality (Aya-23-8B)
- **Quality Benchmarking** - BLEU score testing against reference translations
- **Batch Processing** - Translate entire games automatically
- **Full GPU Acceleration** - Fast translation with CUDA support
- **Human Review Workflow** - Edit translations in YAML format before merging
- **Git-Friendly** - Track translation changes with clean diffs
- **Low spec requirements** - NVIDIA GPU with 6GB+ VRAM (CUDA 12.4) - tested with RTX3060 on Windows. Uses all available GPU layers.

## Prerequisites

- **NVIDIA GPU** with 6 GB+ VRAM (CUDA 12.4) - tested on RTX 3060
- **Windows Developer Mode** - must be enabled for HuggingFace hub symlinks
  - Without it, model files are duplicated on disk, using ~2x storage
  - Enable: `Settings > System > For developers > toggle ON`

## How to use (Setup & Translation Workflows)

### 0. **Automated Setup**

First, run setup (once). Check models/README-MODELS.md for setup details and a comparison of selectable LLMs.

```powershell
.\0-setup.ps1 # select desired model(s), language(s) at the prompt
```

**Model storage:** Models download to the default HuggingFace cache
(`%USERPROFILE%\.cache\huggingface\hub\`), not into the project. This way they
are shared across all projects on your machine and downloaded only once. The
project's `models/` folder keeps only small YAML config files.

This app supports **three translation workflows**:

### 1. **All-in-One Workflow**

Simple, automated translation in a single step. Best for quick translations.

```powershell
.\5-3-translate.ps1  # Processes .rpy files directly
```

### 2. **Modular Workflow** (Recommended for Control)

Three-phase pipeline with human review checkpoints. See Pipelines section below for more info..

```powershell
# interactive scripts - allow you to select desired game, target language(s), model(s)
.\1-config.ps1  # game setup (one-time)
.\2-extract.ps1  # extract clean text and tags to separate files, token-efficient
.\3-translate.ps1 # batched
.\4-correct.ps1 # optional. human readable format
.\5-merge.ps1 # merge translation and tags back to .rpy
```

---

### 3. **Manual Mode** (Advanced)

You can also call the Python scripts directly with specific arguments:

**Example: Aya-23-8B (23 Languages, Higher Quality):**

```powershell
python scripts\translate_with_aya23.py "Example\game\tl\<language>" --language <Language> # Translate entire game
python scripts\translate_with_aya23.py "path\to\file.rpy" # Translate single file
python scripts\correct.py "path\to\game\game\tl\<language>"

# Fast pattern corrections only
.\4-correct.ps1 "path\to\game\game\tl\<language>" --patterns-only

# Full correction (patterns + LLM)
.\4-correct.ps1 "path\to\game\game\tl\<language>"

# Preview changes without modifying files
.\4-correct.ps1 "path\to\game\game\tl\<language>" --dry-run
```

### 4. Optional: Benchmark Script: `scripts\benchmark.py`

Benchmarks translation quality using BLEU scores by comparing model outputs to reference translations.

**Features:**

- Calculates BLEU scores for translation quality assessment
- Auto-detects language from filename
- Auto-detects matching glossary
- Shows average, min, max scores and best/worst examples

**Benchmark Data Format:**
Create `data/<language>_benchmark.json` with reference translations:

```json
[
  {
    "source": "English text to translate",
    "target": "Reference translation",
    "context": "Optional previous dialogue"
  }
]
```

**Usage:**

```powershell
# Run benchmark with auto-detected glossary
.\8-compare.ps1 data\ro_benchmark.json

# Run with explicit glossary
.\8-compare.ps1 data\ro_benchmark.json --glossary data\ro_glossary.json

# Run for other languages
.\8-compare.ps1 data\de_benchmark.json
```

**Template Files:**

- `data/ro_glossary.json` - Example glossary template
- `data/ro_benchmark.json` - Example benchmark data template

## Configuration

### Custom Prompts

The translation and correction prompts can be customized by editing the template files in `data/prompts/`:

- `data/prompts/translate.txt` - Translation prompt template
- `data/prompts/correct.txt` - Grammar correction prompt template

**Fallback hierarchy:**

1. Try custom templates in `data/prompts/`
2. Fall back to embedded template in code

These files use Python's `{variable}` syntax for placeholders. Edit them to adjust the translation style, add language-specific rules, or modify the behavior.

### Custom Glossary

Create a glossary JSON file to enforce consistent translations for game-specific terms:

```json
{
  "health potion": "poțiune de viață",
  "magic points": "puncte magice",
  "inventory": "inventar"
}
```

Place in `data\<language_code>_glossary.json` (e.g., `data\ro_glossary.json`, `data\es_glossary.json`, `data\ja_glossary.json`)

**Template:** See `data/ro_glossary.json` for an example with UI elements, character stats, and common game terms.

**Fallback Hierarchy:**

1. `data\<code>_glossary.json` (language-specific glossary)
2. No glossary (translation without term enforcement)

Both Aya-23-8B and MADLAD-400-3B support glossaries.

### Grammar Correction Rules

Create correction rules JSON file for pattern-based corrections:

```json
{
  "exact_replacements": {
    "wrong phrase": "correct phrase"
  },
  "verb_conjugations": [
    {
      "pattern": "incorrectă forma",
      "replacement": "formă corectă"
    }
  ],
  "protected_words": ["ProperName", "GameTitle"]
}
```

Place in `data\<language>_corrections.json`

## File Structure

```
├── src/                    # Core translation modules
│   ├── models.py          # Type-safe data structures for modular pipeline
│   ├── extract.py      # Extract .rpy → clean YAML + tags JSON
│   ├── merger.py          # Merge YAML + JSON → .rpy with validation
│   ├── batch_translator.py # Context-aware batch translation
│   ├── renpy_utils.py     # Ren'Py parsing and tag handling utilities
│   ├── core.py            # Aya23Translator class (original pipeline)
│   └── prompts.py         # Translation/correction prompts
├── scripts/               # Translation engine scripts (called by launchers)
│   ├── translate_with_aya23.py     # Aya-23-8B translation engine
│   ├── translate_with_madlad.py    # MADLAD-400-3B translation engine
│   ├── correct.py       # Aya-23-8B grammar correction engine
│   ├── benchmark.py       # BLEU benchmark script
│   ├── common.ps1         # Shared PowerShell functions
│   └── config_selector.ps1 # Interactive game/language selection
├── tests/                 # Automated tests
│   ├── test_end_to_end.py           # End-to-end translation tests
│   ├── test_unit_renpy_tags.py           # Tag preservation tests
│   └── test_unit_extract_merge.py     # Modular pipeline tests (NEW)
├── data/                  # Prompts, glossaries, benchmarks, and correction rules
│   ├── prompts/
│   │   ├── translate.txt              # Translation prompt template (customizable)
│   │   └── correct.txt                # Correction prompt template (customizable)
│   ├── ro_glossary.json          # Example glossary template
│   ├── ro_benchmark.json         # Example benchmark data template
│   └── ro_corrections.json       # Example correction rules
├── models/                # Model configs only (models live in the HF cache)
│   ├── models_config.yaml    # Model registry (repos, languages, sizes)
│   ├── compute_profiles.yaml # Per-tier hardware params
│   └── current_config.yaml   # Active game/model/language selection
├── tools/                 # External tools (gitignored)
├── renpy/                 # Ren'Py SDK (gitignored)
│   └── tools_config.json  # External tools configuration
├── games/                 # Game directories (gitignored)
│   └── <Game>/
│       └── game/tl/<language>/
│           ├── characters.json        # Character mappings (NEW)
│           ├── *.rpy                  # Original translation files
│           ├── *.parsed.yaml          # Clean text for editing (NEW)
│           ├── *.tags.json            # Tags and metadata (NEW)
│           └── *.translated.rpy       # Merged output (NEW)
├── requirements.txt       # Python dependencies
├── 0-setup.ps1            # Automated setup script
├── 8-compare.ps1        # PowerShell launcher for benchmark.py
├── 2-test.ps1             # Test runner
├── 1-config.ps1           # Character discovery & game setup (NEW)
├── 2-extract.ps1          # Extract .rpy → YAML/JSON (NEW)
├── 3-translate.ps1        # Interactive launcher (selects model, language, game)
├── 4-correct.ps1          # Interactive launcher for grammar correction
├── 5-merge.ps1            # Merge YAML/JSON → .rpy (NEW)
├── PIPELINE_USAGE.md      # Modular pipeline user guide (NEW)
├── MODULARISATION_PLAN.md # Technical specification (NEW)
└── IMPLEMENTATION_SUMMARY.md # Implementation details (NEW)

```

## License & Acknowledgments

MIT License - Use for any purpose, including commercial projects.

Acknowledgments

- **Models:**
  - [Aya-23-8B](https://huggingface.co/CohereForAI/aya-23-8B) by Cohere For AI
  - [MADLAD-400-3B](https://huggingface.co/google/madlad400-3b-mt) by Google Research
- **Quantization:**
  - [bartowski&#39;s GGUF conversion](https://huggingface.co/bartowski/aya-23-8B-GGUF)
  - [unsloth&#39;s 4-bit conversion](https://huggingface.co/unsloth/madlad400-3b-mt-4bit)
- **Frameworks:**
  - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (Aya-23-8B)
  - [unsloth](https://github.com/unslothai/unsloth) (MADLAD-400-3B)
