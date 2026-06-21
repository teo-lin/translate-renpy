# Automated Batch Ren'Py Translation

Local, fully offline translation of Ren'Py visual novels from English into
Romanian (and 400+ other languages) using state-of-the-art local LLMs and MT
models on a consumer laptop GPU (tested on RTX 3060 / RTX 5070 Laptop, Windows).

Context-aware, grammatically correct, tone-preserving, and uncensored (able to
translate explicit adult content with correct Romanian declensions, conjugations,
and syntax).

> **This repo (`translate-renpy`) is the full Ren'Py kit:** it downloads the
> Ren'Py SDK plus `rpaExtract` and `UnRen` so you can unpack `.rpa` archives,
> generate translation templates, and rebuild the game. If you already have the
> unpacked `game/tl/<language>/*.rpy` files and want only the translation engine,
> use the lighter `translate-local` repo instead. A future single-codebase split
> is described in `DECOUPLING PLAN README.md`.

## Features

- **Automated setup** - one interactive script installs the venv, dependencies,
  models, and Ren'Py tooling.
- **Multi-model support** - choose among 13 translation models (LLM and MT
  backends); see `models/README-MODELS.md` for the full ranked comparison.
- **Hardware-aware** - detects a compute tier (cpu_only / low / medium / high)
  and picks model quantization and batch size accordingly.
- **Modular pipeline** - Config -> Extract -> Translate -> Correct -> Merge, with
  human-reviewable YAML between every stage.
- **Preserves Ren'Py formatting** - `{color=...}`, `{size=...}`, `{b}...{/b}`,
  and `[variables]` are stripped before translation and restored on merge.
- **Glossary + correction support** - enforce terminology and fix recurring
  model errors via per-language YAML files.
- **Quality benchmarking** - COMET / chrF++ / METEOR / BLEU compound scoring
  against reference translations.
- **Grammar correction** - optional LLM + pattern post-processing pass.
- **Full GPU acceleration** - CUDA 12.4, uses all available GPU layers.

## Prerequisites

- **NVIDIA GPU** with 6 GB+ VRAM (CUDA 12.4) - tested on RTX 3060. CPU-only also
  works for the small MT models, slower.
- **Windows Developer Mode** - enable so the HuggingFace cache can use symlinks;
  without it model files are duplicated (~2x disk).
  Enable: `Settings > System > For developers > toggle ON`.

## Setup

Run once. It walks through 8 steps: hardware detection, language selection, model
selection, Python environment, model download, **external tools (Ren'Py SDK +
rpaExtract + UnRen)**, verification, and writing the compute profile.

```powershell
.\0-setup.ps1            # interactive: pick languages and models at the prompt
```

Skip flags for re-runs:

```powershell
.\0-setup.ps1 --skip-model      # skip model download
.\0-setup.ps1 --skip-tools      # skip Ren'Py SDK / tools download
.\0-setup.ps1 --skip-python     # skip venv / dependency setup
.\0-setup.ps1 --languages all   # non-interactive language selection
```

**Model storage:** models download to the default HuggingFace cache
(`%USERPROFILE%\.cache\huggingface\hub\`), shared across projects and fetched
only once. The project's `models/` folder holds only small YAML config files.

## Translation Pipeline

Each launcher is interactive and reads the selections you made during setup. They
forward to the Python entry points shown in parentheses.

```powershell
.\1-config.ps1      # one-time per game: set game path, language, model  (scripts/config.py)
.\2-extract.ps1     # .rpy -> .parsed.yaml (clean text) + .tags.yaml     (src/extract.py)
.\3-translate.ps1   # batched translation of the .parsed.yaml files       (scripts/translate.py)
.\4-correct.ps1     # optional grammar/pattern correction pass            (scripts/correct.py)
.\5-merge.ps1       # .parsed.yaml + .tags.yaml -> .translated.rpy        (src/merge.py)
```

Common arguments (the launchers also accept these and pass them through):

```powershell
.\1-config.ps1 -GamePath "games\MyGame" -Language ro -Model euroLLM9b
.\2-extract.ps1 -GameName MyGame -All          # or -Source script.rpy
.\3-translate.ps1                              # uses current_game from config
.\4-correct.ps1 "games\MyGame\game\tl\romanian" --patterns-only   # or --dry-run / --llm-only
.\5-merge.ps1 -GameName MyGame -All            # or -Source script ; --skip-validation
```

See `games/Example/README.md` for an end-to-end walkthrough on the bundled
example game.

## Benchmarking

```powershell
.\8-compare.ps1     # translate a reference set with a model and score it (scripts/compare.py)
.\9-benchmark.ps1   # score existing translations against a benchmark file (scripts/benchmark.py)
```

Reference data lives in `data/` as YAML, e.g. `data/ro_benchmark.yaml` and
`data/ro_uncensored_benchmark.yaml`. Compound score:
`0.60*COMET + 0.25*chrF++ + 0.10*METEOR + 0.05*BLEU` (computed only when all four
metrics are available). COMET and METEOR are optional downloads offered during
setup. Full methodology and the model ranking are in `models/README-MODELS.md`.

## Testing

```powershell
.\7-test.ps1                          # interactive test runner
.\venv\Scripts\pytest.exe tests/      # or run pytest directly
```

See `tests/TESTS-README.md` for the test map and execution details.

## Configuration

### Prompts

Translation and correction prompts are plain-text templates in `data/prompts/`:

- `translate_uncensored.txt` / `translate.txt` - translation (the `_uncensored`
  variant is tried first, then the plain one)
- `correct_uncensored.txt` / `correct.txt` - grammar correction

They use Python `{variable}` placeholders. Edit them to change style or rules.

### Glossary

Per-language YAML enforcing consistent terminology, e.g. `data/ro_glossary.yaml`
and `data/ro_uncensored_glossary.yaml`:

```yaml
health potion: poțiune de viață
magic points: puncte magice
inventory: inventar
```

### Correction rules

Per-language YAML of pattern fixes applied after translation, e.g.
`data/ro_uncensored_corrections.yaml`: exact replacements, source-conditioned
replacements, verb conjugations, gender agreement, and protected words. These
matter most for the SFW-trained MT models (see `models/README-MODELS.md`).

## File Structure

```
.
├── 0-setup.ps1                # setup (venv, deps, models, Ren'Py tools)
├── 1-config.ps1               # game setup launcher
├── 2-extract.ps1              # .rpy -> YAML launcher
├── 3-translate.ps1            # translation launcher
├── 4-correct.ps1              # grammar correction launcher
├── 5-merge.ps1                # YAML -> .rpy launcher
├── 7-test.ps1                 # test runner
├── 8-compare.ps1              # benchmark-by-translating launcher
├── 9-benchmark.ps1            # score-existing-translations launcher
├── src/
│   ├── setup.py               # setup implementation (8-step flow)
│   ├── extract.py             # .rpy -> .parsed.yaml + .tags.yaml
│   ├── merge.py               # .parsed.yaml + .tags.yaml -> .translated.rpy
│   ├── renpy_utils.py         # Ren'Py tag handling + shared utils
│   ├── config_utils.py        # shared game-config loading
│   ├── models.py              # typed data structures for the pipeline
│   ├── hardware.py            # compute-tier detection / profile
│   ├── prompts.py             # prompt-template loading
│   └── translators/           # one module per model backend
│       ├── llama_cpp_translator.py   # GGUF LLMs (Aya, AyaExpanse, EuroLLM)
│       ├── aya23_translator.py
│       ├── madlad400_translator.py
│       ├── mbartRo_translator.py
│       ├── nllb200_translator.py
│       ├── seamless96_translator.py
│       ├── helsinkyRo_translator.py
│       └── translator_utils.py
├── scripts/
│   ├── config.py              # game configuration
│   ├── translate.py           # ModularBatchTranslator engine + dispatch
│   ├── correct.py             # grammar correction engine
│   ├── correct_utils.py       # correction argument helper (used by 4-correct.ps1)
│   ├── compare.py             # translate + score
│   ├── benchmark.py           # score existing translations
│   └── config_selector.py     # interactive selection helpers
├── data/
│   ├── prompts/               # translate(.|_uncensored).txt, correct(.|_uncensored).txt
│   ├── ro_glossary.yaml       # glossary templates
│   ├── ro_*_benchmark.yaml    # benchmark reference data
│   └── ro_uncensored_corrections.yaml
├── models/
│   ├── models_config.yaml     # model registry (repos, languages, sizes)
│   ├── compute_profiles.yaml  # per-tier hardware params + batch sizes
│   ├── current_config.yaml    # active game/model/language selection
│   └── README-MODELS.md       # model comparison, benchmarks, setup details
├── renpy/                     # Ren'Py SDK + tools (gitignored except rpaExtract/unRen)
│   ├── rpaExtract.exe
│   └── unRen/
├── games/                     # game directories (gitignored except Example)
│   └── Example/               # bundled demo game (see its README.md)
├── tests/                     # pytest suite (see TESTS-README.md)
├── requirements.txt
├── pyproject.toml
└── DECOUPLING PLAN README.md  # plan to split core engine from the Ren'Py plugin
```

## Models

13 models are supported across two backends (llama.cpp GGUF for the LLMs, HF
Transformers for the MT models). For the full ranked comparison, per-model error
analysis, VRAM requirements, and the EN->RO benchmark results, see
**`models/README-MODELS.md`**.

## License & Acknowledgments

MIT License - use for any purpose, including commercial projects.

- **Models:** [EuroLLM](https://huggingface.co/utter-project),
  [Aya-23 / Aya Expanse](https://huggingface.co/CohereForAI) by Cohere For AI,
  [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt) by Google,
  [SeamlessM4T / NLLB / MBART](https://huggingface.co/facebook) by Meta,
  [OPUS-MT](https://huggingface.co/Helsinki-NLP) by Helsinki-NLP.
- **Frameworks:** [llama-cpp-python](https://github.com/abetlen/llama-cpp-python),
  [transformers](https://github.com/huggingface/transformers),
  [unbabel-comet](https://github.com/Unbabel/COMET).
