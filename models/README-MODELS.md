# 2024-2025 Model Options (Ranked by Quality)

The core requirements are: Gramatically correct, culturally and contextually aware translation EN to RO, uncensored, able to translate explicit adult content, able to use correct declensions, conjugations, syntax and topic in Romanian. They must run on a Windows PC with 16GB RAM, RTX3060 with 6GB VRAM + shared VRAM.

## Overview Table

| Model                                                                                       | Type                                          | Params (B, billions) | BLEU Score             | Tatoeba Score | Flores Score | Cell02 BLEU | VRAM GB required | Notes                                                                                                                                                                |
| :------------------------------------------------------------------------------------------ | :-------------------------------------------- | :------------------- | :--------------------- | :------------ | :----------- | :---------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Aya-23-8B](https://huggingface.co/cohere/aya-23-8B)**                                 | Multilingual LLM `<br>` GGUF                | 8                    |                        |               | 34.8         | 0.4157      | 5.8              | 👍 Uncensored, GGUF, 23 languages `<br>`👎 Slower, larger VRAM                                                                                                     |
| **[MBART-Ro-1B](https://huggingface.co/facebook/mbart-large-en-ro)**                     | __Ro-Translation__ `<br>` safetensors | 0.6                  | __38.0__ (WMT16) |               |              | 0.1563      | 2                | 👍 Largest RO-specific, good context.`<br>`👎 Smaller than multilinguals                                                                                           |
| **[MADLAD-400-3B](https://huggingface.co/google/madlad-400-3b-mt)**                      | Translations `<br>` safetensors             | 3                    | ~35.11                 |               | 38.4         | 0.4106      | 4                | 👍 Uncensored, safetensors, 400+ languages `<br>`👎 Requires `trust_remote_code`, lower quality for some languages                                               |
| **[Seamless-96-2B](https://huggingface.co/facebook/seamless-m4t-v2-large)**              | Multimodal `<br>` safetensors               | 2.3                  |                        |               | 38.8         | 0.4861      | 5+               | 👍 Most recent from Meta, better than NLLB.`<br>`👎 Includes unneeded speech features                                                                              |
| **[Helsinki-Ro-0B](https://huggingface.co/Helsinki-NLP/opus-mt-en-ro)**                  | __Ro-Translation__ `<br>` safetensors | 0.075                | 34.0 (WMT16)           |               |              | 0.3275      | 0.3              | 👍 Fast, lightweight Marian MT model.`<br>`👎 Smaller than MBART. Good for low VRAM.                                                                               |
| **[NLLB-200](https://huggingface.co/facebook/nllb-200-3.3B)**                            | Translations `<br>` safetensors             | 3.3                  | ~31.17                 |               | 37.5         | 0.2781      | 4.5              | 👍 Proven, stable, large community.`<br>`👎 Older model (2022).  Good for reliability.                                                                             |
| **[OPUS-MT-TC-Big](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ro)**           | Large OPUS `<br>` safetensors               | 0.2                  | 34.0 (Newstest2016)    | 48.6          | 40.4         |             | 1                | 👍 Good grammar for size, small footprint.`<br>`👎 Smaller than MBART, may be censored.  Good for low VRAM.                                                        |
| **[Helsinki-Tatoeba](https://huggingface.co/Helsinki-NLP/opus-tatoeba-en-ro)**           | Transformer-align `<br>` safetensors        | 0.078                | 31.7 (Newstest2016)    | 46.9          |              |             | 0.2              | 👍 Better than standard OPUS, tiny footprint.`<br>`👎 Small model, not for complex grammar.  Requires `>>ron<<` token.                                           |
| **[suzume-llama-3-8B](https://huggingface.co/lightblue/suzume-llama-3-8B-multilingual)** | Multilingual LLM `<br>` safetensors         | 8                    |                        |               |              |             | ~5-6             | 👍 Based on powerful Llama 3, likely uncensored, very new (Oct 2024).`<br>`👎 Romanian is not a focus, EN-RO performance is unknown.  Experimental high-potential  |
| **[Marcoroni-7B-v3](https://huggingface.co/TheBloke/Marcoroni-7B-v3-GGUF)**              | Instruct LLM `<br>` GGUF                    | 7                    |                        |               |              |             | ~4.8             | 👍 Strong Mistral base, likely uncensored, was #1 on 7B leaderboard.`<br>`👎 Not for translations, for general tasks. EN-RO performance is unknown.  Experimental. |
| **[OLMo-7B](https://huggingface.co/allenai/OLMo-7B)**                                    | Multilingual LLM `<br>` safetensors         | 7                    |                        |               |              |             | 5                | 👍 Fully open source.`<br>`👎 Research-focused, may not match SOTA. For open-source enthusiasts.                                                                   |
| **[BlackKakapo-MT](https://huggingface.co/BlackKakapo/opus-mt-en-ro)**                   | Community OPUS `<br>` safetensors           | 0.075                | ~24.5 (Estimated)      | 53.1          |              |             | 0.5              | 👍 Community fine-tuned.`<br>`👎 Single-person project, weakest grammar.  For extreme VRAM constraints.                                                            |
| **[Orion-14B](https://huggingface.co/OrionStarAI/Orion-14B)**                            | Multilingual LLM `<br>` safetensors         | 14                   |                        |               |              |             | 9                | 👍 Large context window.`<br>`👎 Too heavy for 6GB VRAM.                                                                                                           |
| **[OpenELM-3B](https://huggingface.co/apple/OpenELM-3B)**                                | Multilingual LLM `<br>` safetensors         | 3                    |                        |               |              |             | 2.5              | 👍 Very fast and lightweight.`<br>`👎 Too small for complex Romanian.  NOT RECOMMENDED.                                                                            |

---

## Removed/Unsupported Models

### LLMic-3B ❌ REMOVED

**Reason:** Translation functionality non-operational

While the [faur-ai/LLMic](https://huggingface.co/faur-ai/LLMic) model claims BLEU 41.01 on WMT16 EN-RO translation in its paper ([arXiv:2501.07721](https://arxiv.org/abs/2501.07721)), the publicly available Hugging Face model does not translate.

**Issues encountered:**

- Model generates random Romanian text unrelated to English input
- Multiple prompt formats tested (parallel corpus, few-shot, instruction-based) - all failed
- Model appears to be base pretrained version, not the translation-tuned variant
- No documentation on Hugging Face for translation usage or prompt format
- Suspected missing: translation adapter/LoRA or specific fine-tuned checkpoint

**Status:** The translation-capable version referenced in the paper is not publicly available or requires undocumented configuration. Removed from available models until proper translation checkpoint is released.

---

## Model Types Explained

- **Multilingual LLM:** General-purpose Large Language Models trained on many languages (e.g., Aya, Orion). They are good at understanding context but are not exclusively built for translation.
- **Instruct LLM:** A general-purpose LLM that has been fine-tuned to be good at following user commands or "instructions." Their translation ability varies.
- **Translations / Ro-Translations:** Models designed and trained specifically for translation tasks, either between many languages (Translations) or focused on Romanian (Ro-Translations).
- **Bilingual Ro-En:** Foundation models trained extensively on both Romanian and English, making them highly effective for translation between the two.
- **Multimodal:** Models that can process more than one type of data, such as both text and audio (e.g., SeamlessM4T).
- **OPUS / Transformer-align:** Architectures that are highly effective for translation. OPUS is a popular framework, and many models are built on it, sometimes with community fine-tuning.

  Current Status:

  | Model            | Status              | Notes                                         |
  | ---------------- | ------------------- | --------------------------------------------- |
  | Aya-23-8B        | ✅ Production Ready | Uses llama-cpp-python                         |
  | MADLAD-400-3B    | ✅ Production Ready | Works with float16 fallback                   |
  | SeamlessM4T-v2   | ✅ Production Ready | Works, slow to load (~90s)                    |
  | MBART-En-Ro      | ✅ Production Ready | Fixed source language setting                 |
  | Helsinki OPUS-MT | ✅ Production Ready | Fast Marian MT, sacremoses warning suppressed |
  | LLMic-3B         | ❌ REMOVED          | Doesn't translate (see above)                 |

  What Changed:


  1. ✅ Removed torchao package (it was causing the conflict)
  2. ✅ Fixed Unicode arrows in translator print statements
  3. ✅ Fixed MADLAD test to use HuggingFace auto-download instead of looking for GGUF file

# SETUP

## Model storage & loading

- **Location:** Models download to the default HuggingFace cache
  (`%USERPROFILE%\.cache\huggingface\hub\`), not into the project. They are
  shared across every project on the machine and downloaded only once. The
  project's `models/` folder holds only small YAML config files.
- **Loading:** At translation time models load **offline** from that cache (no
  network), so a throttled or offline connection never slows things down. If a
  needed file is missing it is fetched once, then cached.
- **Windows Developer Mode** must be enabled so the HF cache can use symlinks;
  without it, files are copied (~2x disk per model). Enable via
  `Settings > System > For developers`.

## Installation

**Setup Steps:**

1. **Model Selection** - Choose which models to install:

   - Aya-23-8B (4.8GB) - 23 languages, higher quality
   - MADLAD-400-3B (~6GB) - 400+ languages, broader coverage
   - Or install both models
2. **Python Environment** - Automatically:

   - Creates virtual environment (detects and repairs corruption)
   - Checks pip version (takes ~1 minute, only upgrades if needed)
   - Installs PyTorch with CUDA 12.4 (shows installation progress)
   - Installs model-specific packages:
     - llama-cpp-python with CUDA for Aya-23-8B (verifies CUDA support)
     - transformers for MADLAD-400-3B
   - Checks if packages already installed before reinstalling
   - Automatically uninstalls and reinstalls broken CUDA packages
3. **Model Download** - Downloads your selected models from HuggingFace into the
   default HF cache (`%USERPROFILE%\.cache\huggingface\hub\`), shared across all
   projects and downloaded only once
4. **External Tools** (translate-renpy only; skipped in translate-local):

   - Ren'Py SDK (downloads if missing)
   - rpaExtract.exe (included at `renpy/rpaExtract.exe`)
   - UnRen (included at `renpy/unRen/`)
5. **Language Configuration** - Select which languages you'll work with

   - Only shows languages supported by your selected models
   - Saves to `models/current_config.yaml`
   - Used to filter language choices in `3-translate.ps1` and `4-correct.ps1`
6. **Verification** - Tests all components:

   - Verifies Python packages can actually import (not just installed)
   - Checks CUDA availability
   - Confirms selected models are downloaded

**Optional Skip Flags:**

```powershell
.\0-setup.ps1 --skip-model      # Skip model download
.\0-setup.ps1 --skip-tools      # Skip Ren'Py SDK/tools download (translate-renpy only)
.\0-setup.ps1 --skip-python     # Skip Python environment setup
```

**Reconfigure Languages Later:**

```powershell
# Re-run setup with skip flags to only change language configuration
.\0-setup.ps1 --skip-python --skip-model --skip-tools
```

**Troubleshooting Setup Issues:**

If setup completes with warnings about missing packages:

```powershell
# Fix broken llama-cpp-python (if "NOT INSTALLED" warning appears)
.\0-setup.ps1 --skip-model --skip-tools

# The script will:
# 1. Detect the broken installation
# 2. Uninstall the CPU-only version
# 3. Reinstall with CUDA support
# 4. Verify it actually works
```

**Common Issues:**

- **"llama-cpp-python: NOT INSTALLED"** - CUDA wheel didn't install properly. Re-run setup with skip flags.
- **"Could not find module 'llama.dll'"** - CPU-only torch installed instead of CUDA. The setup script now automatically detects this and reinstalls torch with CUDA support. Re-run `.\0-setup.ps1`.
- **"CMake Error: CMAKE_C_COMPILER not set" or "Building wheel failed"** - Setup tried to build from source instead of using prebuilt wheel:
  - **Cause:** Your Python version (3.12+) may not have prebuilt CUDA wheels available
  - **Solution 1:** Use Python 3.10 or 3.11 (best wheel support)
  - **Solution 2:** Setup will automatically fallback to CPU-only version
  - **Solution 3:** Install Visual Studio Build Tools if you want to compile from source
- **Pip check takes forever** - This is normal, checking for outdated packages takes ~1 minute.
- **Virtual environment corrupted** - Setup automatically detects and recreates it.

### Manual Setup

If you prefer manual installation:

#### 1. Install Python Dependencies

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Or install manually with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

#### 2. Download Model

Pick a model from the Overview Table above.

```powershell
# Using huggingface-cli (downloads into the default HF cache, ~/.cache/huggingface)
huggingface-cli download bartowski/aya-23-8B-GGUF aya-23-8B-Q4_K_M.gguf
```

**Model:** Aya-23-8B Q4_K_M (4.8GB, January 2025 SOTA multilingual model)
Pick a model from the Overview Table above.

#### 3. Download Tools (Optional)

Download from `renpy/tools_config.yaml` or manually:

- [Ren&#39;Py SDK](https://www.renpy.org/latest.html)
- [rpaExtract](https://github.com/Kaskadee/rpaextract) (multiple fallback URLs configured)
- UnRen (already included in the repository at `renpy/unRen/`)

---

# Benchmark Quality Analysis — Cell02_Wonderland (70 blocks, EN→RO)

> Translations run on 2026-05-30 against `games/Example Uncensored/game/tl/romanian/Cell02_Wonderland.parsed.yaml`
> BLEU scored on 2026-05-31 via `9-benchmark.ps1` (N=31 blocks matched against `ro_uncensored_benchmark.yaml`)

## Compound Score Formula

`Score = 0.60 × COMET + 0.25 × chrF++ + 0.10 × METEOR + 0.05 × BLEU`

**Score is only calculated when all four metrics are available.** If any is missing the Score column is omitted entirely — partial scores computed with redistributed weights are not comparable across runs.

- **COMET — 60%.** The only metric here that was trained on actual human translation judgments. It uses the source sentence as a third input (not just hypothesis vs. reference), which means it can detect meaning loss that surface metrics miss entirely. Highest proven correlation with human evaluation. Dominant weight is justified: it is doing qualitatively different work from the other three.

- **chrF++ — 25%.** Character n-gram F-score extended with word-level bigrams. Operates at the character level with no language-specific resources, which makes it reliable across all FLORES languages without exception. Especially valuable for Romanian, where morphological inflections (verb conjugations, noun declensions, adjective agreement) produce surface variation that word-level metrics penalise as errors. High weight because it is both language-agnostic and sensitive to exactly the kind of morphological accuracy this pipeline requires.

- **METEOR — 10%.** Word-level matching with stemming and synonym lookup via WordNet. Better than BLEU in principle, but its synonym tables are English-centric: Romanian coverage in `omw-1.4` is thin, and NLTK's PorterStemmer was designed for English morphology, not Romanian. The semantic advantage METEOR holds over chrF++ on English largely disappears here. Kept in because it still adds a recall signal that the other metrics lack, but weighted low to reflect its diminished reliability on this language pair.

- **BLEU — 5%.** Simple n-gram precision with no linguistic knowledge. Correlates poorly with human judgment on short sentences, penalises valid paraphrases, and adds little that chrF++ does not already cover. Retained at a token weight for historical comparability with published benchmarks and older runs in `benchmarks.yaml`.

---

## Overall Ranking (Best → Worst for Adult EN→RO)

Timings are for 70 blocks from `Cell02_Wonderland` (RTX 5070 Laptop, 8GB VRAM). BLEU is avg sentence-BLEU over 31 matched blocks.

| Rank   | Key    | Model            | Type             | Params | Size     | Precision | Released | Duration | Avg BLEU‡ | Verdict                                                          |
| ------ | ------ | ---------------- | ---------------- | ------ | -------- | --------- | -------- | -------- | ---------- | ---------------------------------------------------------------- |
| 1      | `eu` | EuroLLM-9B       | EU-focused LLM   | 9B     | ~6.5GB   | Q5_K_M    | Oct 2024 | 259.43s  | 0.4899     | Best grammar and idiom; rare hallucinations                      |
| 2      | `ae` | Aya Expanse 8B   | Multilingual LLM | 8B     | ~5.5GB   | Q5_K_M    | Oct 2024 | 211.36s  | 0.4915     | Strong adult vocabulary; minor diacritic lapses                  |
| 3      | `ay` | Aya-23-8B        | Multilingual LLM | 8B     | 4.8GB    | Q4_K_M    | May 2024 | 191.51s  | 0.4157     | Solid baseline; misses English slang idioms                      |
| 4      | `se` | Seamless M4T-v2  | Multimodal MT    | 2.3B   | ~5GB     | float16   | Dec 2023 | 115.64s  | 0.4861     | Good structure; uses sanitizer sometimes                         |
| 5      | `he` | Helsinki OPUS-MT | Ro-Specific MT   | 75M    | ~0.3GB   | float16   | 2020     | 42.61s   | 0.3275     | Fast and usually correct; catastrophic on edge cases             |
| 6      | `nl` | NLLB-200         | Multilingual MT  | 600M   | ~1.2GB   | float16   | Jul 2022 | 39.60s   | 0.2781     | Uses sanitizer sometimes                                         |
| 7      | `ma` | MADLAD-400       | Multilingual MT  | 3B     | ~6GB     | float16   | Sep 2023 | 237.34s  | 0.4106     | Diacritics missing; leaves English words; wrong conjugations     |
| 8      | `mb` | MBART-En-Ro      | Ro-Specific MT   | 0.6B   | ~2GB     | float16   | Jan 2020 | 43.49s   | 0.1563     | Fundamental semantic failures; sanitizer                         |
| est.   | `el` | EuroLLM-22B-2512 | EU-focused LLM   | 22B    | 11–14GB | Q3/4_K_M  | Dec 2025 | —       | —         | Not benchmarked; exceeds 8GB VRAM; expected ≥ rank 1            |
| 4–5† | `tc` | OPUS-MT-TC-Big   | Ro-Specific MT   | 200M   | ~1GB     | float16   | 2022     | —       | —         | Not benchmarked; Flores 40.4; expected between `se` and `he` |
| 5–6† | `nb` | NLLB-1.3B        | Multilingual MT  | 1.3B   | ~2.5GB   | float16   | Jul 2022 | —       | —         | Not benchmarked; 2×`nl` params; expected above `nl`         |

† Not yet benchmarked against Cell02_Wonderland. Ranks estimated from Flores/Tatoeba scores and architecture. `e22` out of benchmark hardware range (requires 16GB+ VRAM).

‡ BLEU rank differs from subjective rank in two notable places: `se` (BLEU 2nd among non-LLMs at 0.4861) and `ma` (BLEU 0.4106, higher than `he`/`nl`). BLEU rewards n-gram overlap on matched sentences; the subjective rank penalises catastrophic failures, sanitizer substitutions, and explicit-vocabulary gaps that BLEU cannot detect on clean reference sentences.

---

## Per-Model Analysis

### 🥇 eu — EuroLLM-9B-Instruct (Q5_K_M)

**Strengths:**

- Best colloquial Romanian register of all models — sounds like a native speaker
- Correct subjunctive after `să` in most cases
- Strong adult vocabulary with proper declensions
- Context-aware: correctly handles gendered forms across turns

**Weaknesses:**

- **Line 7** `"Hit me."` → `"Love-me."` — catastrophic confusion between Romanian `lovești` (you hit) and English "love"; a likely tokenization hallucination
- **Line 40** output wrapped in `**bold markdown**` — model leaked formatting into output
- **Line 42** leaked `"Narrator:"` speaker label into output — role tag bleed, distinct from markdown injection
- **Line 55** `"Do you fuck me now or...?"` → `"Te fut acum sau...?"` — person inversion (outputs "I fuck you" instead of "you fuck me")
- **Line 63** `"Make him fuck me!"` → `"Fute-mă!"` — dropped the "him" agent entirely
- **Line 69** `"Jerk me until I ejaculate on your boobs"` → described a blowjob instead — complete act hallucination
- **Line 75** `"We fuck every night"` → `"ne împerechem"` (we mate/pair) — sanitizer substitution; unexpected residual register bias on longer context
- **Line 88** `"We fucked all night"` → `"Am făcut amor"` (made love) — same sanitizer pattern
- **Line 93** `"I will fuck you tomorrow"` → `"O să te fută mâine"` — 3rd-person subjunctive ("for him to fuck you") instead of 1st-person future ("I will fuck you")

**Patterns fixed:** Added prompt rule 10 prohibiting markdown output and non-source formatting symbols.

---

### 🥈 ae — Aya Expanse 8B (Q5_K_M)

**Strengths:**

- Correct adult terminology (pulă/pizdă/cur) in nearly all cases
- Good handling of imperative forms and second-person singular
- Better at natural Romanian word order than older Aya-23

**Weaknesses:**

- **Line 7** `"Hit me."` → `"Loveste-ma."` — missing diacritics on output (ș/ă absent)
- **Line 27** `"longjohn"` → `"pantalonii lungi"` (long trousers) — idiom not recognized as penis slang
- **Line 29** `"nipples"` → `"puii tăi"` (your chicks/offspring) — anatomical term absent from glossary
- **Line 55** `"Do you fuck me now or...?"` → `"Tu mă futei acum sau...?"` — imperfect tense `futei` instead of present `fuți`
- **Line 62** `"They all fucked me."` → `"Ei m-au fute toate."` — double error: past participle `fute` instead of `futut`; `toate` (neuter/fem) instead of `toți` (masc) for the subject
- **Line 89** `"They fucked each other."` → `"S-au futeau unul pe altul."` — impossible mixed aspect: perfect auxiliary + imperfect verb

**Patterns fixed:** Added `longjohn: "pulă"`, `nipple: "sfârc"`, `nipples: "sfârcuri"` to `ro_uncensored_glossary.yaml`.

---

### 🥉 ay — Aya-23-8B (Q4_K_M)

**Strengths:**

- Strong vocabulary for direct adult terminology
- Usually correct diacritics
- Good handling of context (speaker-aware)

**Weaknesses:**

- **Line 27** `"longjohn"` → `"pantalonii lungi"` — same idiom gap as Aya Expanse
- **Line 8** `"I will propose her to free herself"` → `"O voi propune să se elibereze"` — wrong object pronoun (should be `Îi voi propune`)
- **Line 41** added spurious asterisks around output

**Patterns fixed:** `longjohn` added to glossary (affects both Aya models via `glossary_prompt_entries`).

---

### se — Seamless M4T-v2 Large

**Strengths:**

- Grammatically structured output; handles complex sentences well
- Good sentence-level fluency for non-adult content

**Weaknesses:**

- **Lines 9, 40** `"cock"` → `"cocoșul/cocoș"` (rooster) — `cocoșă` (feminized form) was NOT in corrections, so it slipped through
- **Lines 38–39** replaced `*asterisk*` markers with `♪ music note ♪` — the model substitutes formatting symbols it associates with "emphasis"
- **Line 25** `"balls"` → `"bilele"` (marbles) — not caught before this fix cycle
- Uses `"ouă"` (eggs) as slang for balls — colloquially acceptable in Romanian but inconsistent with the glossary's `coaie`
- **Line 45** `"titties"` → `"sângele"` (blood) — confused `sânii` (breasts) with `sângele` (blood); form-collision in seq2seq attention
- **Lines 54, 78, 81** `"Fuck me/her/us"` → `"Dă-mi drumul / Dă-i drumul / La naiba"` — interprets sexual imperative "Fuck" as a release/freedom interjection; same pattern as `he`/`nl`/`ma` (see below)

**Patterns fixed:** Added `cocoșă` → `pulă`, `bilele/bile` → `coaiele/coaie`, `ouăle/ouă` → `coaiele/coaie` to `source_conditioned_replacements`. ♪ substitution is a known Seamless training artifact (music/lyrics data) and is not corrected post-processing.

---

### he — Helsinki OPUS-MT (opus-mt-en-ro)

**Strengths:**

- Very fast inference (< 1 min for 70 blocks)
- Zero VRAM requirement — runs on CPU
- Correct output for the majority of standard dialogue

**Weaknesses:**

- **Line 38** `"*Keep spanking her ass while cumming on her back*"` → `"* Keep pupking her cur while colibri on her back *"` — catastrophic: left English, invented words ("pupking", "colibri")
- **Line 39** `"*Keep fucking her cunt a little more*"` → `"* Keep futând her pizdă a little more *"` — partial failure: English leaking into output
- **Line 41** major content truncation: full sentence reduced to `"Suge-mi-o!"`
- Occasional `he/she` pronoun agreement errors on Romanian feminine nouns
- **Lines 54, 60, 81** imperative-as-profanity: `"Fuck me/my face/us!"` → `"La naiba cu mine/faţa mea/amândoi"` — parses sexual imperative "Fuck" as a Romanian profanity exclamation ("To hell with X"); shared pattern with `nl`, `ma`, `se`

**Assessment:** Acceptable for SFW content; unreliable for complex or explicit lines. Best used as a fast first-pass that gets corrected afterward.

---

### nl — NLLB-200-distilled-600M

**Strengths:**

- Reasonable sentence structure for non-explicit content
- Handles negation and subordinate clauses adequately

**Weaknesses:**

- **Lines 20–21** `"pussy (lover)"` → `"păsări"` (birds, plural) — the diminutive forms `păsărică/păsăricii` were in corrections but the base plural `păsări/păsările` was not
- **Line 25** `"balls"` → `"bilele"` (marbles) — same gap now fixed
- **Line 12** `"blowjobs"` → `"sexuri sexuale"` — "sexual sexes", a complete mistranslation with no equivalent form in corrections
- **Lines 54, 78, 81** imperative-as-profanity: `"Fuck me/her/us!"` → `"La naiba cu mine/cu ea/cu amândoi"` — same pattern as `he`/`ma`/`se`; parses "Fuck" as Romanian profanity exclamation
- Occasionally uses formal `dvs` register unpredictably

**Patterns fixed:** Added `păsări` → `pizde`, `păsările` → `pizdele`, `bilele/bile` → `coaiele/coaie` to corrections.

---

### ma — MADLAD-400-3B

**Strengths:**

- Broad language coverage; generally understands sentence structure

**Weaknesses:**

- **Lines 13, 25** missing diacritics in output (`pasarica`, `coaiele`)
- **Line 38** left English word "spanking" untranslated
- **Line 39** `"Keep fucking"` → `"Păstrați dracu"` (`dracu` = the devil, not a verb)
- **Lines 54, 81** imperative-as-profanity: `"Fuck me/us!"` → `"La dracu cu mine/cu noi"` — same pattern as `he`/`nl`/`se`
- Occasional wrong conjugation register (formal plural `Păstrați` for singular imperatives)
- Less fluent with long compound sentences

**Assessment:** Third-tier HF model — better than MBART but noticeably weaker than Seamless or Helsinki. Best for draft quality on non-explicit content.

---

### mb — MBART-En-Ro ❌ Not recommended for adult content

**Strengths:**

- Fastest among the fine-tuned Romanian-specific models
- Handles formal register Romanian well

**Critical Failures:**

- **Line 5** `"fuck like a goddess"` → `"să se sărute ca o zeiţă"` (to kiss like a goddess) — swapped explicit verb for euphemism
- **Line 7** `"Hit me."` → `"Daţi-mi voie să vă dau cuvântul."` ("Allow me to give you the floor") — semantically unrelated
- **Line 10** `"cock in her wet pussy"` → `"băţ în gâtul ei umed"` ("stick in her wet throat") — anatomical location error
- **Line 11** `"fucking her ass"` → `"bată în aer"` ("beat in the air") — meaning entirely lost
- **Lines 29, 50, 67, 69, 54, 60** — left English words untranslated in output (`"Show your sâni"`, `"Slide your pulă"`, `"fute me!"`)
- Inconsistent random use of formal `dvs/dumneavoastră` register

**Root cause:** MBART-large-en-ro was trained on Wikipedia and news corpora, which are (a) censored and (b) formal register. Adult terminology causes it to substitute safe alternatives or produce incoherent output. The fine-tuning data does not include any explicit content, so the model's training distribution strongly disagrees with the task.

**Verdict:** Do not use for adult content translation. Acceptable only for SFW games with formal dialogue.

---

### e22 — EuroLLM-22B-Instruct-2512 (Q3/Q4_K_M) — not yet benchmarked

**Expected characteristics:**

- Same EU-focused training pipeline as `eu` (EuroLLM-9B) with 22B parameters — significant quality jump expected across all Romanian-specific metrics
- Requires 16GB+ VRAM (Q3_K_M = 11.1GB, Q4_K_M = 13.7GB); out of range for the 8GB benchmark hardware
- Model description: matches Qwen-3-32B and Gemma-3-27B quality on EU languages
- Expected to fix or reduce `eu`'s person-inversion and act-hallucination errors by virtue of larger capacity
- Drop-in replacement in the pipeline for `eu`; uses the same `LlamaCppTranslator` path

**Status:** Added to `models_config.yaml`; not yet run against Cell02_Wonderland. Benchmark requires ≥16GB VRAM.

---

### tc — OPUS-MT-TC-Big (Helsinki-NLP/opus-mt-tc-big-en-ro) — not yet benchmarked

**Expected characteristics:**

- EN→RO specialist MT model, same Marian architecture as `he` (Helsinki OPUS-MT) but 200M vs 75M params
- Tatoeba score 48.6 (vs `he` ~34), Flores 40.4 (vs `he` ~34.8), BLEU 34.0 (Newstest2016) — substantially better on standard benchmarks
- Same failure mode as `he` for adult vocabulary (trained on SFW parallel corpora); explicit terminology will drift to safe alternatives
- `source_conditioned_replacements` post-processing will be needed, same as `he`
- Low VRAM (~1GB); fast inference; good for low-resource machines needing better quality than `he`
- Uses existing `QuickMTTranslator` class (same as `helsinkiRo`) — no new translator code required

**Estimated rank:** 4–5 — better grammar/structure than `he` for standard dialogue; same explicit-content limitations. Likely competitive with `se` on non-explicit lines.

---

### n13 — NLLB-200-distilled-1.3B — not yet benchmarked

**Expected characteristics:**

- Same Facebook NLLB-200 architecture as `nl` (600M), 2× the parameters at 1.3B
- Higher Flores score than `nllb200`; better diacritics and grammar are expected
- Same adult vocabulary limitations as `nl`; `source_conditioned_replacements` will still be needed
- VRAM ~2.5GB; fits on CPU-only or low-VRAM machines
- Uses existing `NLLB200Translator` class — no new translator code required

**Estimated rank:** 5–6 — above `nl`, same patterns but fewer grammar failures and better diacritic consistency.

---

## Glossary and Corrections Fixes Applied (2026-05-30)

### `data/ro_uncensored_glossary.yaml` additions

| English term | Romanian translation | Reason                                                              |
| ------------ | -------------------- | ------------------------------------------------------------------- |
| `longjohn` | `pulă`            | Slang for penis; LLM models translated literally as "long trousers" |
| `nipple`   | `sfârc`           | Anatomical term missing from glossary; models used "pui" (chick)    |
| `nipples`  | `sfârcuri`        | Plural form added alongside singular                                |

### `data/ro_uncensored_corrections.yaml` additions (source-conditioned replacements)

These fire post-translation for HF models that cannot follow prompt-injected glossaries:

| Source contains | Wrong form in translation | Corrected to                                          |
| --------------- | ------------------------- | ----------------------------------------------------- |
| `cock`        | `cocoșă`              | `pulă` (feminized rooster form — Seamless output) |
| `pussy`       | `păsări`              | `pizde` (bare plural "birds" — NLLB output)        |
| `pussy`       | `păsările`            | `pizdele` (definite plural)                         |
| `cunt`        | `păsări`              | `pizde`                                             |
| `cunt`        | `păsările`            | `pizdele`                                           |
| `balls`       | `bilele`                | `coaiele` (marbles — NLLB output)                  |
| `balls`       | `bile`                  | `coaie`                                             |
| `balls`       | `ouăle`                | `coaiele` (eggs — Seamless colloquial output)      |
| `balls`       | `ouă`                  | `coaie`                                             |
| `blowjob`     | `felații`              | `muie` (plural form added)                          |

### `data/prompts/translate_uncensored.txt` changes

Added rule 10:

> **OUTPUT FORMATTING:** Output ONLY the translated text. No markdown, no bold (**text**), no labels, no commentary. Preserve source formatting exactly: if the source uses *asterisks*, keep *asterisks*; if it uses ''double single quotes'', keep ''double single quotes''. Do NOT substitute other symbols (♪, #, etc.).

This targets EuroLLM's markdown injection (`**bold**`) and Seamless's `♪` substitution (note: Seamless cannot follow prompts, so the fix only helps LLM models; Seamless formatting remains a known limitation).

---

## Recurring Error Categories Across All Models

### 1. English slang idioms for anatomy not recognized

Models (all types) fail on non-obvious slang: `longjohn` (penis), `longjohn` (same), `glory hole` (in glossary — mostly OK).
**Fix:** Extend glossary proactively with English slang synonyms.

### 2. Source-conditioned vocabulary drift in HF models

HF seq2seq models (Helsinki, NLLB, MBART, MADLAD, Seamless) produce domain-shifted vocabulary because their training data is SFW. The `source_conditioned_replacements` mechanism in `translator_utils.py` mitigates this but requires enumerating every wrong surface form.
**Fix:** The corrections file must be maintained incrementally as new wrong forms surface.

### 3. Formal register contamination

MBART and occasionally NLLB use `dvs/dumneavoastră` (formal "you") forms. The prompt rule about singular informal "you" helps LLM models but HF models ignore it.
**Fix:** Add `source_conditioned_replacements` for common formal→informal substitutions (e.g. `vă` → `te`, `dumneavoastră` → `tu`) when source is clearly informal — but this risks over-correction on genuinely formal lines. Currently out of scope.

### 4. Subjunctive after `să` (indicative used instead)

Multiple models produce `să fute` (indicative) instead of correct `să fută` (subjunctive). The `verb_conjugations` patterns in the corrections file already catch this for `fute`, `suge`, `linge`, and several others.
**Fix:** Already handled; monitor for new verb forms emerging in other benchmark files.

### 5. Formatting symbol substitution (Seamless-specific)

Seamless M4T-v2 substitutes `♪` for `*` and other formatting. This is a quirk of its training data (which includes lyrics). No post-processing fix is viable without risking false positives on content that genuinely contains notes.
**Recommendation:** Strip all Ren'Py formatting markers before sending to Seamless and re-inject them afterward using the `.tags.yaml` reconstruction template.

### 6. Imperative-as-profanity confusion (HF models: `he`, `nl`, `ma`, `se`)

MT models trained on SFW corpora parse `"Fuck [X]!"` as a Romanian profanity exclamation rather than a sexual imperative:

- `"Fuck me!"` → `"La naiba cu mine!"` / `"La dracu cu mine!"` / `"Dă-mi drumul!"`
- `"Fuck her hard!"` → `"Ia dracu' cu ea!"` / `"Dă-i drumul tare!"`
- `"Fuck us both!"` → `"La naiba cu amândoi!"`

The model associates "Fuck" with the Romanian exclamation register rather than the imperative sexual verb. LLM models (`eu`, `ae`, `ay`) handle this correctly.
**Fix:** Add `source_conditioned_replacements` for these exact surface forms when source line starts with `"Fuck "` followed by a pronoun or possessive. Alternatively: add explicit `"Fuck me" → "Fute-mă"` pairs to the corrections file.

### 7. EuroLLM (`eu`) residual sanitizer on extended context

On longer or context-heavy lines, `eu` occasionally substitutes euphemisms — `"împerechem"` (to mate) for `futem`, `"am făcut amor"` for explicit past tense. This is not consistent (it translates `"fuck"` correctly on short direct lines) but suggests a context-length or probability-based trigger.
**Fix:** Add `source_conditioned_replacements` for `împerechem` → `futem` and `am/ai/a făcut amor` → explicit past-tense forms; also reinforce the prompt's "always use the explicit term" rule with an example pair.

---

## Recommendations

| Use case                             | Recommended model(s)                           |
| ------------------------------------ | ---------------------------------------------- |
| Production-quality adult translation | `eu` (EuroLLM) → `ae` (Aya Expanse)       |
| High VRAM available (16GB+)          | `e22` (EuroLLM-22B) — not yet benchmarked   |
| Upgrade from `he` (same VRAM)      | `tc` (OPUS-MT-TC-Big) — not yet benchmarked |
| Upgrade from `nl` (low VRAM)       | `n13` (NLLB-1.3B) — not yet benchmarked     |
| Fast first-pass, SFW content         | `he` (Helsinki) or `tc` (OPUS-MT-TC-Big)   |
| CPU-only / low VRAM                  | `he` (Helsinki) or `nl` (NLLB)             |
| Best bang-for-buck (quality/size)    | `eu` EuroLLM-9B at Q5_K_M                    |
| Do NOT use for adult content         | `mb` (MBART)                                 |

# PERFORMANCE IMPROVEMENT PLAN: Improve Translation Speed & Quality

## Context

The enro pipeline translates Ren'Py scripts using 8 local models across two backends:

- **llama.cpp** (Aya23, AyaExpanse8b, EuroLLM) — via `LlamaCppTranslator`
- **HF Transformers** (NLLB200, MADLAD400, SeamlessM4T, Helsinki, MBART) — each has its own translator class; all are now reachable from `scripts/translate.py` via the dispatch dict

GPU utilization during inference is 20-30%. This is expected for single-sequence autoregressive LLM decoding (memory-bandwidth bound), but HF seq2seq models can be improved significantly by batching multiple sentences per `model.generate()` call.

## Already completed (2026-05-30)

- `apply_ro_subjunctive()` wired into all 8 translators — fixes `să fute` → `să fută` and 17 other verb pairs, including contracted `s-o` forms
- `apply_source_conditioned()` wired into all 3 LLM translators — glossary misses (e.g. `cocoș` → `pulă`) are now caught post-generation
- `data/ro_uncensored_corrections.yaml` — exists and is fully populated with source-conditioned replacements, exact replacements, verb conjugations, gender agreement, and protected words
- `Aya23Translator` migrated to thin subclass of `LlamaCppTranslator` — no more duplicate code
- **Stage 1** — `translate_batch()` implemented on all 5 HF translators; unit-tested in `tests/test_unit_translate_batch.py`
- **Stage 2** — `scripts/translate.py` dispatch + batch loop (see below); unit-tested in `tests/test_unit_translate.py`

---

## Planned Improvements

### ✅ Stage 1 — `translate_batch()` for all 5 HF translators [Speed — HIGH IMPACT]

`translate_batch(texts: list[str]) -> list[str]` added to each HF translator. Calls `model.generate()` once for N sentences instead of N times, directly raising GPU utilization.

Files: `nllb200_translator.py`, `madlad400_translator.py`, `seamless96_translator.py`, `helsinkyRo_translator.py`, `mbartRo_translator.py`

### ✅ Stage 2 — `translate.py` model dispatch [Speed — required for batching in production]

`scripts/translate.py` no longer hard-codes `Aya23Translator`. A dispatch dict routes any configured model to its translator class; `ModularBatchTranslator` accumulates blocks and calls `translate_batch()` in chunks when available.

```python
_HF_TRANSLATORS = {
    'nllb200':    ('translators.nllb200_translator',    'NLLB200Translator'),
    'madlad400':  ('translators.madlad400_translator',  'MADLAD400Translator'),
    'seamlessm96':('translators.seamless96_translator', 'SeamlessM4Tv2Translator'),
    'helsinkiRo': ('translators.helsinkyRo_translator', 'QuickMTTranslator'),
    'mbartRo':    ('translators.mbartRo_translator',    'MBARTTranslator'),
}
_LLAMA_MODELS = {'aya23', 'ayaExpanse8b', 'euroLLM9b', 'euroLLM22b'}
```

`hf_batch_size` added to `compute_profiles.yaml` per tier: `cpu_only: 4`, `low: 8`, `medium: 16`, `high: 32`. LLAMA models always use `hf_batch_size=1` (single-item `translate()`). `compute_profile.yaml` is read at runtime to resolve both llama hw params and the current tier's batch size.

### Stage 3a — Add `euroLLM9b2512` (LLAMA/GGUF) [Quality — drop-in upgrade for `eu`]

Add EuroLLM-9B-Instruct-2512 as a LLAMA model. It is architecturally identical to `euroLLM9b` — the only difference is the GGUF file and improved post-training. No new translator code required.

**`models/models_config.yaml`** — add `euroLLM9b2512` entry following the `euroLLM9b` pattern: `repo`, `format: GGUF`, `Q4_K_M` and `Q5_K_M` file variants (no `destination` — models resolve to the HF cache by repo id).

GGUF repo: expected `bartowski/utter-project_EuroLLM-9B-Instruct-2512-GGUF` — verify on HuggingFace (same naming convention as `bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF`).

**`models/compute_profiles.yaml`** — add `euroLLM9b2512` to all four tiers, copying `euroLLM9b` params exactly.

**`scripts/translate.py`**:

```python
_LLAMA_MODELS = {'aya23', 'ayaExpanse8b', 'euroLLM9b', 'euroLLM9b2512', 'euroLLM22b'}
```

**PowerShell scripts** — add `euroLLM9b2512` everywhere `euroLLM9b` appears:

- **`0-setup.ps1`** — model-selection menu entry
- **`1-config.ps1`** — valid-model list + display name
- **`3-translate.ps1`** — verify `-Model euroLLM9b2512` is accepted (no model-specific logic expected)
- **`4-correct.ps1`** — add to the LLM branch (alongside `euroLLM9b`) if the script branches on model type
- **`7-test.ps1`**, **`8-compare.ps1`**, **`9-benchmark.ps1`** — add key to model lists

**Verification** — `3-translate.ps1 -Model euroLLM9b2512` loads and produces Romanian output; `8-compare.ps1` shows it alongside `euroLLM9b`; benchmark score should exceed `euroLLM9b` on translation metrics.

---

### Stage 3b — Add `opusTCBig` + `nllb1300` (HF models) [Quality — better small-model options]

Add two HF models that reuse existing translator classes. Grouped together because the work pattern is identical for both — no compute-profile entries, same dispatch path.

| Key           | Model                   | Translator class      | Reuses                 | Repo                                  |
| ------------- | ----------------------- | --------------------- | ---------------------- | ------------------------------------- |
| `opusTCBig` | opus-mt-tc-big-en-ro    | `QuickMTTranslator` | same as `helsinkiRo` | `Helsinki-NLP/opus-mt-tc-big-en-ro` |
| `nllb1300`  | nllb-200-distilled-1.3B | `NLLB200Translator` | same as `nllb200`    | `facebook/nllb-200-distilled-1.3B`  |

**`models/models_config.yaml`**:

- `opusTCBig` follows `helsinkiRo` pattern: safetensors, `model_class: MarianMTModel`, languages: `['ro']`
- `nllb1300` follows `nllb200` pattern: safetensors, same language list as `nllb200`

**`scripts/translate.py`**:

```python
_HF_TRANSLATORS = {
    ...
    'opusTCBig': ('translators.helsinkyRo_translator', 'QuickMTTranslator'),
    'nllb1300':  ('translators.nllb200_translator',    'NLLB200Translator'),
}
```

No translator code changes — both classes already accept `model_path` and `lang_code`.

**PowerShell scripts** — add both keys wherever HF model keys appear:

- **`0-setup.ps1`** — model-selection menu + HuggingFace download entries for each
- **`1-config.ps1`** — valid-model list + display names
- **`3-translate.ps1`** — no model-specific logic expected; verify `-Model` parameter accepts new keys
- **`4-correct.ps1`** — add to HF branch if the script branches on model type
- **`7-test.ps1`**, **`8-compare.ps1`**, **`9-benchmark.ps1`** — add both keys to model lists

**Verification**:

- `3-translate.ps1 -Model opusTCBig / nllb1300`: each loads and produces Romanian output
- `8-compare.ps1`: both appear in comparison output
- `9-benchmark.ps1`: `opusTCBig` expected Tatoeba ~48.6 / Flores ~40.4; `nllb1300` expected higher Flores than `nllb200`

### Stage 4 — MADLAD400 4-bit quantization [Speed + VRAM]

`BitsAndBytesConfig` is already imported but unused. Use `device_map={"": device}` (not `"auto"`) to avoid the large-vocabulary CPU-offload bug. Add `use_4bit: true` to `low` and `medium` tier sections in `compute_profiles.yaml`.

```python
if use_4bit and device == "cuda":
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(
        ..., quantization_config=quant_config, device_map={"": device}
    )
else:
    self.model = AutoModelForSeq2SeqLM.from_pretrained(..., torch_dtype=torch.bfloat16)
    self.model = self.model.to(self.device)
    self.model.decoder.embed_tokens.weight = self.model.shared.weight
```

Not applying to SeamlessM4T (different model class, less VRAM payoff at float16).

### Stage 5 — Clearer LLM speaker context [Quality — low effort]

In `llama_cpp_translator.py:_build_translation_prompt()`, the speaker is currently a detached trailing hint (`\nSpeaker: {speaker}`). Moving it inline with the translation instruction makes the prompt clearer:

```python
speaker_hint = f"spoken by {speaker}" if speaker else "narration"
# prompt template: "Translate this line ({speaker_hint}) to {target_language}:"
```

Requires updating `{speaker_hint}` usage in `data/prompts/translate_uncensored.txt` (and `translate.txt`).

---

## Implementation Order

```
✅ Stage 1  — translate_batch() for 5 HF translators       → tests/test_unit_translate_batch.py
✅ Stage 2  — translate.py model dispatch + batch loop      → tests/test_unit_translate.py
   Stage 3a — Add euroLLM9b2512 (LLAMA/GGUF)               → test
   Stage 3b — Add opusTCBig + nllb1300 (HF)                → test
   Stage 4  — MADLAD 4-bit quantization                     → test
   Stage 5  — LLM speaker context formatting                → test
```

## Verification

- **Stage 1** ✅: `tests/test_unit_translate_batch.py` — batch output matches single-item for all 5 HF translators
- **Stage 2** ✅: `tests/test_unit_translate.py` — batch path and single-path routing verified with mocks; run `3-translate.ps1` with each HF model key for live confirmation
- **Stage 3a**: `3-translate.ps1 -Model euroLLM9b2512` loads and translates; `8-compare.ps1` shows it alongside `euroLLM9b`
- **Stage 3b**: `3-translate.ps1 -Model opusTCBig / nllb1300` loads and translates; `9-benchmark.ps1` records scores
- **Stage 4**: Check VRAM in Task Manager before/after; run `8-compare.ps1` on MADLAD; output should remain mostly-Latin
- **Stage 5**: Run `8-compare.ps1` on LLM models; check for improved speaker attribution in complex dialogue blocks

---

# PRESENTATION

## EN→RO translation on a laptop GPU: what actually works

I spent a while automating the translation of visual novels from English to Romanian — fully local, no cloud, running on an RTX 3060 with 6GB VRAM. Romanian isn't exactly a priority for most NLP teams, so I went through everything I could find: roughly a dozen models evaluated, a few discarded before even making it to the benchmark (LLMic-3B from faur-ai claims BLEU 41 in its paper but the public checkpoint simply doesn't translate — it generates random Romanian text unrelated to the input), and 8 that made it to a proper head-to-head on the same 70 lines.

Here's what came out of it.

**EuroLLM-9B came out on top**

A relatively niche 9B model built around European languages beat everything else. It sounds like a native speaker — correct subjunctive after "să", consistent gendered forms across lines, natural informal register. Its quirks: it leaked markdown bold into output a couple of times, and once translated "Hit me" as "Love me" (a tokenization hallucination that's more funny than harmful). Both fixable with a prompt rule. Nothing else came close on overall quality.

**Aya Expanse 8B and Aya-23-8B are solid**

Cohere's multilingual models do well. Aya Expanse edges out the older Aya-23 on word order and vocabulary. Both get tripped up by English slang that doesn't have an obvious Romanian equivalent — "longjohn" confidently becomes "long trousers" — but you can patch these with a glossary file and move on.

**Helsinki OPUS-MT is 75MB and mostly fine**

A 2020 model that runs on CPU. For plain dialogue it's usually correct and it's very fast. Push it toward anything grammatically complex and it starts inventing words mid-sentence. Good for a quick first pass on simple content, not much else.

**MBART was trained specifically on Romanian and is somehow the worst**

This one is actually interesting. MBART has a BLEU score, it has a paper, it was fine-tuned on Romanian. It also translated "fuck like a goddess" as "to kiss like a goddess", and "Hit me" as "Allow me to give you the floor."

It's not a bad model — it was trained on Wikipedia and news corpora, which are both censored and very formal. When it hits vocabulary outside that distribution it doesn't fail noisily, it substitutes the nearest formal-register equivalent and delivers it with complete confidence. That's arguably worse than failing openly.

The takeaway: benchmark scores on news datasets say almost nothing about how a model handles everyday or domain-specific language.

**Romanian-specific things that matter**

Romanian has case, grammatical gender, and a subjunctive mood that's not optional. A model that uses the formal "dvs" register for a 20-year-old talking to a friend produces a translation that sounds wrong even if it's technically accurate. EuroLLM handles this without being told. MADLAD and NLLB often don't.

Diacritics are another common failure point. Smaller encoder-decoder models drop them under load: "pasarica" instead of "păsărică", "sa" instead of "să". The LLMs are generally consistent here.

**Short version**

- **EuroLLM-9B Q5_K_M** — best quality, fits in 6.5GB VRAM, current recommendation
- **Helsinki OPUS-MT** — when speed matters more than quality, SFW content only
- **MBART** — avoid unless the content is strictly formal

The gap between dedicated MT models and general-purpose LLMs has closed. For a morphologically complex language like Romanian, a well-trained multilingual LLM now beats purpose-built translation systems. That wasn't obviously true two years ago.

# PRESENTATION
