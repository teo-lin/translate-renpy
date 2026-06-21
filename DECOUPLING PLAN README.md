# Decoupling Plan — Core Translation Engine + Ren'Py Plugin

**Status:** proposal, not yet implemented
**Goal:** Collapse the two diverging repos (`translate-renpy`, `translate-local`) into **one** codebase with a clean core/plugin split, so the engine can run:

- **standalone** (plain text / `.parsed.yaml` in, translated out) — replaces `translate-local`
- **with the Ren'Py plugin** (`.rpy` extract/merge) — replaces `translate-renpy`

---

## 1. Why this is feasible (the boundary already exists)

The translation **core is already decoupled** from Ren'Py. `scripts/translate.py` imports only the generic `show_progress` from `renpy_utils` and operates entirely on a **format-neutral intermediate**:

```
.parsed.yaml   # clean translatable text blocks (Dict[block_id, ParsedBlock])
.tags.yaml     # structure + metadata needed to reconstruct the source
```

It never reads `.rpy`. The Ren'Py-specific code is confined to a small surface:

| Ren'Py-specific (→ plugin) | Generic (→ core, currently mislabeled) |
|---|---|
| `RenpyTagExtractor`, `RenpyBlock` (in `renpy_utils.py`) | `show_progress`, `language_name_from_code`, `detect_language_from_path` (also in `renpy_utils.py`) |
| `src/extract.py` (`.rpy` → `.parsed.yaml`/`.tags.yaml`) | `translators/`, `models.py`, `translate.py`, `correct.py`, `benchmark.py`, `compare.py`, `hardware.py`, `prompts.py`, `config.py` |
| `src/merge.py` (`.parsed.yaml` → `.rpy`) | |
| `renpy/` SDK, `rpaExtract.exe`, `unRen`, setup's tools download | |

**Root problem:** `renpy_utils.py` is a grab-bag — only 2 of its 5 members are actually Ren'Py-specific. This mislabeling is also the root cause of the cross-repo drift (each repo inlined/refactored these helpers differently).

The pipeline maps cleanly onto core vs. plugin:

```
1-config    → core
2-extract   → REN'PY PLUGIN   (.rpy → .parsed.yaml/.tags.yaml)
3-translate → core            (operates on .parsed.yaml + .tags.yaml)
4-correct   → core
5-merge     → REN'PY PLUGIN   (.parsed.yaml → .rpy)
8-compare   → core
9-benchmark → core
```

> Note: the deleted `packages/` experiment in `translate-local` (`poly_trans` = core, `poly_ren` = Ren'Py, `poly_bench` = benchmarking) was reaching for exactly this split. The decomposition was correct; it was simply never wired into the run path or installed. This plan finishes that idea.

---

## 2. Target architecture

One repo, a core package plus optional plugins discovered via entry points:

```
poly_trans/                 # CORE — zero Ren'Py imports
  models.py                 # ParsedBlock + the intermediate-format types
  translate.py              # ModularBatchTranslator (engine)
  correct.py
  benchmark.py  compare.py
  hardware.py  prompts.py  config.py
  utils.py                  # show_progress, language_name_from_code, detect_language_from_path
  translators/              # aya23, madlad400, mbartRo, seamless96, nllb200, llama_cpp, ...
  adapters/
    base.py                 # FormatAdapter ABC (the plugin contract)
    passthrough.py          # standalone: plain text / .parsed.yaml passthrough

poly_ren/                   # OPTIONAL PLUGIN — depends on poly_trans
  renpy_utils.py            # RenpyTagExtractor, RenpyBlock (Ren'Py-only)
  extract.py                # .rpy → .parsed.yaml/.tags.yaml
  merge.py                  # .parsed.yaml → .rpy
  sdk.py                    # Ren'Py SDK / rpaExtract / unRen download (moved out of setup.py)
  # registers adapter "renpy" via [project.entry-points."poly_trans.adapters"]
```

Install profiles:

- `pip install poly_trans` → **standalone** (only the `passthrough` adapter is available)
- `pip install poly_trans poly_ren` → **Ren'Py support** (adds the `renpy` adapter + SDK tooling)

### The plugin contract

```python
# poly_trans/adapters/base.py
class FormatAdapter(ABC):
    name: str                       # e.g. "renpy", "passthrough"

    @abstractmethod
    def extract(self, source: Path, out_dir: Path) -> list[Path]:
        """Source format -> .parsed.yaml (+ .tags.yaml). Returns parsed files."""

    @abstractmethod
    def merge(self, parsed_dir: Path, out_dir: Path) -> list[Path]:
        """Translated .parsed.yaml (+ .tags.yaml) -> source format."""
```

`2-extract` / `5-merge` resolve the configured adapter by name (default `passthrough`).
If no plugin is installed, only `passthrough` exists and Ren'Py stages are unavailable — by design.

---

## 3. Current drift to reconcile (decide canonical version per item)

Before merging the repos, settle the divergences found between them:

| Item | `translate-renpy` | `translate-local` | Recommended canonical |
|---|---|---|---|
| `load_game_config` | shared `config_utils.py` | inlined into `extract.py` **and** `merge.py` (duplicated) | renpy's shared module → move to core `config.py`/`utils.py` |
| `madlad400_translator` | no quantization | `BitsAndBytesConfig` (quant) | local (keep quantization) |
| `mbartRo_translator` | `AutoTokenizer` only | adds `MBartTokenizer` | local |
| `correct.py` lang helpers | imports from `renpy_utils` | inlined own copy | core `utils.py` (single source) |
| `setup.py` | downloads Ren'Py SDK | SDK step removed (already done) | core has no SDK step; plugin adds it |
| tests readme name | `TESTS-README.md` | `README.md` | pick one |

(Dead code in `translate-local` — `batch_translator.py`, unused TypedDicts, orphaned `detect_language_from_path` — has already been removed.)

---

## 4. Staged migration (test + commit between each stage)

Each stage is independently testable; nothing later depends on a half-done earlier stage beyond what's stated.

### Stage 1 — Split `renpy_utils.py` (un-mislabel)
- Create `utils.py`; move `show_progress`, `language_name_from_code`, `detect_language_from_path` into it.
- Leave only `RenpyTagExtractor` + `RenpyBlock` in `renpy_utils.py`.
- Update imports (`translate.py`, `compare.py`, `correct.py`, `extract.py`, `merge.py`).
- **Do this in BOTH repos identically** so they stop drifting on these helpers.
- Test: full pipeline still runs; `pytest` green.

### Stage 2 — Reconcile drift (Section 3)
- Port quantization changes renpy<-local; port `config_utils` refactor local<-renpy.
- After this, `src/` + `scripts/` should differ between repos **only** in: `extract.py`, `merge.py`, `renpy_utils.py`, `setup.py` (the genuine Ren'Py surface).
- Test: `diff -rq` between repos shows only the expected Ren'Py files.

### Stage 3 — Introduce the adapter interface
- Add `adapters/base.py` (`FormatAdapter`) and `adapters/passthrough.py`.
- Refactor `extract.py`/`merge.py` to implement `FormatAdapter` (Ren'Py adapter).
- Make `2-extract`/`5-merge` resolve an adapter by name (config key, default `passthrough`).
- Test: Ren'Py flow works via the `renpy` adapter; standalone flow works via `passthrough`.

### Stage 4 — Package split
- Carve the tree into `poly_trans/` (core) and `poly_ren/` (plugin) with `pyproject.toml` each.
- Register `poly_ren` as a `poly_trans.adapters` entry point.
- Move Ren'Py SDK download out of `setup.py` into `poly_ren/sdk.py`.
- Actually `pip install -e` both into the venv (the step the old `packages/` attempt skipped).
- Test: `pip install poly_trans` alone → standalone works, `renpy` adapter absent; `+ poly_ren` → Ren'Py works.

### Stage 5 — Collapse repos
- `translate-local` becomes "core installed without the plugin." Retire the second repo (or keep it as a thin install profile / launcher set).
- Rewrite the README(s) — see Section 5; they are currently badly out of date.
- Test: both install profiles run end-to-end from a clean checkout.

---

## 5. README debt (must fix during Stage 5)

Both repos' READMEs are **byte-identical and stale**, and `translate-local`'s isn't even adapted to standalone (still titled "Ren'Py"). Known wrong items:

- `5-3-translate.ps1`, `translate_with_aya23.py`, `translate_with_madlad.py`, `merger.py`, `core.py`, `batch_translator.py` — none exist.
- Config files described as `.json`; all are `.yaml` (benchmark, glossary, corrections, tags, characters, tools_config).
- `2-test.ps1` is actually `7-test.ps1`; `9-benchmark.ps1` unlisted.
- References `PIPELINE_USAGE.md`, `MODULARISATION_PLAN.md`, `IMPLEMENTATION_SUMMARY.md` — none exist.
- Prompts: primary templates are the `_uncensored` variants in `data/prompts/`; README omits them.
- Quality metrics: now COMET / METEOR / chrF (setup prompts for them), not just BLEU.
- File Structure block is largely fiction — rewrite from the actual tree.

After the split, the core README should describe the standalone/passthrough flow, and the Ren'Py plugin should carry its own README for the `.rpy` extract/merge stages.

---

## 6. Risks / open questions

- **`.tags.yaml` generality:** confirm the intermediate format carries nothing Ren'Py-specific in its schema (it currently holds Ren'Py tag positions). If tag metadata is Ren'Py-shaped, the passthrough adapter must emit a trivial/empty `.tags.yaml` and the core must tolerate that.
- **Config schema:** `current_config.yaml` mixes engine config (model/language) with game/source-path config (Ren'Py-ish). The plugin-specific keys should move under an adapter namespace.
- **`correct.py` language detection:** depends on path conventions (`game/tl/<lang>`); the standalone adapter needs its own language-resolution path or an explicit `--language`.
- **Distribution:** decide whether "standalone" ships as a pip package, a repo, or just an install profile of the unified repo.
