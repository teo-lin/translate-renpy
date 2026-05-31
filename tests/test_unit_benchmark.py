"""
Unit tests for scripts/benchmark.py — scoring helpers only, no model loading.
"""
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from benchmark import (
    tokenize,
    calculate_bleu,
    calculate_chrf,
    load_benchmark_data,
    load_glossary,
    detect_language_from_filename,
    detect_lang_code_from_filename,
    _lang_code_from_path,
    run_score_parsed,
)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_split(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_lowercases(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_renpy_color_tags(self):
        result = " ".join(tokenize("{color=#ff0000}Red text{/color}"))
        assert "{color=#ff0000}" not in result
        assert "red" in result

    def test_strips_renpy_variable_brackets(self):
        result = " ".join(tokenize("Hello [player_name]!"))
        assert "[player_name]" not in result

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_tags_returns_empty_or_no_tag_tokens(self):
        result = tokenize("{b}{/b}")
        assert all("{" not in t for t in result)


# ---------------------------------------------------------------------------
# calculate_bleu
# ---------------------------------------------------------------------------

_LONG_REF = "bună ziua dragă mea prietenă cum mai ești"   # 8 words → all 4-gram orders present


class TestCalculateBleu:
    def test_perfect_match(self):
        # Use a sentence long enough for all 4-gram orders so BLEU reaches 1.0
        assert calculate_bleu(_LONG_REF, _LONG_REF) == pytest.approx(1.0)

    def test_zero_overlap(self):
        score = calculate_bleu("bună ziua", "xyz abc def ghi")
        assert score == pytest.approx(0.0, abs=0.05)

    def test_partial_overlap_between_zero_and_one(self):
        score = calculate_bleu(_LONG_REF, _LONG_REF.split(" ", 3)[0])
        assert 0.0 < score < 1.0

    def test_multi_reference_perfect_on_second_ref(self):
        ref1 = "un text complet diferit de cel de mai jos"
        ref2 = _LONG_REF
        score = calculate_bleu([ref1, ref2], _LONG_REF)
        assert score == pytest.approx(1.0)

    def test_single_string_reference_accepted(self):
        # must not raise — function must handle plain string, not just list
        score = calculate_bleu(_LONG_REF, _LONG_REF)
        assert score == pytest.approx(1.0)

    def test_adding_alt_target_can_only_raise_score(self):
        score_single = calculate_bleu("bună ziua", "salut")
        score_multi = calculate_bleu(["bună ziua", "salut"], "salut")
        assert score_multi >= score_single


# ---------------------------------------------------------------------------
# calculate_chrf
# ---------------------------------------------------------------------------

class TestCalculateChrf:
    def test_perfect_match(self):
        assert calculate_chrf(_LONG_REF, _LONG_REF) == pytest.approx(1.0)

    def test_low_score_for_unrelated_text(self):
        # chrF is character-based so some incidental char overlap is expected
        score = calculate_chrf("bună ziua", "xyz abc def ghi")
        assert score < 0.20

    def test_partial_overlap_between_zero_and_one(self):
        score = calculate_chrf("bună ziua dragă", "bună ziua")
        assert 0.0 < score < 1.0

    def test_multi_reference_best_match(self):
        score = calculate_chrf(["un text complet diferit", _LONG_REF], _LONG_REF)
        assert score == pytest.approx(1.0)

    def test_single_string_reference_accepted(self):
        score = calculate_chrf(_LONG_REF, _LONG_REF)
        assert score == pytest.approx(1.0)

    def test_score_in_zero_one_range(self):
        for hyp in ["salut", "bună ziua dragă mea", _LONG_REF, "xyz"]:
            score = calculate_chrf("bună ziua dragă mea prietenă", hyp)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# load_benchmark_data
# ---------------------------------------------------------------------------

class TestLoadBenchmarkData:
    @staticmethod
    def _write(data, tmp_path):
        p = tmp_path / "bench.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        return p

    def test_valid_minimal(self, tmp_path):
        data = [{"source": "Hello", "target": "Bună"}]
        result = load_benchmark_data(self._write(data, tmp_path))
        assert len(result) == 1
        assert result[0]["source"] == "Hello"

    def test_valid_with_alt_targets(self, tmp_path):
        data = [{"source": "Hi", "target": "Salut", "alt_targets": ["Bună ziua"]}]
        result = load_benchmark_data(self._write(data, tmp_path))
        assert result[0]["alt_targets"] == ["Bună ziua"]

    def test_multiple_items(self, tmp_path):
        data = [
            {"source": "One", "target": "Unu"},
            {"source": "Two", "target": "Doi"},
        ]
        result = load_benchmark_data(self._write(data, tmp_path))
        assert len(result) == 2

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            load_benchmark_data(self._write([{"target": "Bună"}], tmp_path))

    def test_missing_target_raises(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            load_benchmark_data(self._write([{"source": "Hello"}], tmp_path))

    def test_bad_alt_targets_not_a_list_raises(self, tmp_path):
        data = [{"source": "Hi", "target": "Salut", "alt_targets": "not a list"}]
        with pytest.raises(ValueError):
            load_benchmark_data(self._write(data, tmp_path))

    def test_non_list_root_raises(self, tmp_path):
        p = tmp_path / "bench.yaml"
        p.write_text("source: Hello\ntarget: Bună\n", encoding="utf-8")
        with pytest.raises(ValueError, match="array"):
            load_benchmark_data(p)


# ---------------------------------------------------------------------------
# load_glossary
# ---------------------------------------------------------------------------

class TestLoadGlossary:
    def test_filters_underscore_keys(self, tmp_path):
        data = {"hello": "bună", "_comment": "ignored", "world": "lume"}
        p = tmp_path / "gloss.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        result = load_glossary(p)
        assert "_comment" not in result
        assert result["hello"] == "bună"
        assert result["world"] == "lume"

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert load_glossary(tmp_path / "nonexistent.yaml") == {}


# ---------------------------------------------------------------------------
# detect_language_from_filename
# ---------------------------------------------------------------------------

class TestDetectLanguageFromFilename:
    def test_ro_prefix(self):
        assert detect_language_from_filename("ro_benchmark.yaml") == "Romanian"

    def test_de_prefix(self):
        assert detect_language_from_filename("de_benchmark.yaml") == "German"

    def test_fr_prefix(self):
        assert detect_language_from_filename("fr_benchmark.yaml") == "French"

    def test_unknown_defaults_to_romanian(self):
        assert detect_language_from_filename("mystery.yaml") == "Romanian"


# ---------------------------------------------------------------------------
# detect_lang_code_from_filename
# ---------------------------------------------------------------------------

class TestDetectLangCodeFromFilename:
    def test_ro_prefix(self):
        assert detect_lang_code_from_filename("ro_benchmark.yaml") == "ro"

    def test_uncensored_variant(self):
        assert detect_lang_code_from_filename("ro_uncensored_benchmark.yaml") == "ro"

    def test_de_prefix(self):
        assert detect_lang_code_from_filename("de_benchmark.yaml") == "de"

    def test_unknown_defaults_to_ro(self):
        assert detect_lang_code_from_filename("mystery.yaml") == "ro"


# ---------------------------------------------------------------------------
# _lang_code_from_path
# ---------------------------------------------------------------------------

class TestLangCodeFromPath:
    def test_romanian_dir(self):
        p = Path("games/MyGame/game/tl/romanian/script.parsed.yaml")
        assert _lang_code_from_path(p) == "ro"

    def test_german_dir(self):
        p = Path("games/MyGame/game/tl/german/script.parsed.yaml")
        assert _lang_code_from_path(p) == "de"

    def test_spanish_dir(self):
        p = Path("games/MyGame/game/tl/spanish/script.parsed.yaml")
        assert _lang_code_from_path(p) == "es"

    def test_no_lang_segment_defaults_ro(self):
        p = Path("some/other/path/file.yaml")
        assert _lang_code_from_path(p) == "ro"


# ---------------------------------------------------------------------------
# run_score_parsed — no model loading, uses temp YAML files
# ---------------------------------------------------------------------------

class TestRunScoreParsed:
    @staticmethod
    def _setup(tmp_path, blocks, bench_items):
        (tmp_path / "models").mkdir()
        parsed = tmp_path / "test.parsed.yaml"
        bench = tmp_path / "ro_benchmark.yaml"
        with open(parsed, "w", encoding="utf-8") as f:
            yaml.dump(blocks, f, allow_unicode=True)
        with open(bench, "w", encoding="utf-8") as f:
            yaml.dump(bench_items, f, allow_unicode=True)
        return parsed, bench

    def _load_records(self, tmp_path):
        out = tmp_path / "models" / "benchmarks.yaml"
        if not out.exists():
            return []
        with open(out, encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    def test_scores_all_model_columns(self, tmp_path):
        blocks = {"line-1": {"en": "Hello", "ay": "Salut", "eu": "Bună"}}
        bench = [{"source": "Hello", "target": "Salut"}]
        parsed, bench_path = self._setup(tmp_path, blocks, bench)

        run_score_parsed(parsed, bench_path, tmp_path)

        records = self._load_records(tmp_path)
        model_keys = {r["model"] for r in records}
        assert "ay" in model_keys
        assert "eu" in model_keys

    def test_skips_unmatched_en_text(self, tmp_path):
        blocks = {"line-1": {"en": "Unmatched text", "ay": "Something"}}
        bench = [{"source": "Different text", "target": "Alta"}]
        parsed, bench_path = self._setup(tmp_path, blocks, bench)

        run_score_parsed(parsed, bench_path, tmp_path)

        records = self._load_records(tmp_path)
        assert records == []

    def test_perfect_bleu_for_exact_match(self, tmp_path):
        sentence = "bună ziua dragă mea prietenă cum mai ești"
        blocks = {"line-1": {"en": "Hello world", "ay": sentence}}
        bench = [{"source": "Hello world", "target": sentence}]
        parsed, bench_path = self._setup(tmp_path, blocks, bench)

        run_score_parsed(parsed, bench_path, tmp_path)

        records = self._load_records(tmp_path)
        ay = next(r for r in records if r["model"] == "ay")
        assert ay["avg_bleu"] == pytest.approx(1.0)

    def test_appends_to_existing_benchmarks_file(self, tmp_path):
        (tmp_path / "models").mkdir()
        existing = [{"model": "old", "avg_bleu": 0.5}]
        out = tmp_path / "models" / "benchmarks.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.dump(existing, f)

        blocks = {"line-1": {"en": "Hello", "ay": "Salut"}}
        bench = [{"source": "Hello", "target": "Salut"}]
        parsed = tmp_path / "test.parsed.yaml"
        bench_path = tmp_path / "ro_benchmark.yaml"
        with open(parsed, "w", encoding="utf-8") as f:
            yaml.dump(blocks, f, allow_unicode=True)
        with open(bench_path, "w", encoding="utf-8") as f:
            yaml.dump(bench, f, allow_unicode=True)

        run_score_parsed(parsed, bench_path, tmp_path)

        records = self._load_records(tmp_path)
        assert any(r.get("model") == "old" for r in records), "existing record lost"
        assert any(r.get("model") == "ay" for r in records), "new record missing"

    def test_ignores_blocks_without_en(self, tmp_path):
        blocks = {
            "line-1": {"en": "Hello", "ay": "Salut"},
            "separator-1": {"type": "separator"},
        }
        bench = [{"source": "Hello", "target": "Salut"}]
        parsed, bench_path = self._setup(tmp_path, blocks, bench)

        run_score_parsed(parsed, bench_path, tmp_path)

        records = self._load_records(tmp_path)
        ay = next(r for r in records if r["model"] == "ay")
        assert ay["total"] == 1
