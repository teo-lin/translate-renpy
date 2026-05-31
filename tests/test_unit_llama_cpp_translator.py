"""
Unit tests for LlamaCppTranslator and Aya23Translator.
llama_cpp is mocked in sys.modules so no real GGUF model is required.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Mock llama_cpp before any translator imports so the DLL load never runs.
_mock_llama_module = MagicMock()
_mock_llama_instance = MagicMock()
_mock_llama_module.Llama.return_value = _mock_llama_instance
sys.modules["llama_cpp"] = _mock_llama_module

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translators.llama_cpp_translator import LlamaCppTranslator
from translators.aya23_translator import Aya23Translator, _get_default_model_path
import translators.aya23_translator as _aya_mod


def _reset_llama_mock():
    _mock_llama_module.reset_mock()
    _mock_llama_instance.reset_mock()
    _mock_llama_module.Llama.return_value = _mock_llama_instance


@pytest.fixture(autouse=True)
def reset_mocks(monkeypatch):
    _reset_llama_mock()
    # Aya23Translator() with no path resolves the GGUF via hf_hub_download —
    # stub it so tests don't hit the network or HF cache.
    monkeypatch.setattr(
        _aya_mod, "hf_hub_download",
        lambda repo_id, filename, **kw: f"/fake/cache/{filename}",
    )


@pytest.fixture
def translator():
    return LlamaCppTranslator(
        model_path="/fake/model.gguf",
        target_language="Romanian",
        n_gpu_layers=-1,
        n_ctx=8192,
        n_batch=256,
    )


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestLlamaCppTranslatorInit:
    def test_llama_constructed_with_correct_params(self):
        LlamaCppTranslator(
            model_path="/path/model.gguf",
            n_gpu_layers=20,
            n_ctx=4096,
            n_batch=128,
        )
        _mock_llama_module.Llama.assert_called_once_with(
            model_path="/path/model.gguf",
            n_gpu_layers=20,
            n_ctx=4096,
            n_batch=128,
            verbose=False,
        )

    def test_target_language_property(self, translator):
        assert translator.target_language == "Romanian"

    def test_glossary_defaults_to_empty_dict(self, translator):
        assert translator.glossary == {}

    def test_custom_glossary_stored(self):
        t = LlamaCppTranslator("/fake.gguf", glossary={"hello": "salut"})
        assert t.glossary == {"hello": "salut"}

    def test_none_glossary_becomes_empty_dict(self):
        t = LlamaCppTranslator("/fake.gguf", glossary=None)
        assert t.glossary == {}

    def test_prompt_template_stored(self):
        tmpl = "Translate {text} to {target_language}.{glossary_instructions}{context_section}{speaker_hint}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl)
        assert t.prompt_template == tmpl


# ── _build_translation_prompt ─────────────────────────────────────────────────

class TestBuildTranslationPrompt:
    def test_custom_template_used_when_set(self):
        tmpl = "Lang:{target_language} Text:{text}{glossary_instructions}{context_section}{speaker_hint}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl, target_language="Spanish")
        result = t._build_translation_prompt("Hello")
        assert "Spanish" in result
        assert "Hello" in result

    def test_glossary_term_included_when_present_in_text(self):
        tmpl = "Translate {text}.{glossary_instructions}{context_section}{speaker_hint}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl, glossary={"hello": "salut"})
        result = t._build_translation_prompt("hello world")
        assert "salut" in result

    def test_glossary_term_skipped_when_absent_from_text(self):
        tmpl = "Translate {text}.{glossary_instructions}{context_section}{speaker_hint}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl, glossary={"goodbye": "la revedere"})
        result = t._build_translation_prompt("hello world")
        assert "la revedere" not in result

    def test_comment_keys_excluded_from_glossary(self):
        tmpl = "Translate {text}.{glossary_instructions}{context_section}{speaker_hint}"
        t = LlamaCppTranslator(
            "/fake.gguf",
            prompt_template=tmpl,
            glossary={"_comment": "ignore this", "hi": "buna"},
        )
        result = t._build_translation_prompt("hi")
        assert "ignore this" not in result
        assert "buna" in result

    def test_context_included_in_template(self):
        tmpl = "Context:{context_section} Text:{text}{glossary_instructions}{speaker_hint}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl)
        result = t._build_translation_prompt("Hello", context=["A: Hi", "B: Hey"])
        assert "A: Hi" in result

    def test_only_last_4_context_lines_used(self):
        tmpl = "Context:{context_section} Text:{text}{glossary_instructions}{speaker_hint}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl)
        context = ["L1", "L2", "L3", "L4", "L5"]
        result = t._build_translation_prompt("Hello", context=context)
        assert "L1" not in result  # only last 4 lines kept
        assert "L2" in result

    def test_speaker_included_in_template(self):
        tmpl = "Speaker:{speaker_hint} Text:{text}{glossary_instructions}{context_section}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl)
        result = t._build_translation_prompt("Hello", speaker="Alice")
        assert "Alice" in result

    def test_no_speaker_produces_empty_hint(self):
        tmpl = "Speaker:[{speaker_hint}] Text:{text}{glossary_instructions}{context_section}"
        t = LlamaCppTranslator("/fake.gguf", prompt_template=tmpl)
        result = t._build_translation_prompt("Hello", speaker=None)
        assert "Speaker:[]" in result

    def test_fallback_prompt_when_import_fails(self, translator):
        with patch.dict(sys.modules, {"prompts": None}):
            result = translator._build_translation_prompt("Hello world")
        assert "Hello world" in result
        assert "Romanian" in result

    def test_fallback_prompt_contains_target_language_label(self, translator):
        with patch.dict(sys.modules, {"prompts": None}):
            result = translator._build_translation_prompt("Test")
        assert "Romanian:" in result


# ── _clean_translation ────────────────────────────────────────────────────────

class TestCleanTranslation:
    def test_strips_translation_prefix(self, translator):
        assert translator._clean_translation("Translation: Salut") == "Salut"

    def test_keeps_first_line_only(self, translator):
        assert translator._clean_translation("Salut\nExtra line") == "Salut"

    def test_strips_double_quotes(self, translator):
        assert translator._clean_translation('"Salut"') == "Salut"

    def test_strips_single_quotes(self, translator):
        assert translator._clean_translation("'Salut'") == "Salut"

    def test_strips_romanian_prefix(self, translator):
        assert translator._clean_translation("Romanian: Salut") == "Salut"

    def test_strips_french_prefix(self, translator):
        assert translator._clean_translation("French: Bonjour") == "Bonjour"

    def test_strips_german_prefix(self, translator):
        assert translator._clean_translation("German: Hallo") == "Hallo"

    def test_strips_spanish_prefix(self, translator):
        assert translator._clean_translation("Spanish: Hola") == "Hola"

    def test_unknown_language_prefix_kept(self, translator):
        assert translator._clean_translation("Klingon: qapla") == "Klingon: qapla"

    def test_clean_text_unchanged(self, translator):
        assert translator._clean_translation("Salut lume") == "Salut lume"

    def test_multiline_with_translation_prefix(self, translator):
        result = translator._clean_translation("Translation: Salut\nSecond line")
        assert result == "Salut"

    def test_whitespace_stripped(self, translator):
        assert translator._clean_translation("  Salut  ") == "Salut"


# ── translate ─────────────────────────────────────────────────────────────────

class TestTranslate:
    def _setup_llm_response(self, text):
        _mock_llama_instance.return_value = {"choices": [{"text": text}]}

    def test_translate_returns_cleaned_text(self, translator):
        self._setup_llm_response("Salut lume")
        assert translator.translate("Hello world") == "Salut lume"

    def test_translate_strips_language_prefix(self, translator):
        self._setup_llm_response("Romanian: Salut")
        assert translator.translate("Hello") == "Salut"

    def test_llm_called_with_stop_sequences(self, translator):
        self._setup_llm_response("ok")
        translator.translate("Hi")
        kwargs = _mock_llama_instance.call_args[1]
        assert "stop" in kwargs
        assert "English:" in kwargs["stop"]

    def test_llm_called_with_temperature(self, translator):
        self._setup_llm_response("ok")
        translator.translate("Hi", temperature=0.5)
        kwargs = _mock_llama_instance.call_args[1]
        assert kwargs["temperature"] == 0.5

    def test_llm_called_with_max_tokens(self, translator):
        self._setup_llm_response("ok")
        translator.translate("Hi", max_tokens=256)
        kwargs = _mock_llama_instance.call_args[1]
        assert kwargs["max_tokens"] == 256

    def test_translate_with_context_and_speaker_does_not_crash(self, translator):
        self._setup_llm_response("Salut")
        result = translator.translate("Hello", context=["A: Hi"], speaker="Alice")
        assert isinstance(result, str)

    def test_echo_is_false(self, translator):
        self._setup_llm_response("ok")
        translator.translate("Hi")
        kwargs = _mock_llama_instance.call_args[1]
        assert kwargs["echo"] is False


# ── Aya23Translator ────────────────────────────────────────────────────

class TestAya23Translator:
    def test_uses_default_model_path(self):
        Aya23Translator()
        called_path = _mock_llama_module.Llama.call_args[1]["model_path"]
        assert "aya-23-8B-Q4_K_M.gguf" in called_path

    def test_default_model_path_resolves(self):
        # Resolves the GGUF filename via hf_hub_download (mocked in reset_mocks)
        assert _get_default_model_path().endswith("aya-23-8B-Q4_K_M.gguf")

    def test_custom_model_path_overrides_default(self):
        Aya23Translator(model_path="/custom/model.gguf")
        called_path = _mock_llama_module.Llama.call_args[1]["model_path"]
        assert called_path == "/custom/model.gguf"

    def test_default_target_language_is_romanian(self):
        t = Aya23Translator()
        assert t.target_language == "Romanian"

    def test_custom_target_language(self):
        t = Aya23Translator(target_language="Spanish")
        assert t.target_language == "Spanish"

    def test_inherits_translate_method(self):
        t = Aya23Translator()
        assert hasattr(t, "translate")
        assert callable(t.translate)

    def test_inherits_clean_translation_method(self):
        t = Aya23Translator()
        assert hasattr(t, "_clean_translation")

    def test_default_n_ctx_is_8192(self):
        Aya23Translator()
        kwargs = _mock_llama_module.Llama.call_args[1]
        assert kwargs["n_ctx"] == 8192

    def test_default_n_batch_is_256(self):
        Aya23Translator()
        kwargs = _mock_llama_module.Llama.call_args[1]
        assert kwargs["n_batch"] == 256

    def test_default_n_gpu_layers_is_minus_one(self):
        Aya23Translator()
        kwargs = _mock_llama_module.Llama.call_args[1]
        assert kwargs["n_gpu_layers"] == -1

    def test_is_subclass_of_llama_cpp_translator(self):
        assert issubclass(Aya23Translator, LlamaCppTranslator)
