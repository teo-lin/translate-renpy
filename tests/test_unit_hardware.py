"""
Unit tests for src/hardware.py.
All file I/O is redirected to tmp_path via monkeypatch — no real YAML files needed.
GGUF resolution calls hf_hub_download, which is mocked so no network/cache is touched.
"""

import sys
import pytest
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hardware
from hardware import (
    _detect_tier,
    _resolve_gguf_path,
    resolve_model_path,
    is_model_available,
)


def _fake_hf_download(repo_id, filename, **kwargs):
    """Stand-in for hf_hub_download: returns a deterministic fake cache path."""
    return str(Path("/fake/cache") / repo_id.replace("/", "--") / filename)


# ── _detect_tier ──────────────────────────────────────────────────────────────

class TestDetectTier:
    def test_high_tier_16gb(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 16}}) == "high"

    def test_high_tier_boundary_10gb(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 10.0}}) == "high"

    def test_medium_tier_8gb(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 8}}) == "medium"

    def test_medium_tier_boundary_6gb(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 6.0}}) == "medium"

    def test_low_tier_4gb(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 4}}) == "low"

    def test_low_tier_boundary_01gb(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 0.1}}) == "low"

    def test_cpu_only_below_threshold(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 0.09}}) == "cpu_only"

    def test_cpu_only_zero_vram(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": 0}}) == "cpu_only"

    def test_cpu_only_missing_gpu_primary(self):
        assert _detect_tier({}) == "cpu_only"

    def test_cpu_only_none_vram(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": None}}) == "cpu_only"

    def test_cpu_only_missing_vram_key(self):
        assert _detect_tier({"gpu_primary": {}}) == "cpu_only"

    def test_string_vram_coerced(self):
        assert _detect_tier({"gpu_primary": {"vram_gb": "8"}}) == "medium"


# ── _resolve_gguf_path ──────────────────────────────────────────────────────────

_MODELS_CONFIG = {
    "available_models": {
        "euroLLM9b": {
            "repo": "org/EuroLLM-9B-GGUF",
            "format": "GGUF",
            "files": {
                "Q4_K_M": "EuroLLM-9B.Q4_K_M.gguf",
                "Q5_K_M": "EuroLLM-9B.Q5_K_M.gguf",
            },
        },
        "aya23": {
            "repo": "org/aya-23-8B-GGUF",
            "format": "GGUF",
            "file": "aya-23-8B-Q4_K_M.gguf",
        },
        "no_file_model": {
            "repo": "org/nfm-GGUF",
            "format": "GGUF",
        },
        "no_repo_model": {
            "format": "GGUF",
            "file": "x.gguf",
        },
    }
}


class TestResolveGgufPath:
    def test_multi_quant_q4(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        result = _resolve_gguf_path("euroLLM9b", "Q4_K_M", _MODELS_CONFIG)
        assert result.endswith("EuroLLM-9B.Q4_K_M.gguf")

    def test_multi_quant_q5(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        result = _resolve_gguf_path("euroLLM9b", "Q5_K_M", _MODELS_CONFIG)
        assert result.endswith("EuroLLM-9B.Q5_K_M.gguf")

    def test_multi_quant_missing_quant_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        assert _resolve_gguf_path("euroLLM9b", "Q3_K_M", _MODELS_CONFIG) is None

    def test_single_file_field(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        result = _resolve_gguf_path("aya23", "Q4_K_M", _MODELS_CONFIG)
        assert result.endswith("aya-23-8B-Q4_K_M.gguf")

    def test_no_file_field_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        assert _resolve_gguf_path("no_file_model", "Q4_K_M", _MODELS_CONFIG) is None

    def test_no_repo_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        assert _resolve_gguf_path("no_repo_model", "Q4_K_M", _MODELS_CONFIG) is None

    def test_unknown_model_key_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)
        assert _resolve_gguf_path("nonexistent", "Q4_K_M", _MODELS_CONFIG) is None

    def test_empty_available_models_returns_none(self):
        assert _resolve_gguf_path("euroLLM9b", "Q4_K_M", {}) is None

    def test_download_failure_returns_none(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("network down")
        monkeypatch.setattr(hardware, "hf_hub_download", _boom)
        assert _resolve_gguf_path("aya23", "Q4_K_M", _MODELS_CONFIG) is None


# ── detect_and_write_profile ──────────────────────────────────────────────────

_SYSTEM_YAML = {
    "gpu_primary": {"model": "RTX 3090", "vram_gb": 24, "cuda": True},
}

_PROFILES_YAML = {
    "profiles": {
        "high": {
            "euroLLM9b": {
                "n_gpu_layers": -1,
                "n_ctx": 32768,
                "n_batch": 512,
                "quant": "Q5_K_M",
            }
        }
    }
}

_MODELS_CONFIG_YAML = {
    "available_models": {
        "euroLLM9b": {
            "repo": "org/EuroLLM-9B-GGUF",
            "format": "GGUF",
            "files": {
                "Q5_K_M": "EuroLLM-9B.Q5_K_M.gguf",
            },
        }
    }
}


def _write_fixtures(tmp_path, system=None, profiles=None, models=None):
    system_file = tmp_path / "system.yaml"
    profiles_file = tmp_path / "profiles.yaml"
    models_file = tmp_path / "models_config.yaml"
    profile_out = tmp_path / "compute_profile.yaml"
    system_file.write_text(yaml.dump(system or _SYSTEM_YAML), encoding="utf-8")
    profiles_file.write_text(yaml.dump(profiles or _PROFILES_YAML), encoding="utf-8")
    models_file.write_text(yaml.dump(models or _MODELS_CONFIG_YAML), encoding="utf-8")
    return system_file, profiles_file, models_file, profile_out


def _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f):
    monkeypatch.setattr(hardware, "SYSTEM_FILE", sys_f)
    monkeypatch.setattr(hardware, "PROFILES_FILE", prof_f)
    monkeypatch.setattr(hardware, "MODELS_CONFIG_FILE", mod_f)
    monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)
    # GGUF resolution downloads via hf_hub_download — stub it out.
    monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)


class TestDetectAndWriteProfile:
    def test_creates_profile_file(self, tmp_path, monkeypatch):
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        hardware.detect_and_write_profile()

        assert out_f.exists()

    def test_returns_correct_tier(self, tmp_path, monkeypatch):
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        assert result["tier"] == "high"

    def test_returns_gpu_info(self, tmp_path, monkeypatch):
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        assert result["gpu"] == "RTX 3090"
        assert result["vram_gb"] == 24

    def test_model_params_resolved(self, tmp_path, monkeypatch):
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        m = result["models"]["euroLLM9b"]
        assert m["n_ctx"] == 32768
        assert m["n_batch"] == 512
        assert m["quant"] == "Q5_K_M"
        assert m["n_gpu_layers"] == -1
        assert "EuroLLM-9B.Q5_K_M.gguf" in m["file"]

    def test_hf_model_stores_repo(self, tmp_path, monkeypatch):
        profiles = {"profiles": {"high": {}}}
        models = {
            "available_models": {
                "nllb200": {
                    "repo": "facebook/nllb-200-distilled-600M",
                    "format": "safetensors",
                    "huggingface_download": True,
                }
            }
        }
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path, profiles=profiles, models=models)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        m = result["models"]["nllb200"]
        assert m["type"] == "hf"
        assert m["repo"] == "facebook/nllb-200-distilled-600M"

    def test_written_yaml_is_loadable(self, tmp_path, monkeypatch):
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        hardware.detect_and_write_profile()

        with open(out_f, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["tier"] == "high"

    def test_model_not_in_config_is_skipped(self, tmp_path, monkeypatch):
        profiles = {
            "profiles": {
                "high": {
                    "ghost_model": {
                        "n_gpu_layers": -1, "n_ctx": 8192,
                        "n_batch": 256, "quant": "Q4_K_M",
                    }
                }
            }
        }
        models = {"available_models": {}}
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path, profiles=profiles, models=models)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        assert result["models"] == {}

    def test_cpu_only_tier_no_vram(self, tmp_path, monkeypatch):
        system = {"gpu_primary": {"model": "iGPU", "vram_gb": 0}}
        profiles = {
            "profiles": {
                "cpu_only": {
                    "aya23": {
                        "n_gpu_layers": 0, "n_ctx": 4096,
                        "n_batch": 128, "quant": "Q4_K_M",
                    }
                }
            }
        }
        models = {
            "available_models": {
                "aya23": {
                    "repo": "org/aya-23-8B-GGUF",
                    "format": "GGUF",
                    "file": "aya-23-8B-Q4_K_M.gguf",
                }
            }
        }
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path, system=system, profiles=profiles, models=models)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        assert result["tier"] == "cpu_only"
        assert result["models"]["aya23"]["n_gpu_layers"] == 0

    def test_medium_tier_selection(self, tmp_path, monkeypatch):
        system = {"gpu_primary": {"model": "RTX 5070", "vram_gb": 8}}
        profiles = {
            "profiles": {
                "medium": {
                    "ayaExpanse8b": {
                        "n_gpu_layers": -1, "n_ctx": 16384,
                        "n_batch": 512, "quant": "Q5_K_M",
                    }
                }
            }
        }
        models = {
            "available_models": {
                "ayaExpanse8b": {
                    "repo": "org/aya-expanse-8b-GGUF",
                    "format": "GGUF",
                    "files": {"Q5_K_M": "aya-expanse-8b-Q5_K_M.gguf"},
                }
            }
        }
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path, system=system, profiles=profiles, models=models)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        assert result["tier"] == "medium"
        assert "ayaExpanse8b" in result["models"]

    def test_default_quant_q4_when_not_specified(self, tmp_path, monkeypatch):
        profiles = {
            "profiles": {
                "high": {
                    "euroLLM9b": {
                        "n_gpu_layers": -1, "n_ctx": 8192, "n_batch": 256
                        # quant key deliberately omitted — should default to Q4_K_M
                    }
                }
            }
        }
        models = {
            "available_models": {
                "euroLLM9b": {
                    "repo": "org/EuroLLM-9B-GGUF",
                    "format": "GGUF",
                    "files": {"Q4_K_M": "EuroLLM-9B.Q4_K_M.gguf"},
                }
            }
        }
        sys_f, prof_f, mod_f, out_f = _write_fixtures(tmp_path, profiles=profiles, models=models)
        _patch_hardware(monkeypatch, sys_f, prof_f, mod_f, out_f)

        result = hardware.detect_and_write_profile()

        assert result["models"]["euroLLM9b"]["quant"] == "Q4_K_M"


# ── resolve_model_path (public) ───────────────────────────────────────────────

class TestResolveModelPath:
    def test_gguf_from_profile(self, tmp_path, monkeypatch):
        profile = {"models": {"aya23": {"file": "/cache/aya.gguf", "quant": "Q4_K_M"}}}
        out_f = tmp_path / "compute_profile.yaml"
        out_f.write_text(yaml.dump(profile), encoding="utf-8")
        monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)

        assert resolve_model_path("aya23") == "/cache/aya.gguf"

    def test_hf_repo_from_profile(self, tmp_path, monkeypatch):
        profile = {"models": {"nllb200": {"repo": "facebook/nllb-200-distilled-600M", "type": "hf"}}}
        out_f = tmp_path / "compute_profile.yaml"
        out_f.write_text(yaml.dump(profile), encoding="utf-8")
        monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)

        assert resolve_model_path("nllb200") == "facebook/nllb-200-distilled-600M"

    def test_hf_fallback_to_models_config(self, tmp_path, monkeypatch):
        # No profile present -> fall back to models_config repo
        monkeypatch.setattr(hardware, "PROFILE_OUT", tmp_path / "missing.yaml")
        mod_f = tmp_path / "models_config.yaml"
        mod_f.write_text(yaml.dump({
            "available_models": {
                "nllb200": {"repo": "facebook/nllb-200-distilled-600M", "format": "safetensors"}
            }
        }), encoding="utf-8")
        monkeypatch.setattr(hardware, "MODELS_CONFIG_FILE", mod_f)

        assert resolve_model_path("nllb200") == "facebook/nllb-200-distilled-600M"

    def test_gguf_fallback_downloads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hardware, "PROFILE_OUT", tmp_path / "missing.yaml")
        mod_f = tmp_path / "models_config.yaml"
        mod_f.write_text(yaml.dump({
            "available_models": {
                "aya23": {"repo": "org/aya-23-8B-GGUF", "format": "GGUF",
                          "file": "aya-23-8B-Q4_K_M.gguf"}
            }
        }), encoding="utf-8")
        monkeypatch.setattr(hardware, "MODELS_CONFIG_FILE", mod_f)
        monkeypatch.setattr(hardware, "hf_hub_download", _fake_hf_download)

        assert resolve_model_path("aya23").endswith("aya-23-8B-Q4_K_M.gguf")

    def test_unknown_model_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hardware, "PROFILE_OUT", tmp_path / "missing.yaml")
        mod_f = tmp_path / "models_config.yaml"
        mod_f.write_text(yaml.dump({"available_models": {}}), encoding="utf-8")
        monkeypatch.setattr(hardware, "MODELS_CONFIG_FILE", mod_f)

        with pytest.raises(KeyError):
            resolve_model_path("nonexistent")


# ── is_model_available ────────────────────────────────────────────────────────

class TestIsModelAvailable:
    def test_gguf_file_present(self, tmp_path, monkeypatch):
        gguf = tmp_path / "model.gguf"
        gguf.write_text("x", encoding="utf-8")
        profile = {"models": {"aya23": {"file": str(gguf)}}}
        out_f = tmp_path / "compute_profile.yaml"
        out_f.write_text(yaml.dump(profile), encoding="utf-8")
        monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)

        assert is_model_available("aya23") is True

    def test_gguf_file_missing(self, tmp_path, monkeypatch):
        profile = {"models": {"aya23": {"file": str(tmp_path / "nope.gguf")}}}
        out_f = tmp_path / "compute_profile.yaml"
        out_f.write_text(yaml.dump(profile), encoding="utf-8")
        monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)

        assert is_model_available("aya23") is False

    def test_unknown_model_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hardware, "PROFILE_OUT", tmp_path / "missing.yaml")
        mod_f = tmp_path / "models_config.yaml"
        mod_f.write_text(yaml.dump({"available_models": {}}), encoding="utf-8")
        monkeypatch.setattr(hardware, "MODELS_CONFIG_FILE", mod_f)

        assert is_model_available("nonexistent") is False


# ── load_profile ──────────────────────────────────────────────────────────────

class TestLoadProfile:
    def test_loads_existing_profile(self, tmp_path, monkeypatch):
        profile_data = {
            "tier": "medium",
            "gpu": "RTX 5070",
            "vram_gb": 8,
            "models": {},
        }
        out_f = tmp_path / "compute_profile.yaml"
        out_f.write_text(yaml.dump(profile_data), encoding="utf-8")
        monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)

        result = hardware.load_profile()

        assert result["tier"] == "medium"
        assert result["gpu"] == "RTX 5070"
        assert result["vram_gb"] == 8

    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hardware, "PROFILE_OUT", tmp_path / "nonexistent.yaml")

        with pytest.raises(FileNotFoundError, match="0-setup.ps1"):
            hardware.load_profile()

    def test_error_message_contains_profile_path(self, tmp_path, monkeypatch):
        import re
        missing = tmp_path / "compute_profile.yaml"
        monkeypatch.setattr(hardware, "PROFILE_OUT", missing)

        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            hardware.load_profile()

    def test_returns_full_profile_dict(self, tmp_path, monkeypatch):
        profile_data = {
            "tier": "high",
            "gpu": "RTX 3090",
            "vram_gb": 24,
            "models": {
                "euroLLM9b": {
                    "file": "/cache/EuroLLM-9B.Q5_K_M.gguf",
                    "n_gpu_layers": -1,
                    "n_ctx": 32768,
                    "n_batch": 512,
                    "quant": "Q5_K_M",
                }
            },
        }
        out_f = tmp_path / "compute_profile.yaml"
        out_f.write_text(yaml.dump(profile_data), encoding="utf-8")
        monkeypatch.setattr(hardware, "PROFILE_OUT", out_f)

        result = hardware.load_profile()

        assert result["models"]["euroLLM9b"]["n_ctx"] == 32768
