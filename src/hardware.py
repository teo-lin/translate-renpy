"""
Hardware detection and compute profile resolution.
Called once during 0-setup.ps1 to write models/compute_profile.yaml.
At translation time, only load_profile() is used — no re-detection.
"""

from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).parent.parent
SYSTEM_FILE = ROOT_DIR / "models" / "current_system.yaml"
PROFILES_FILE = ROOT_DIR / "models" / "compute_profiles.yaml"
MODELS_CONFIG_FILE = ROOT_DIR / "models" / "models_config.yaml"
PROFILE_OUT = ROOT_DIR / "models" / "compute_profile.yaml"

# (min_vram_gb, tier_name) — checked top-down, first match wins
_TIER_THRESHOLDS = [
    (10.0, "high"),
    (6.0,  "medium"),
    (0.1,  "low"),
]


def _detect_tier(system: dict) -> str:
    gpu = system.get("gpu_primary", {})
    vram = float(gpu.get("vram_gb", 0) or 0)
    for min_vram, tier_name in _TIER_THRESHOLDS:
        if vram >= min_vram:
            return tier_name
    return "cpu_only"


def _resolve_gguf_path(model_key: str, quant: str, models_config: dict) -> str | None:
    """Resolve GGUF model to HF cache blob path using hf_hub_download."""
    model = models_config.get("available_models", {}).get(model_key)
    if not model:
        return None

    repo = model.get("repo")
    if not repo:
        return None

    files = model.get("files", {})
    if files:
        filename = files.get(quant)
    else:
        filename = model.get("file")

    if not filename:
        return None

    try:
        return hf_hub_download(repo_id=repo, filename=filename, local_files_only=False)
    except Exception as e:
        print(f"Failed to download {model_key} from {repo}: {e}")
        return None


def detect_and_write_profile() -> dict:
    """
    Read current_system.yaml, derive tier, resolve per-model params,
    write models/compute_profile.yaml. Returns the written profile dict.
    """
    with open(SYSTEM_FILE, "r", encoding="utf-8") as f:
        system = yaml.safe_load(f)

    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        profiles_cfg = yaml.safe_load(f)

    with open(MODELS_CONFIG_FILE, "r", encoding="utf-8") as f:
        models_config = yaml.safe_load(f)

    tier = _detect_tier(system)
    tier_params = profiles_cfg.get("profiles", {}).get(tier, {})

    resolved_models = {}
    for model_key, params in tier_params.items():
        if not isinstance(params, dict):
            continue
        quant = params.get("quant", "Q4_K_M")

        model_cfg = models_config.get("available_models", {}).get(model_key, {})
        if model_cfg.get("format") == "GGUF":
            file_path = _resolve_gguf_path(model_key, quant, models_config)
            if file_path:
                resolved_models[model_key] = {
                    "file": file_path,
                    "n_gpu_layers": params.get("n_gpu_layers", -1),
                    "n_ctx": params.get("n_ctx", 8192),
                    "n_batch": params.get("n_batch", 256),
                    "quant": quant,
                }

    for model_key, model_cfg in models_config.get("available_models", {}).items():
        if model_key in resolved_models:
            continue
        if model_cfg.get("huggingface_download"):
            repo = model_cfg.get("repo", "")
            resolved_models[model_key] = {
                "repo": repo,
                "type": "hf",
            }

    gpu = system.get("gpu_primary", {})
    profile = {
        "tier": tier,
        "gpu": gpu.get("model", "unknown"),
        "vram_gb": gpu.get("vram_gb", 0),
        "models": resolved_models,
    }

    with open(PROFILE_OUT, "w", encoding="utf-8") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)

    return profile


def load_profile() -> dict:
    """Load the resolved compute profile written by detect_and_write_profile()."""
    if not PROFILE_OUT.exists():
        raise FileNotFoundError(
            f"Compute profile not found at {PROFILE_OUT}. "
            "Please run 0-setup.ps1 first."
        )
    with open(PROFILE_OUT, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gguf_filename(model_cfg: dict) -> str | None:
    """The GGUF filename to use for a model (single-file or smallest available quant)."""
    files = model_cfg.get("files", {})
    return (
        model_cfg.get("file")
        or files.get("Q4_K_M")
        or files.get("Q3_K_M")
        or next(iter(files.values()), None)
    )


def resolve_model_path(model_name: str) -> str:
    """
    Resolve a model to the reference a translator needs:
      - GGUF models       -> local file path inside the HF cache (downloaded if missing)
      - HF safetensors    -> the repo id (from_pretrained resolves the cache itself)

    Prefers compute_profile.yaml (written at setup) and falls back to models_config.yaml.
    """
    if PROFILE_OUT.exists():
        with open(PROFILE_OUT, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}
        mp = profile.get("models", {}).get(model_name, {})
        if mp.get("file"):
            return mp["file"]
        if mp.get("repo"):
            return mp["repo"]

    with open(MODELS_CONFIG_FILE, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f).get("available_models", {}).get(model_name)
    if not model_cfg:
        raise KeyError(f"Model '{model_name}' not found in models_config.yaml")

    repo = model_cfg.get("repo")
    if model_cfg.get("format") == "GGUF":
        filename = _gguf_filename(model_cfg)
        if not filename:
            raise KeyError(f"No GGUF filename configured for '{model_name}'")
        return hf_hub_download(repo_id=repo, filename=filename)
    return repo


def is_model_available(model_name: str) -> bool:
    """True if the model is already present in the HF cache (never downloads)."""
    from huggingface_hub import try_to_load_from_cache

    def _repo_in_cache(repo: str) -> bool:
        # A repo downloaded via from_pretrained has only the files it needed
        # (not the full snapshot), so checking config.json presence matches what
        # the translators actually load — and is per-repo, never touching others.
        return isinstance(try_to_load_from_cache(repo, "config.json"), str)

    # Prefer the tier-resolved profile (records the exact GGUF quant/file).
    if PROFILE_OUT.exists():
        with open(PROFILE_OUT, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}
        mp = profile.get("models", {}).get(model_name, {})
        if mp.get("file"):
            return Path(mp["file"]).exists()
        if mp.get("repo"):
            return _repo_in_cache(mp["repo"])

    with open(MODELS_CONFIG_FILE, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f).get("available_models", {}).get(model_name)
    if not model_cfg:
        return False

    repo = model_cfg.get("repo")
    if model_cfg.get("format") == "GGUF":
        filename = _gguf_filename(model_cfg)
        if not filename:
            return False
        return try_to_load_from_cache(repo, filename) is not None
    return _repo_in_cache(repo)


if __name__ == "__main__":
    profile = detect_and_write_profile()
    print(f"Tier  : {profile['tier']}")
    print(f"GPU   : {profile['gpu']} ({profile['vram_gb']}GB)")
    print("Models available in this tier:")
    for name, params in profile["models"].items():
        if params.get("type") == "hf":
            print(f"  {name:20s} HF safetensors  repo={params.get('repo', '?')}")
        else:
            print(f"  {name:20s} GGUF  n_ctx={params['n_ctx']:6d}  quant={params['quant']}")
