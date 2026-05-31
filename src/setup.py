import argparse
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Add scripts directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

try:
    import yaml
    from config_selector import select_multiple_items, select_languages_single_row
except ImportError as e:
    print(f"Error: A required library is missing: {e.name}")
    print(f"Please install it by running: pip install {e.name}")
    sys.exit(1)

# Constants
ROOT_DIR = Path(__file__).parent.parent
MODELS_CONFIG_PATH = ROOT_DIR / "models" / "models_config.yaml"
TOOLS_CONFIG_PATH = ROOT_DIR / "renpy" / "tools_config.yaml"
CURRENT_CONFIG_PATH = ROOT_DIR / "models" / "current_config.yaml"
VENV_PATH = ROOT_DIR / "venv"
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"


class ProjectSetup:
    """
    Main class to handle the setup process for the Ren'Py Translation System.
    """

    def __init__(self, args):
        self.args = args
        self.models_config = {}
        self.tools_config = {}
        self.selected_languages = []
        self.selected_models = []
        self.all_languages = []
        self.available_models = []
        self.venv_python = self._get_venv_python_path()
        self.tier = "low"
        self.wants_comet = False

    def run(self):
        """Main method to run the setup process in sequence."""
        self._set_hf_home()
        self._print_header()

        self._load_configs()
        print()
        print("=" * 70)
        print("Detecting Hardware Tier [1/8]")
        print("=" * 70)
        print()
        self.tier = self._detect_hardware_tier()
        print(f"  Tier: {self.tier}")

        # Build all_languages list (needed for later steps)
        self._build_all_languages_list()

        if not self.args.skip_model:
            # Normal flow: select languages then models
            self._select_languages()
            self._select_models()
            self._save_config()
        else:
            # Skip model flow: load from config or use all languages
            print()
            print("=" * 70)
            print("Skipping Model Selection [3/8]")
            print("=" * 70)
            print()
            if self.args.languages:
                # If languages specified via parameter, use them
                self._select_languages()
            self._load_installed_models_from_current_config()

        self._prompt_comet()

        if not self.args.skip_python:
            self._setup_python_env()
            self._setup_comet()
        else:
            print()
            print("=" * 70)
            print("Skipping Python Setup [4/8]")
            print("=" * 70)
            print()

        if not self.args.skip_model and self.selected_models:
            self._download_models(self.tier)
        else:
            print()
            print("=" * 70)
            print("Skipping Model Download [5/8]")
            print("=" * 70)
            print()

        if not self.args.skip_tools:
            self._download_tools()
        else:
            print()
            print("=" * 70)
            print("Skipping Tools Download [6/8]")
            print("=" * 70)
            print()

        all_good = self._verify_installation()
        self._detect_hardware()
        self._print_footer(all_good)

    def _set_hf_home(self):
        hf_home = ROOT_DIR / "models"
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        print(f"Hugging Face cache set to: {hf_home}")

        # HF_HOME is redirected to models/, but huggingface-cli login stores
        # the token at the default cache path. Forward it as HF_TOKEN so the
        # library sees it and rate-limit warnings disappear.
        if not os.environ.get("HF_TOKEN"):
            for candidate in [
                Path.home() / ".cache" / "huggingface" / "token",
                Path.home() / ".huggingface" / "token",
            ]:
                if candidate.exists():
                    token = candidate.read_text(encoding="utf-8").strip()
                    if token:
                        os.environ["HF_TOKEN"] = token
                    break

    def _print_header(self):
        print("=" * 70)
        print("  Ren'Py Translation System - Python Setup Script")
        print("=" * 70)
        print()

    def _load_configs(self):
        try:
            with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.models_config = yaml.safe_load(f)
            with open(TOOLS_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.tools_config = yaml.safe_load(f)
        except FileNotFoundError as e:
            print(f"Error: Configuration file not found at {e.filename}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)

    def _build_all_languages_list(self):
        """Build the complete list of all available languages."""
        languages_config = self.models_config.get("languages", [])

        # Fix for YAML parsing 'no' as False
        for lang_item in languages_config:
            if lang_item.get('code') is False:
                lang_item['code'] = 'no'

        # Sort languages by code, ensuring 'ro' is first if present
        sorted_languages = sorted(languages_config, key=lambda x: x['code'])
        ro_language = None
        for i, lang in enumerate(sorted_languages):
            if lang['code'] == 'ro':
                ro_language = sorted_languages.pop(i)
                break
        if ro_language:
            sorted_languages.insert(0, ro_language)

        self.all_languages = sorted_languages

    def _select_languages(self):
        if self.args.languages:
            print()
            print("=" * 70)
            print("Select Languages to Work With [2/8]")
            print("=" * 70)
            print()
            if self.args.languages.lower() == 'all':
                self.selected_languages = self.all_languages
                print("  Auto-selected: All languages")
            else:
                selected_codes = {code.strip().lower() for code in self.args.languages.split(',')}
                self.selected_languages = [lang for lang in self.all_languages if lang['code'] in selected_codes]
                print(f"  Auto-selected: {[lang['name'] for lang in self.selected_languages]}")
            return

        # Interactive selection
        def lang_formatter_func(lang, num):
            return f"[{num:2d}] {lang['name']}"

        self.selected_languages = select_languages_single_row(
            "Select Languages to Work With", self.all_languages, lang_formatter_func, "language", step_info="[2/8]"
        )

    def _select_models(self):
        all_models_dict = self.models_config.get("available_models", {})

        _tier_rank = {'cpu_only': 0, 'low': 1, 'medium': 2, 'high': 3}
        current_rank = _tier_rank.get(self.tier, 1)

        selected_lang_codes = {lang['code'] for lang in self.selected_languages}
        self.available_models = []
        for key, value in all_models_dict.items():
            model_langs = set(value.get("languages", []))
            if not model_langs.intersection(selected_lang_codes):
                continue
            min_tier = value.get('min_tier', 'cpu_only')
            if _tier_rank.get(min_tier, 0) > current_rank:
                continue
            value['key'] = key
            self.available_models.append(value)

        if not self.available_models:
            print("\nError: No available models support the selected languages.")
            sys.exit(1)

        if self.args.models:
            print()
            print("=" * 70)
            print("Select Translation Models to Install [3/8]")
            print("=" * 70)
            print()
            if self.args.models.lower() == 'all':
                self.selected_models = self.available_models
                print("  Auto-selected: All available models")
            else:
                try:
                    model_indices = [int(i.strip()) - 1 for i in self.args.models.split(',')]
                    self.selected_models = [self.available_models[i] for i in model_indices if 0 <= i < len(self.available_models)]
                    print(f"  Auto-selected: {[m['name'] for m in self.selected_models]}")
                except ValueError:
                    print("Invalid --models argument. Please provide comma-separated numbers.")
                    sys.exit(1)
            return

        # Interactive selection
        def model_formatter_func(model, num):
            import re
            supported_count = len(set(model['languages']).intersection(selected_lang_codes))
            files = model.get('files', {})
            if files:
                quant = self._quant_for_tier(model, self.tier)
                size = model.get(f'size_{quant}') or '?'
                quant_label = f"{quant} Â· {size}"
            else:
                filename = model.get('file') or model.get('destination') or ''
                m = re.search(r'Q\d+_K_[MS]', filename)
                quant = m.group(0) if m else None
                size = model.get('size') or '?'
                quant_label = f"{quant} Â· {size}" if quant else size
            return (
                f"  [{num:2d}] {model['name']}  [{quant_label}]\n"
                f"      - Supports {supported_count}/{len(self.selected_languages)} of your languages\n"
                f"      - {model.get('description', 'No description available.')}"
            )

        self.selected_models = select_multiple_items(
            "Select Translation Models to Install", self.available_models, model_formatter_func, "model", step_info="[3/8]"
        )

    def _save_config(self):
        """Saves the selected languages and models to current_config.yaml."""
        if not self.args.skip_model:
            current_config = {
                "installed_languages": self.selected_languages,
                "installed_models": [model['key'] for model in self.selected_models]
            }
            
            with open(CURRENT_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
            
            print(f"\n  Configuration saved to: {CURRENT_CONFIG_PATH}")

    def _load_installed_models_from_current_config(self):
        if not CURRENT_CONFIG_PATH.exists():
            print(f"Warning: {CURRENT_CONFIG_PATH} not found. Cannot determine installed models.")
            # Use all languages as default when skipping model selection
            self.selected_languages = self.all_languages
            return

        with open(CURRENT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)

        # Load installed languages
        installed_languages = current_config.get("installed_languages", [])
        if installed_languages:
            self.selected_languages = installed_languages
            print(f"  Loaded configuration from: {CURRENT_CONFIG_PATH}")
            print(f"    - Languages: {len(self.selected_languages)} configured")
        else:
            # Fallback to all languages if not found in config
            self.selected_languages = self.all_languages
            print(f"    - Languages: Using all {len(self.all_languages)} supported languages")

        # Load installed models
        installed_keys = current_config.get("installed_models", [])
        all_models_dict = self.models_config.get("available_models", {})
        self.selected_models = []
        for key in installed_keys:
            if key in all_models_dict:
                model_data = all_models_dict[key]
                model_data['key'] = key
                self.selected_models.append(model_data)

        print(f"    - Models: {len(self.selected_models)} installed")

    def _setup_python_env(self):
        print()
        print("=" * 70)
        print("Setting up Python Environment [4/8]")
        print("=" * 70)
        print()

        # Check Python version
        version_info = sys.version_info
        print(f"  Found: Python {version_info.major}.{version_info.minor}.{version_info.micro}")
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 10):
            print("  WARNING: Python 3.10+ recommended")

        # Create virtual environment
        if not VENV_PATH.exists():
            print("  Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True, capture_output=True)
        elif not self.venv_python.exists():
            print("  Virtual environment corrupted, recreating...")
            shutil.rmtree(VENV_PATH)
            subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True, capture_output=True)
        else:
            print("  Virtual environment already exists")

        # Upgrade pip
        print("  Checking pip version... (patience, this one takes about a minute)")
        self._run_venv_pip(["install", "--upgrade", "pip"], quiet=True)
        print("  pip up to date")

        # Install PyTorch with CUDA support
        torch_has_cuda = self._check_torch_cuda()
        if torch_has_cuda:
            print("  PyTorch already installed with CUDA support")
        else:
            if self._check_package_installed("torch"):
                print("  PyTorch found but without CUDA support, reinstalling with CUDA 12.4...")
            else:
                print("  Installing PyTorch with CUDA 12.4 (this may take a few minutes)...")
            self._run_venv_pip([
                "install", "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu124",
                "--force-reinstall"
            ], quiet=False)

        # Check which packages are needed based on selected models
        needs_llama_cpp = False
        needs_transformers = False
        needs_tiktoken = False

        _llama_cpp_models = {'aya23', 'orion-14b', 'ayaExpanse8b', 'euroLLM9b', 'euroLLM22b'}
        for model in self.selected_models:
            model_key = model.get('key', '')
            if model_key in _llama_cpp_models:
                needs_llama_cpp = True
            if model.get('huggingface_download'):
                needs_transformers = True
            if model.get('requires_tiktoken'):
                needs_tiktoken = True

        # Install llama-cpp-python if needed
        if needs_llama_cpp:
            # Use pip check instead of import check since import can fail due to CUDA issues
            if self._check_package_in_pip("llama-cpp-python"):
                print("  llama-cpp-python already installed")
            else:
                print("  Installing llama-cpp-python with CUDA (this may take a few minutes)...")
                print("  Using prebuilt CUDA 12.4 wheels from abetlen...")
                try:
                    self._run_venv_pip([
                        "install", "llama-cpp-python",
                        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124",
                        "--only-binary", ":all:"
                    ], quiet=False)
                    print("  llama-cpp-python installed successfully!")
                except subprocess.CalledProcessError:
                    print("  WARNING: CUDA wheel installation failed, trying CPU fallback...")
                    self._run_venv_pip([
                        "install", "llama-cpp-python",
                        "--only-binary", ":all:"
                    ], quiet=False)
        else:
            print("  Skipping llama-cpp-python (not needed for selected models)")

        # Install transformers if needed
        if needs_transformers:
            if self._check_package_installed("transformers"):
                print("  transformers already installed")
            else:
                print("  Installing transformers...")
                self._run_venv_pip(["install", "transformers", "sentencepiece"], quiet=True)
            if not self._check_package_installed("sentencepiece"):
                print("  Installing sentencepiece...")
                self._run_venv_pip(["install", "sentencepiece"], quiet=True)

            # Install tiktoken and protobuf if needed
            if needs_tiktoken:
                if not self._check_package_installed("tiktoken"):
                    print("  Installing tiktoken for SeamlessM4T...")
                    self._run_venv_pip(["install", "tiktoken"], quiet=True)
                if not self._check_package_installed("protobuf"):
                    print("  Installing protobuf for SeamlessM4T...")
                    self._run_venv_pip(["install", "protobuf"], quiet=True)
        else:
            print("  Skipping transformers (not needed for selected models)")

        # Install other requirements
        print("  Installing other dependencies from requirements.txt...")
        self._run_venv_pip(["install", "-r", str(REQUIREMENTS_PATH)], quiet=True)

        print("  Python environment setup complete!")

    def _prompt_comet(self):
        """Ask the user whether to download the COMET scoring model."""
        print()
        print("=" * 70)
        print("Optional: COMET Benchmark Scoring")
        print("=" * 70)
        print()
        print("  COMET is a neural MT metric that correlates better with human")
        print("  judgement than BLEU. Requires ~1.6 GB for the wmt22-comet-da model.")
        print()
        answer = input("  Download COMET model for benchmarking? (Y/N): ").strip().lower()
        self.wants_comet = answer in ('y', 'yes')
        if not self.wants_comet:
            print("  Skipping COMET. You can re-run setup later to add it.")

    def _setup_comet(self):
        """Install unbabel-comet and download the COMET model (only if user opted in)."""
        if not self.wants_comet:
            return

        if not self._check_package_in_pip("unbabel-comet"):
            print("  Installing unbabel-comet...")
            self._run_venv_pip(["install", "unbabel-comet"], quiet=False)
        else:
            print("  unbabel-comet already installed")

        print("  Downloading wmt22-comet-da model (~1.6 GB)...")
        py_code = (
            "from comet import download_model; "
            "path = download_model('Unbabel/wmt22-comet-da'); "
            "print('Model cached at:', path)"
        )
        try:
            subprocess.run([str(self.venv_python), "-c", py_code], check=True)
            print("  COMET model ready!")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: COMET model download failed: {e}")
            print("  Benchmarking will fall back to BLEU + chrF only.")

    def _get_venv_python_path(self):
        return VENV_PATH / ("Scripts" if platform.system() == "Windows" else "bin") / "python.exe"

    def _run_venv_pip(self, command, quiet=False):
        cmd = [str(self.venv_python), "-m", "pip"] + command
        subprocess.run(cmd, check=True, capture_output=quiet, text=True)

    def _get_torch_lib_path(self) -> str:
        return str(VENV_PATH / "Lib" / "site-packages" / "torch" / "lib")

    def _check_package_installed(self, package_name, extra_path: str = None):
        """Check if a Python package is installed in the virtual environment."""
        try:
            import os
            env = os.environ.copy()
            if extra_path:
                env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [str(self.venv_python), "-c", f"import {package_name}"],
                capture_output=True,
                text=True,
                env=env,
            )
            return result.returncode == 0
        except:
            return False

    def _check_package_in_pip(self, package_name):
        """Check if a package is listed in pip (doesn't verify it can be imported)."""
        try:
            result = subprocess.run(
                [str(self.venv_python), "-m", "pip", "show", package_name],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def _check_torch_cuda(self):
        """Check if PyTorch is installed with CUDA support."""
        if not self._check_package_installed("torch"):
            return False
        try:
            result = subprocess.run(
                [str(self.venv_python), "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0 and "True" in result.stdout
        except:
            return False

    def _quant_for_tier(self, model_config: dict, tier: str) -> str:
        """Return the quant name (e.g. 'Q5_K_M') for this model at the given tier."""
        try:
            import yaml as _yaml
            profiles_path = ROOT_DIR / "models" / "compute_profiles.yaml"
            with open(profiles_path, "r", encoding="utf-8") as f:
                profiles = _yaml.safe_load(f)
            model_key = model_config.get('key', '')
            quant = profiles.get("profiles", {}).get(tier, {}).get(model_key, {}).get("quant")
            if quant and quant in model_config.get('files', {}):
                return quant
        except Exception:
            pass
        files = model_config.get('files', {})
        return next((q for q in ('Q4_K_M', 'Q3_K_M') if q in files), next(iter(files), '?'))

    def _pick_quant_for_tier(self, model_config: dict, tier: str) -> str | None:
        """Return the filename to download for a multi-quant model given the current tier."""
        files = model_config.get('files', {})
        if not files:
            return None
        try:
            sys.path.insert(0, str(ROOT_DIR / "src"))
            import yaml as _yaml
            profiles_path = ROOT_DIR / "models" / "compute_profiles.yaml"
            with open(profiles_path, "r", encoding="utf-8") as f:
                profiles = _yaml.safe_load(f)
            model_key = model_config.get('key', '')
            quant = profiles.get("profiles", {}).get(tier, {}).get(model_key, {}).get("quant")
            if quant and quant in files:
                return files[quant]
        except Exception:
            pass
        # Fallback: smallest available quant
        return files.get('Q4_K_M') or files.get('Q3_K_M') or next(iter(files.values()))

    def _detect_hardware_tier(self) -> str:
        """Detect hardware tier without writing the full profile. Returns tier string."""
        try:
            sys.path.insert(0, str(ROOT_DIR / "src"))
            from hardware import _detect_tier
            import yaml as _yaml
            system_file = ROOT_DIR / "models" / "current_system.yaml"
            with open(system_file, "r", encoding="utf-8") as f:
                system = _yaml.safe_load(f)
            return _detect_tier(system)
        except Exception:
            return "low"  # safe default

    def _download_models(self, tier: str = "low"):
        from huggingface_hub import hf_hub_download
        print()
        print("=" * 70)
        print("Downloading Selected Translation Models [5/8]")
        print("=" * 70)
        print()
        for model in self.selected_models:
            model_config = model
            dest_path = ROOT_DIR / model_config['destination']

            if model_config.get('huggingface_download'):
                already_done = dest_path.is_dir() and any(
                    f for f in dest_path.iterdir() if f.suffix in ('.safetensors', '.bin', '.json')
                )
            elif dest_path.is_dir():
                already_done = any(dest_path.glob('*.gguf'))
            else:
                already_done = dest_path.exists()
            if already_done:
                print(f"    Model '{model['name']}' already downloaded.")
                continue

            print(f"    Downloading {model['name']}...")
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            repo_id = model_config['repo']
            if model_config.get('huggingface_download'):
                model_class = model_config.get('model_class', 'AutoModelForSeq2SeqLM')
                tokenizer_class = model_config.get('tokenizer_class', 'AutoTokenizer')
                py_code = (
                    f"from transformers import {model_class}, {tokenizer_class}; "
                    f"model = {model_class}.from_pretrained('{repo_id}'); "
                    f"tokenizer = {tokenizer_class}.from_pretrained('{repo_id}'); "
                    f"model.save_pretrained('{dest_path.as_posix()}'); "
                    f"tokenizer.save_pretrained('{dest_path.as_posix()}')"
                )
                subprocess.run([str(self.venv_python), "-c", py_code], check=True)
            else:
                if 'file' in model_config:
                    filename = model_config['file']
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_path.parent))
                    downloaded_file = dest_path.parent / filename
                    if downloaded_file.exists() and downloaded_file != dest_path:
                        downloaded_file.rename(dest_path)
                else:
                    filename = self._pick_quant_for_tier(model_config, tier)
                    print(f"      Quant: {filename} (tier={tier})")
                    dest_path.mkdir(parents=True, exist_ok=True)
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_path))

    def _download_tools(self):
        print()
        print("=" * 70)
        print("Downloading External Tools [6/8]")
        print("=" * 70)
        print()
        renpy_config = self.tools_config['tools']['renpy']
        renpy_path = ROOT_DIR / renpy_config['destination']

        if renpy_path.exists():
            print("  Ren'Py SDK already exists.")
        else:
            print(f"  Downloading Ren'Py SDK {renpy_config['version']}...")
            temp_zip = ROOT_DIR / "renpy.zip"
            try:
                import urllib.request
                with urllib.request.urlopen(renpy_config['url']) as response, open(temp_zip, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                with zipfile.ZipFile(temp_zip, 'r') as zf: zf.extractall(ROOT_DIR)
                extracted = next(ROOT_DIR.glob('renpy-*/'), None)
                if extracted: extracted.rename(renpy_path)
            except Exception as e:
                print(f"  WARNING: Could not download Ren'Py SDK: {e}")
            finally:
                if temp_zip.exists(): temp_zip.unlink()

        # Check for rpaExtract.exe
        rpa_extract_path = ROOT_DIR / "renpy" / "rpaExtract.exe"
        if rpa_extract_path.exists():
            print("  rpaExtract.exe already in repository")
        else:
            print("  WARNING: rpaExtract.exe not found at renpy/rpaExtract.exe")
            print("  rpaExtract is optional and only needed for extracting RPA archives")

        # Check for UnRen
        unren_path = ROOT_DIR / "renpy" / "unRen"
        if unren_path.exists():
            print("  UnRen already in repository")
        else:
            print("  WARNING: UnRen folder not found at renpy/unRen")

    def _verify_installation(self):
        print()
        print("=" * 70)
        print("Verifying Installation [7/8]")
        print("=" * 70)
        print()

        all_good = True

        # Check Python packages
        print("  Checking Python packages...")

        # Check PyTorch
        if self._check_package_installed("torch"):
            print("    - PyTorch: installed")
        else:
            print("    - PyTorch: NOT INSTALLED")
            all_good = False

        # Check model-specific packages
        _llama_cpp_models = {'aya23', 'orion-14b', 'ayaExpanse8b', 'euroLLM9b', 'euroLLM22b'}
        for model in self.selected_models:
            model_key = model.get('key', '')
            if model_key in _llama_cpp_models:
                # Use pip check instead of import check for llama-cpp-python
                # because import can fail due to CUDA/DLL issues even when installed
                if self._check_package_in_pip("llama-cpp-python"):
                    # Double-check if it can be imported (add torch lib path so CUDA DLLs are found)
                    if self._check_package_installed("llama_cpp", extra_path=self._get_torch_lib_path()):
                        print("    - llama-cpp-python: installed")
                    else:
                        print("    - llama-cpp-python: installed (warning: import test failed, may have runtime issues)")
                else:
                    print("    - llama-cpp-python: NOT INSTALLED")
                    all_good = False
                break  # Only check once

        for model in self.selected_models:
            model_key = model.get('key', '')
            if model.get('huggingface_download'):
                if self._check_package_installed("transformers"):
                    print("    - transformers: installed")
                else:
                    print("    - transformers: NOT INSTALLED")
                    all_good = False

                # Check tiktoken if needed
                if model.get('requires_tiktoken'):
                    if self._check_package_installed("tiktoken"):
                        print("    - tiktoken: installed")
                    else:
                        print("    - tiktoken: NOT INSTALLED")
                        all_good = False
                break  # Only check once

        # Check CUDA
        print("  Checking CUDA support...")
        if self._check_torch_cuda():
            print("    - CUDA available")
        else:
            print("    - WARNING: CUDA not available (will use CPU)")

        # Check selected models
        if not self.args.skip_model and self.selected_models:
            print("  Checking selected models...")
            for model in self.selected_models:
                model_config = model
                dest_path = ROOT_DIR / model_config['destination']

                if dest_path.exists():
                    if model_config.get('huggingface_download'):
                        # Directory-based model
                        if any(dest_path.iterdir()):
                            print(f"    - {model['name']}: downloaded")
                        else:
                            print(f"    - {model['name']}: NOT FOUND")
                            all_good = False
                    else:
                        # GGUF file model (file path or directory with gguf)
                        if dest_path.is_dir():
                            gguf_files = list(dest_path.glob('*.gguf'))
                            size_gb = sum(f.stat().st_size for f in gguf_files) / (1024**3)
                        else:
                            size_gb = dest_path.stat().st_size / (1024**3)
                        print(f"    - {model['name']}: {size_gb:.2f} GB")
                else:
                    print(f"    - {model['name']}: NOT FOUND")
                    all_good = False

        return all_good

    def _detect_hardware(self):
        print()
        print("=" * 70)
        print("Detecting Hardware & Writing Compute Profile [8/8]")
        print("=" * 70)
        print()

        try:
            sys.path.insert(0, str(ROOT_DIR / "src"))
            from hardware import detect_and_write_profile
            profile = detect_and_write_profile()
            print(f"  Tier  : {profile['tier']}")
            print(f"  GPU   : {profile['gpu']} ({profile['vram_gb']}GB VRAM)")
            print("  Models available in this tier:")
            for name, params in profile["models"].items():
                if params.get("type") == "hf":
                    print(f"    - {name}: HF safetensors")
                else:
                    print(f"    - {name}: n_ctx={params['n_ctx']}, quant={params['quant']}")
            print(f"\n  Profile written to: models/compute_profile.yaml")
        except Exception as e:
            print(f"  WARNING: Hardware detection failed: {e}")
            print("  Translation will still work via translate.py (existing flow)")

    def _print_footer(self, all_good):
        print("\n" + "=" * 70)
        if all_good:
            print("  SETUP COMPLETE!")
            print("\n  You're all set! Next steps:")
            print()
            print("  1. Copy your Ren'Py game to the games/ folder")
            print()
            print("  2. Translate your game using the interactive launcher:")
            print("     ./3-translate.ps1")
            print()
            print("  3. (Optional) Correct grammar with:")
            print("     ./4-correct.ps1")
            print()
            print(f"  The interactive scripts will use your configuration:")
            print(f"    - Languages: {len(self.selected_languages)} configured during setup")
            print(f"    - Models: {len(self.selected_models)} installed during setup")
            print(f"    - Games: auto-scanned from games/ folder")
            print()
            print("  For advanced usage, see README.md")
        else:
            print("  SETUP COMPLETED WITH WARNINGS")
            print("  Please review the messages above and install missing components")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Ren'Py Translation System - Setup Script")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--languages", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    
    args = parser.parse_args()

    try:
        ProjectSetup(args).run()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
