"""
Unit tests for src/setup.py

Tests the ProjectSetup class and its methods:
- Configuration loading and language building
- Language and model selection logic
- Python environment setup (mocked)
- Model downloading (mocked)
- Tool downloading (mocked)
- Installation verification
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from argparse import Namespace

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import setup
from setup import ProjectSetup


# Test fixtures
MOCK_MODELS_CONFIG = {
    "languages": [
        {"name": "Romanian", "code": "ro"},
        {"name": "Spanish", "code": "es"},
        {"name": "French", "code": "fr"},
        {"name": "Norwegian", "code": False},  # YAML parses 'no' as False
    ],
    "available_models": {
        "aya23": {
            "name": "Aya-23-8B",
            "description": "Multilingual model",
            "size": "5.5 GB",
            "languages": ["ro", "es", "fr"],
            "repo": "CohereForAI/aya-23-8B",
            "file": "model.gguf",
            "format": "GGUF",
            "huggingface_download": False
        },
        "madlad-400-3b": {
            "name": "MADLAD-400-3B",
            "description": "Google translation model",
            "size": "6.0 GB",
            "languages": ["ro", "es", "fr", "no"],
            "repo": "google/madlad400-3b",
            "huggingface_download": True
        },
        "seamlessm4t-v2": {
            "name": "SeamlessM4T-v2",
            "description": "Meta multilingual model",
            "size": "9.0 GB",
            "languages": ["ro", "es"],
            "repo": "facebook/seamless-m4t-v2-large",
            "huggingface_download": True,
            "requires_tiktoken": True
        }
    }
}

MOCK_TOOLS_CONFIG = {
    "tools": {
        "renpy": {
            "version": "8.1.3",
            "url": "https://www.renpy.org/dl/8.1.3/renpy-8.1.3-sdk.zip",
            "destination": "renpy"
        }
    }
}


class TestProjectSetup:
    """Test suite for ProjectSetup class"""

    @staticmethod
    def create_mock_args(**kwargs):
        """Create mock arguments with defaults"""
        defaults = {
            'skip_python': False,
            'skip_tools': False,
            'skip_model': False,
            'languages': '',
            'models': ''
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    @staticmethod
    def test_init():
        """Test ProjectSetup initialization"""
        print("\n[TEST] ProjectSetup.__init__")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)

        assert ps.args == args, "Args should be stored"
        assert ps.models_config == {}, "Models config should be empty dict"
        assert ps.tools_config == {}, "Tools config should be empty dict"
        assert ps.selected_languages == [], "Selected languages should be empty list"
        assert ps.selected_models == [], "Selected models should be empty list"
        print("   [PASS] Initialization works correctly")


    @staticmethod
    def test_build_all_languages_list():
        """Test building the complete languages list"""
        print("\n[TEST] _build_all_languages_list")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)
        ps.models_config = MOCK_MODELS_CONFIG.copy()

        ps._build_all_languages_list()

        assert len(ps.all_languages) == 4, "Should have 4 languages"
        assert ps.all_languages[0]['code'] == 'ro', "Romanian should be first"
        assert ps.all_languages[1]['code'] == 'es', "Spanish should be second (alphabetical)"
        assert ps.all_languages[2]['code'] == 'fr', "French should be third"
        assert ps.all_languages[3]['code'] == 'no', "Norwegian 'no' should be fixed from False"
        print("   [PASS] Language list built correctly with 'ro' first and 'no' fixed")


    @staticmethod
    def test_select_languages_all():
        """Test selecting all languages via parameter"""
        print("\n[TEST] _select_languages with --languages all")

        args = TestProjectSetup.create_mock_args(languages='all')
        ps = ProjectSetup(args)
        ps.models_config = MOCK_MODELS_CONFIG.copy()
        ps._build_all_languages_list()

        ps._select_languages()

        assert len(ps.selected_languages) == 4, "Should select all 4 languages"
        print("   [PASS] --languages all selects all languages")


    @staticmethod
    def test_select_languages_specific():
        """Test selecting specific languages via parameter"""
        print("\n[TEST] _select_languages with --languages ro,fr")

        args = TestProjectSetup.create_mock_args(languages='ro,fr')
        ps = ProjectSetup(args)
        ps.models_config = MOCK_MODELS_CONFIG.copy()
        ps._build_all_languages_list()

        ps._select_languages()

        assert len(ps.selected_languages) == 2, "Should select 2 languages"
        codes = [lang['code'] for lang in ps.selected_languages]
        assert 'ro' in codes, "Should have Romanian"
        assert 'fr' in codes, "Should have French"
        print("   [PASS] --languages ro,fr selects Romanian and French")


    @staticmethod
    def test_select_models_filter_by_language():
        """Test that models are filtered by selected languages"""
        print("\n[TEST] _select_models - filtering by language")

        args = TestProjectSetup.create_mock_args(languages='ro', models='all')
        ps = ProjectSetup(args)
        ps.models_config = MOCK_MODELS_CONFIG.copy()
        ps._build_all_languages_list()
        ps._select_languages()

        ps._select_models()

        # All models support 'ro', so all should be available
        assert len(ps.available_models) == 3, "All 3 models support Romanian"
        assert len(ps.selected_models) == 3, "All models selected with 'all'"
        print("   [PASS] Models filtered and selected correctly")


    @staticmethod
    def test_select_models_specific():
        """Test selecting specific models via parameter"""
        print("\n[TEST] _select_models with --models 1,2")

        args = TestProjectSetup.create_mock_args(languages='all', models='1,2')
        ps = ProjectSetup(args)
        ps.models_config = MOCK_MODELS_CONFIG.copy()
        ps._build_all_languages_list()
        ps._select_languages()

        ps._select_models()

        assert len(ps.selected_models) == 2, "Should select 2 models"
        print("   [PASS] --models 1,2 selects first two models")


    @staticmethod
    @patch('setup.yaml')
    def test_save_config(mock_yaml):
        """Test saving configuration to YAML"""
        print("\n[TEST] _save_config")

        # Create a temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the CURRENT_CONFIG_PATH
            with patch('setup.CURRENT_CONFIG_PATH', Path(tmpdir) / 'current_config.yaml'):
                args = TestProjectSetup.create_mock_args()
                ps = ProjectSetup(args)
                ps.selected_languages = [{'name': 'Romanian', 'code': 'ro'}]
                ps.selected_models = [{'key': 'aya23', 'name': 'Aya-23-8B'}]

                # Mock yaml.dump
                mock_yaml.dump = Mock()

                ps._save_config()

                # Verify yaml.dump was called
                assert mock_yaml.dump.called, "yaml.dump should be called"
                call_args = mock_yaml.dump.call_args
                config_data = call_args[0][0]

                assert 'installed_languages' in config_data, "Should save languages"
                assert 'installed_models' in config_data, "Should save models"
                assert config_data['installed_models'] == ['aya23'], "Should save model keys"

        print("   [PASS] Configuration saved correctly")


    @staticmethod
    def test_load_installed_models_from_config():
        """Test loading installed models from current_config.yaml"""
        print("\n[TEST] _load_installed_models_from_current_config")

        # Create a temporary config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'current_config.yaml'
            mock_config = {
                'installed_languages': [{'name': 'Romanian', 'code': 'ro'}],
                'installed_models': ['aya23', 'madlad-400-3b']
            }

            # Write mock config
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(mock_config, f)

            # Mock the CURRENT_CONFIG_PATH
            with patch('setup.CURRENT_CONFIG_PATH', config_path):
                args = TestProjectSetup.create_mock_args(skip_model=True)
                ps = ProjectSetup(args)
                ps.models_config = MOCK_MODELS_CONFIG.copy()
                ps._build_all_languages_list()

                ps._load_installed_models_from_current_config()

                assert len(ps.selected_languages) == 1, "Should load 1 language"
                assert ps.selected_languages[0]['code'] == 'ro', "Should load Romanian"
                assert len(ps.selected_models) == 2, "Should load 2 models"

        print("   [PASS] Loaded config correctly")


    @staticmethod
    def test_load_installed_models_no_config():
        """Test fallback when config file doesn't exist"""
        print("\n[TEST] _load_installed_models_from_current_config (no file)")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'nonexistent.yaml'

            with patch('setup.CURRENT_CONFIG_PATH', config_path):
                args = TestProjectSetup.create_mock_args(skip_model=True)
                ps = ProjectSetup(args)
                ps.models_config = MOCK_MODELS_CONFIG.copy()
                ps._build_all_languages_list()

                ps._load_installed_models_from_current_config()

                # Should fallback to all languages
                assert len(ps.selected_languages) == 4, "Should use all languages as fallback"

        print("   [PASS] Fallback to all languages works")


    @staticmethod
    @patch('setup.subprocess.run')
    def test_check_package_installed(mock_subprocess):
        """Test _check_package_installed helper"""
        print("\n[TEST] _check_package_installed")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)

        # Mock successful package check
        mock_subprocess.return_value = Mock(returncode=0)
        assert ps._check_package_installed('torch') == True, "Should return True when installed"

        # Mock failed package check
        mock_subprocess.return_value = Mock(returncode=1)
        assert ps._check_package_installed('nonexistent') == False, "Should return False when not installed"

        print("   [PASS] Package check works correctly")


    @staticmethod
    @patch('setup.subprocess.run')
    def test_check_torch_cuda(mock_subprocess):
        """Test _check_torch_cuda helper"""
        print("\n[TEST] _check_torch_cuda")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)

        # Mock torch not installed
        mock_subprocess.return_value = Mock(returncode=1)
        assert ps._check_torch_cuda() == False, "Should return False when torch not installed"

        # Mock torch installed with CUDA
        mock_subprocess.return_value = Mock(returncode=0, stdout="True")
        assert ps._check_torch_cuda() == True, "Should return True when CUDA available"

        # Mock torch installed without CUDA
        mock_subprocess.return_value = Mock(returncode=0, stdout="False")
        assert ps._check_torch_cuda() == False, "Should return False when CUDA not available"

        print("   [PASS] CUDA check works correctly")


    @staticmethod
    @patch('huggingface_hub.try_to_load_from_cache')
    @patch('huggingface_hub.scan_cache_dir')
    @patch('setup.subprocess.run')
    @patch('setup.ProjectSetup._check_package_installed')
    @patch('setup.ProjectSetup._check_package_in_pip')
    @patch('setup.ProjectSetup._check_torch_cuda')
    def test_verify_installation(mock_cuda, mock_pip, mock_pkg, mock_subprocess, mock_scan, mock_cache):
        """Test installation verification (model present in HF cache)"""
        print("\n[TEST] _verify_installation")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)
        ps.selected_models = [
            {
                'key': 'aya23',
                'name': 'Aya-23-8B',
                'repo': 'CohereForAI/aya-23-8B-GGUF',
                'file': 'model.gguf',
                'huggingface_download': False
            }
        ]

        # Mock all checks pass
        mock_pkg.return_value = True
        mock_pip.return_value = True
        mock_cuda.return_value = True
        # GGUF file resolves in the HF cache
        mock_scan.return_value.repos = []
        mock_cache.return_value = '/fake/cache/model.gguf'

        result = ps._verify_installation()

        assert result == True, "Should return True when all checks pass"
        print("   [PASS] Verification passes when all components installed")


    @staticmethod
    @patch('huggingface_hub.try_to_load_from_cache')
    @patch('huggingface_hub.scan_cache_dir')
    @patch('setup.subprocess.run')
    @patch('setup.ProjectSetup._check_package_installed')
    @patch('setup.ProjectSetup._check_package_in_pip')
    @patch('setup.ProjectSetup._check_torch_cuda')
    def test_verify_installation_missing_model(mock_cuda, mock_pip, mock_pkg, mock_subprocess, mock_scan, mock_cache):
        """Test verification fails when model is not in the HF cache"""
        print("\n[TEST] _verify_installation - missing model")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)
        ps.selected_models = [
            {
                'key': 'aya23',
                'name': 'Aya-23-8B',
                'repo': 'CohereForAI/aya-23-8B-GGUF',
                'file': 'model.gguf',
                'huggingface_download': False
            }
        ]

        mock_pkg.return_value = True
        mock_pip.return_value = True
        mock_cuda.return_value = True
        # GGUF file NOT in the cache
        mock_scan.return_value.repos = []
        mock_cache.return_value = None

        result = ps._verify_installation()

        assert result == False, "Should return False when model missing"
        print("   [PASS] Verification fails when model missing")


    @staticmethod
    def test_print_footer_success():
        """Test footer printing for successful setup"""
        print("\n[TEST] _print_footer - success")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)
        ps.selected_languages = [{'name': 'Romanian', 'code': 'ro'}]
        ps.selected_models = [{'key': 'aya23', 'name': 'Aya-23-8B'}]

        import io
        captured_output = io.StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = captured_output
            ps._print_footer(all_good=True)
        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        assert "SETUP COMPLETE!" in output, "Should show success message"
        assert "1 configured during setup" in output, "Should show language count"
        assert "1 installed during setup" in output, "Should show model count"
        print("   [PASS] Success footer printed correctly")


    @staticmethod
    def test_print_footer_warnings():
        """Test footer printing with warnings"""
        print("\n[TEST] _print_footer - warnings")

        args = TestProjectSetup.create_mock_args()
        ps = ProjectSetup(args)
        ps.selected_languages = []
        ps.selected_models = []

        import io
        captured_output = io.StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = captured_output
            ps._print_footer(all_good=False)
        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        assert "COMPLETED WITH WARNINGS" in output, "Should show warning message"
        print("   [PASS] Warning footer printed correctly")



def run_all_tests():
    """Run all setup.py tests"""
    print("=" * 70)
    print("UNIT TESTS: setup.py")
    print("=" * 70)

    tests = [
        ("Initialization", TestProjectSetup.test_init),
        ("Build languages list", TestProjectSetup.test_build_all_languages_list),
        ("Select all languages", TestProjectSetup.test_select_languages_all),
        ("Select specific languages", TestProjectSetup.test_select_languages_specific),
        ("Filter models by language", TestProjectSetup.test_select_models_filter_by_language),
        ("Select specific models", TestProjectSetup.test_select_models_specific),
        ("Save configuration", TestProjectSetup.test_save_config),
        ("Load installed models", TestProjectSetup.test_load_installed_models_from_config),
        ("Load models - no config", TestProjectSetup.test_load_installed_models_no_config),
        ("Check package installed", TestProjectSetup.test_check_package_installed),
        ("Check torch CUDA", TestProjectSetup.test_check_torch_cuda),
        ("Verify installation - success", TestProjectSetup.test_verify_installation),
        ("Verify installation - missing model", TestProjectSetup.test_verify_installation_missing_model),
        ("Print footer - success", TestProjectSetup.test_print_footer_success),
        ("Print footer - warnings", TestProjectSetup.test_print_footer_warnings),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"   [FAIL] {test_name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests: {passed}/{passed + failed} passed")
    print()

    if failed == 0:
        print("\033[92m[SUCCESS] All setup.py tests passed!\033[0m")
        return True
    else:
        print(f"\033[91m[FAILURE] {failed} test(s) failed.\033[0m")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
