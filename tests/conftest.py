"""
Pytest configuration and shared fixtures for test suite

This file is automatically discovered by pytest and provides:
- Shared fixtures across all test files
- Custom pytest hooks and configuration
- Test environment setup
"""

import pytest
import sys
from pathlib import Path

# Tests print Romanian content (ă, ș, ț, ...). Force UTF-8 on the output
# streams so direct `pytest` runs don't crash on Windows' cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Add project root to Python path so tests can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


@pytest.fixture
def project_root():
    """Fixture providing the project root directory"""
    return PROJECT_ROOT


@pytest.fixture
def mock_models_config():
    """Fixture providing a mock models configuration"""
    return {
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
            }
        }
    }


@pytest.fixture
def mock_tools_config():
    """Fixture providing a mock tools configuration"""
    return {
        "tools": {
            "renpy": {
                "version": "8.1.3",
                "url": "https://www.renpy.org/dl/8.1.3/renpy-8.1.3-sdk.zip",
                "destination": "renpy"
            }
        }
    }


def pytest_configure(config):
    """Pytest configuration hook - runs before test collection"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    # Auto-add markers based on test file names
    for item in items:
        if "test_unit_" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_e2e_" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
