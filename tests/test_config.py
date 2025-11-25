import pytest
from pathlib import Path
from api_key_sync.config import load_key_list, DEFAULT_KEYS


def test_load_key_list_from_file(tmp_path: Path):
    config = tmp_path / "api_keys.zsh"
    config.write_text('API_KEY_LIST=(KEY_A KEY_B KEY_C)')
    
    result = load_key_list(config)
    
    assert result == ["KEY_A", "KEY_B", "KEY_C"]


def test_load_key_list_dedupes(tmp_path: Path):
    config = tmp_path / "api_keys.zsh"
    config.write_text('API_KEY_LIST=(KEY_A KEY_B KEY_A)')
    
    result = load_key_list(config)
    
    assert result == ["KEY_A", "KEY_B"]


def test_load_key_list_missing_file():
    result = load_key_list(Path("/nonexistent/path"))
    assert result == DEFAULT_KEYS


def test_load_key_list_no_match(tmp_path: Path):
    config = tmp_path / "api_keys.zsh"
    config.write_text('# no API_KEY_LIST here')
    
    result = load_key_list(config)
    
    assert result == DEFAULT_KEYS
