from pathlib import Path
from api_key_sync.config import (
    load_patterns,
    filter_keys_by_pattern,
    matches_key_pattern,
    DEFAULT_PATTERNS,
)


def test_matches_key_pattern_case_sensitive():
    assert matches_key_pattern("OPENAI_API_KEY", case_sensitive=True)
    assert matches_key_pattern("GITHUB_TOKEN", case_sensitive=True)
    assert matches_key_pattern("VAULT_PASSWORD", case_sensitive=True)
    assert matches_key_pattern("AWS_SECRET_ACCESS_KEY", case_sensitive=True)
    assert not matches_key_pattern("MY_RANDOM_VAR", case_sensitive=True)


def test_matches_key_pattern_case_insensitive():
    assert matches_key_pattern("openai_api_key", case_sensitive=False)
    assert matches_key_pattern("Github_Token", case_sensitive=False)
    assert not matches_key_pattern(
        "openai_api_key", case_sensitive=True
    )  # Lowercase fails


def test_filter_keys_by_pattern():
    keys = ["OPENAI_API_KEY", "GITHUB_TOKEN", "MY_VAR", "AWS_SECRET"]
    result = filter_keys_by_pattern(keys)
    assert set(result) == {"OPENAI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET"}


def test_filter_keys_by_pattern_case_insensitive():
    keys = ["openai_api_key", "github_token", "MY_VAR"]
    result = filter_keys_by_pattern(keys, case_sensitive=False)
    assert set(result) == {"openai_api_key", "github_token"}


def test_load_patterns_from_file(tmp_path: Path):
    config = tmp_path / "api_keys.zsh"
    config.write_text("API_KEY_PATTERNS=(_TOKEN _KEY)")

    result = load_patterns(config)

    assert result == ["_TOKEN", "_KEY"]


def test_load_patterns_dedupes(tmp_path: Path):
    config = tmp_path / "api_keys.zsh"
    config.write_text("API_KEY_PATTERNS=(_TOKEN _KEY _TOKEN)")

    result = load_patterns(config)

    assert result == ["_TOKEN", "_KEY"]


def test_load_patterns_missing_file():
    result = load_patterns(Path("/nonexistent/path"))
    assert result == DEFAULT_PATTERNS


def test_load_patterns_no_match(tmp_path: Path):
    config = tmp_path / "api_keys.zsh"
    config.write_text("# no API_KEY_PATTERNS here")

    result = load_patterns(config)

    assert result == DEFAULT_PATTERNS


def test_default_patterns_comprehensive():
    # Verify all expected patterns are in defaults
    assert "_TOKEN" in DEFAULT_PATTERNS
    assert "_API" in DEFAULT_PATTERNS
    assert "_KEY" in DEFAULT_PATTERNS
    assert "_PASSWORD" in DEFAULT_PATTERNS
    assert "_SECRET" in DEFAULT_PATTERNS
    assert "_CREDENTIAL" in DEFAULT_PATTERNS
