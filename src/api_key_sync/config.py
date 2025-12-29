import re
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".dotfiles/.config/zsh/config.d/api_keys.zsh"

# Patterns to match API key names (underscore prefix for better specificity)
DEFAULT_PATTERNS = [
    "_TOKEN",
    "_API",
    "_KEY",
    "_PASSWORD",
    "_SECRET",
    "_CREDENTIAL",
]


def matches_key_pattern(
    name: str, patterns: list[str] | None = None, case_sensitive: bool = True
) -> bool:
    """Check if a key name matches any of the configured patterns."""
    patterns = patterns or DEFAULT_PATTERNS
    check_name = name if case_sensitive else name.upper()
    check_patterns = patterns if case_sensitive else [p.upper() for p in patterns]
    return any(pattern in check_name for pattern in check_patterns)


def filter_keys_by_pattern(
    keys: list[str], patterns: list[str] | None = None, case_sensitive: bool = True
) -> list[str]:
    """Filter a list of key names to only those matching configured patterns."""
    return [k for k in keys if matches_key_pattern(k, patterns, case_sensitive)]


def load_patterns(config_path: Path | None = None) -> list[str]:
    """Load custom patterns from config file, or return defaults."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return DEFAULT_PATTERNS

    content = path.read_text()
    match = re.search(r"API_KEY_PATTERNS=\(([^)]+)\)", content)
    if not match:
        return DEFAULT_PATTERNS

    patterns = match.group(1).split()
    return list(dict.fromkeys(patterns))  # Dedupe preserving order
