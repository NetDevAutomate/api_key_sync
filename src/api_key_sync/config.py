import re
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".dotfiles/.config/zsh/config.d/api_keys.zsh"

DEFAULT_KEYS = [
    "OPENAI_API_KEY", "GITHUB_TOKEN", "VAULT_PASSWORD", "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY", "HF_TOKEN", "SRC_ACCESS_TOKEN", "CODEIUM_API_KEY",
    "GOOGLE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "ZAI_API_KEY", "KIMI_K2_API_KEY",
    "MOONSHOT_API_KEY", "MINIMAX_API_KEY", "SGAI_API_KEY", "PERPLEXITY_API_KEY"
]


def load_key_list(config_path: Path | None = None) -> list[str]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return DEFAULT_KEYS
    
    content = path.read_text()
    match = re.search(r'API_KEY_LIST=\(([^)]+)\)', content)
    if not match:
        return DEFAULT_KEYS
    
    keys = match.group(1).split()
    return list(dict.fromkeys(keys))  # Dedupe preserving order
