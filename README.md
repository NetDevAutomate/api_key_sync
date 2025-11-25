# API Key Sync

Sync API keys between 1Password and Apple Keychain.

## Prerequisites

- macOS (uses `security` CLI for Keychain access)
- [1Password CLI](https://developer.1password.com/docs/cli/) (`op`) installed and authenticated
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv tool install .
```

## Configuration

The tool reads API key names from `~/.dotfiles/.config/zsh/config.d/api_keys.zsh`:

```zsh
API_KEY_LIST=(OPENAI_API_KEY GITHUB_TOKEN ANTHROPIC_API_KEY ...)
```

Or use `--config` to specify a different file.

## Usage

### Sync keys between stores

```bash
# 1Password → Keychain (default)
api-key-sync sync op-to-keychain
api-key-sync sync op-to-keychain --dry-run  # Preview changes

# Keychain → 1Password
api-key-sync sync keychain-to-op

# Delete keys missing from source
api-key-sync sync op-to-keychain --sync-deletions
```

### Get a single key

```bash
api-key-sync get OPENAI_API_KEY --source keychain
api-key-sync get OPENAI_API_KEY --source op
```

### Store a key

```bash
api-key-sync put MY_KEY "secret_value" --target both
api-key-sync put MY_KEY "secret_value" --target keychain
api-key-sync put MY_KEY "secret_value" --target op
```

### List keys

```bash
api-key-sync list --source keychain
api-key-sync list --source op
```

### Export as environment variables

```bash
# One-time in current shell
eval "$(api-key-sync export-env)"

# Add to ~/.zshrc for automatic loading
source <(api-key-sync export-env)
```

## Options

| Option | Description |
|--------|-------------|
| `--vault` | 1Password vault name (default: `API_KEYS`) |
| `--service` | Keychain service name (default: `api-keys`) |
| `--config` | Path to config file with `API_KEY_LIST` |
| `--dry-run` | Preview changes without executing |
| `--sync-deletions` | Delete keys missing from source |

## Development

```bash
uv sync
uv run pytest
uv run pytest --cov=api_key_sync  # With coverage
```

## Architecture

```
src/api_key_sync/
├── models.py      # APIKey dataclass, KeyStore protocol
├── config.py      # Load key list from zsh config
├── sync.py        # SyncEngine with bidirectional sync
├── cli.py         # Typer CLI entry point
└── backends/
    ├── onepassword.py  # 1Password op CLI wrapper
    └── keychain.py     # macOS security CLI wrapper
```

## License

MIT
