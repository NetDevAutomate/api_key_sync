"""Chezmoi age-encrypted secrets backend."""

import json
import subprocess
from pathlib import Path
from typing import Literal

from ..models import APIKey

NameStyle = Literal["upper", "lower", "preserve"]


class ChezmoiStore:
    """KeyStore implementation for chezmoi age-encrypted secrets.json.age files."""

    def __init__(
        self,
        secrets_file: str | Path | None = None,
        identity_file: str | Path | None = None,
        recipient: str | None = None,
        name_style: NameStyle = "preserve",
    ):
        """Initialize ChezmoiStore.

        Args:
            secrets_file: Path to secrets.json.age file. Defaults to
                ~/.local/share/chezmoi/secrets.json.age
            identity_file: Path to age identity file for decryption.
                Defaults to value from chezmoi config.
            recipient: Age recipient public key for encryption.
                Defaults to value from chezmoi config.
            name_style: How to transform key names:
                - "upper": Convert to UPPER_CASE (for 1Password compatibility)
                - "lower": Convert to lower_case
                - "preserve": Keep original case
        """
        self.secrets_file = Path(
            secrets_file or Path.home() / ".local/share/chezmoi/secrets.json.age"
        )
        self.name_style = name_style

        # Get identity and recipient from chezmoi config if not provided
        self.identity_file = Path(identity_file) if identity_file else None
        self.recipient = recipient

        if not self.identity_file or not self.recipient:
            self._load_chezmoi_config()

    def _load_chezmoi_config(self) -> None:
        """Load age configuration from chezmoi."""
        try:
            result = subprocess.run(
                [
                    "chezmoi",
                    "execute-template",
                    "{{ .chezmoi.config.age.identity }}|||{{ .chezmoi.config.age.recipient }}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = result.stdout.strip().split("|||")
            if len(parts) == 2:
                if not self.identity_file and parts[0]:
                    self.identity_file = Path(parts[0]).expanduser()
                if not self.recipient and parts[1]:
                    self.recipient = parts[1]
        except subprocess.CalledProcessError:
            pass

    def _normalize_name(self, name: str) -> str:
        """Normalize key name based on name_style setting."""
        if self.name_style == "upper":
            return name.upper()
        elif self.name_style == "lower":
            return name.lower()
        return name

    def _denormalize_name(self, name: str, existing_keys: dict[str, str]) -> str:
        """Find the actual key name in existing keys, or normalize for new keys."""
        # Try exact match first
        if name in existing_keys:
            return name
        # Try case-insensitive match
        name_lower = name.lower()
        for key in existing_keys:
            if key.lower() == name_lower:
                return key
        # Return normalized name for new keys
        return self._normalize_name(name)

    def _decrypt_secrets(self) -> dict[str, str]:
        """Decrypt and parse the secrets file."""
        if not self.secrets_file.exists():
            return {}

        try:
            result = subprocess.run(
                ["chezmoi", "decrypt", str(self.secrets_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return {}

    def _encrypt_secrets(self, secrets: dict[str, str]) -> bool:
        """Encrypt and write the secrets file."""
        if not self.identity_file or not self.recipient:
            return False

        try:
            json_data = json.dumps(secrets, indent=2)

            # Use age CLI to encrypt
            result = subprocess.run(
                ["age", "-r", self.recipient, "-o", str(self.secrets_file)],
                input=json_data,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def get(self, name: str) -> str | None:
        """Get a secret by name."""
        secrets = self._decrypt_secrets()
        # Try normalized name and original
        for key in [name, name.lower(), name.upper()]:
            if key in secrets:
                return secrets[key]
        return None

    def put(self, key: APIKey) -> bool:
        """Store or update a secret."""
        secrets = self._decrypt_secrets()
        actual_name = self._denormalize_name(key.name, secrets)
        secrets[actual_name] = key.value
        return self._encrypt_secrets(secrets)

    def delete(self, name: str) -> bool:
        """Delete a secret by name."""
        secrets = self._decrypt_secrets()
        # Find the actual key name
        actual_name = None
        for key in [name, name.lower(), name.upper()]:
            if key in secrets:
                actual_name = key
                break

        if actual_name is None:
            return False

        del secrets[actual_name]
        return self._encrypt_secrets(secrets)

    def list_all_keys(self) -> dict[str, str]:
        """List all secrets with normalized names."""
        secrets = self._decrypt_secrets()
        return {self._normalize_name(k): v for k, v in secrets.items()}

    def list_keys(self, key_names: list[str]) -> dict[str, str]:
        """List specific secrets by name."""
        all_keys = self.list_all_keys()
        result = {}
        for name in key_names:
            normalized = self._normalize_name(name)
            if normalized in all_keys:
                result[normalized] = all_keys[normalized]
        return result
