"""Tests for ChezmoiStore backend."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from api_key_sync.backends.chezmoi import ChezmoiStore
from api_key_sync.models import APIKey


class TestChezmoiStore:
    """Tests for ChezmoiStore backend."""

    @pytest.fixture
    def mock_secrets(self):
        """Sample secrets data."""
        return {
            "openai_api_key": "sk-test123",
            "github_token": "ghp_abc",
            "aws_secret_key": "secret",
        }

    @pytest.fixture
    def store(self, tmp_path):
        """Create a ChezmoiStore with mocked config."""
        secrets_file = tmp_path / "secrets.json.age"
        secrets_file.touch()  # Create file so exists() check passes
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=secrets_file,
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
            )
        return store

    def test_decrypt_secrets(self, store, mock_secrets):
        """Test decrypting secrets file."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = store._decrypt_secrets()

        assert result == mock_secrets
        mock_run.assert_called_once()

    def test_decrypt_secrets_file_not_exists(self, tmp_path):
        """Test decrypting when file doesn't exist."""
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=tmp_path / "nonexistent.age",
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
            )
        result = store._decrypt_secrets()
        assert result == {}

    def test_get_existing_key(self, store, mock_secrets):
        """Test getting an existing key."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            result = store.get("openai_api_key")

        assert result == "sk-test123"

    def test_get_missing_key(self, store, mock_secrets):
        """Test getting a non-existent key."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            result = store.get("nonexistent_key")

        assert result is None

    def test_get_case_insensitive(self, store, mock_secrets):
        """Test that get() handles case variations."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            # Should find lowercase key with uppercase query
            result = store.get("OPENAI_API_KEY")

        assert result == "sk-test123"

    def test_list_all_keys_preserve_style(self, tmp_path, mock_secrets):
        """Test listing keys with preserve name style."""
        secrets_file = tmp_path / "secrets.json.age"
        secrets_file.touch()
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=secrets_file,
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
                name_style="preserve",
            )

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            result = store.list_all_keys()

        assert result == mock_secrets

    def test_list_all_keys_upper_style(self, tmp_path, mock_secrets):
        """Test listing keys with upper name style."""
        secrets_file = tmp_path / "secrets.json.age"
        secrets_file.touch()
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=secrets_file,
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
                name_style="upper",
            )

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            result = store.list_all_keys()

        assert "OPENAI_API_KEY" in result
        assert "GITHUB_TOKEN" in result
        assert "AWS_SECRET_KEY" in result

    def test_list_all_keys_lower_style(self, tmp_path):
        """Test listing keys with lower name style."""
        secrets_file = tmp_path / "secrets.json.age"
        secrets_file.touch()
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=secrets_file,
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
                name_style="lower",
            )

        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"UPPER_KEY": "value"})

        with patch("subprocess.run", return_value=mock_result):
            result = store.list_all_keys()

        assert "upper_key" in result

    def test_put_new_key(self, store, mock_secrets, tmp_path):
        """Test adding a new key."""
        # Create the secrets file so it "exists"
        store.secrets_file = tmp_path / "secrets.json.age"
        store.secrets_file.touch()

        decrypt_result = MagicMock()
        decrypt_result.stdout = json.dumps(mock_secrets)

        encrypt_result = MagicMock()
        encrypt_result.returncode = 0

        with patch("subprocess.run", side_effect=[decrypt_result, encrypt_result]):
            result = store.put(APIKey("new_key", "new_value"))

        assert result is True

    def test_delete_existing_key(self, store, mock_secrets, tmp_path):
        """Test deleting an existing key."""
        store.secrets_file = tmp_path / "secrets.json.age"
        store.secrets_file.touch()

        decrypt_result = MagicMock()
        decrypt_result.stdout = json.dumps(mock_secrets)

        encrypt_result = MagicMock()
        encrypt_result.returncode = 0

        with patch("subprocess.run", side_effect=[decrypt_result, encrypt_result]):
            result = store.delete("openai_api_key")

        assert result is True

    def test_delete_nonexistent_key(self, store, mock_secrets, tmp_path):
        """Test deleting a key that doesn't exist."""
        store.secrets_file = tmp_path / "secrets.json.age"
        store.secrets_file.touch()

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            result = store.delete("nonexistent_key")

        assert result is False

    def test_list_keys_filtered(self, store, mock_secrets):
        """Test listing specific keys."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(mock_secrets)

        with patch("subprocess.run", return_value=mock_result):
            result = store.list_keys(["openai_api_key", "github_token"])

        assert len(result) == 2
        assert "openai_api_key" in result
        assert "github_token" in result

    def test_normalize_name_upper(self, tmp_path):
        """Test name normalization to uppercase."""
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=tmp_path / "test.age",
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
                name_style="upper",
            )

        assert store._normalize_name("test_key") == "TEST_KEY"

    def test_normalize_name_lower(self, tmp_path):
        """Test name normalization to lowercase."""
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=tmp_path / "test.age",
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
                name_style="lower",
            )

        assert store._normalize_name("TEST_KEY") == "test_key"

    def test_normalize_name_preserve(self, tmp_path):
        """Test name normalization with preserve."""
        with patch.object(ChezmoiStore, "_load_chezmoi_config"):
            store = ChezmoiStore(
                secrets_file=tmp_path / "test.age",
                identity_file=tmp_path / "key.txt",
                recipient="age1test",
                name_style="preserve",
            )

        assert store._normalize_name("Test_Key") == "Test_Key"
