import subprocess
import getpass
from ..models import APIKey


class KeychainStore:
    def __init__(self, service: str = "api-keys"):
        self.service = service

    def _run(
        self, args: list[str], check: bool = True, input: str | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["security"] + args,
            capture_output=True,
            text=True,
            check=check,
            input=input,
        )

    def unlock(self, password: str | None = None) -> bool:
        """Unlock the login keychain.

        Args:
            password: Keychain password. If None, prompts interactively.

        Returns:
            True if unlock succeeded, False otherwise.
        """
        if password is None:
            password = getpass.getpass("Keychain password: ")

        try:
            self._run(
                ["unlock-keychain", "-p", password, "login.keychain-db"], check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def is_locked(self) -> bool:
        """Check if the keychain requires unlock for write operations."""
        try:
            # Try to add and immediately delete a test key
            self._run(
                [
                    "add-generic-password",
                    "-a",
                    "__test__",
                    "-s",
                    self.service,
                    "-w",
                    "test",
                    "-U",
                ],
                check=True,
            )
            self._run(
                ["delete-generic-password", "-a", "__test__", "-s", self.service],
                check=False,
            )
            return False
        except subprocess.CalledProcessError:
            return True

    def get(self, name: str) -> str | None:
        try:
            result = self._run(
                ["find-generic-password", "-a", name, "-s", self.service, "-w"]
            )
            return result.stdout.strip() or None
        except subprocess.CalledProcessError:
            return None

    def put(self, key: APIKey) -> bool:
        try:
            self._run(
                [
                    "add-generic-password",
                    "-a",
                    key.name,
                    "-s",
                    self.service,
                    "-w",
                    key.value,
                    "-U",  # Update if exists
                ]
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def delete(self, name: str) -> bool:
        try:
            self._run(["delete-generic-password", "-a", name, "-s", self.service])
            return True
        except subprocess.CalledProcessError:
            return False

    def list_all_keys(self) -> dict[str, str]:
        """List all keys for this service in the keychain."""
        import re

        try:
            # Dump keychain and parse for our service
            result = self._run(["dump-keychain", "login.keychain-db"], check=True)
            output = result.stdout

            keys = {}
            # Parse keychain dump format - look for generic passwords with our service
            # Format has "svce"<blob>="service" and "acct"<blob>="account"
            current_service = None
            current_account = None

            for line in output.split("\n"):
                line = line.strip()
                # Match service: "svce"<blob>="api-keys"
                if match := re.match(r'"svce"<blob>="([^"]*)"', line):
                    current_service = match.group(1)
                # Match account: "acct"<blob>="KEY_NAME"
                elif match := re.match(r'"acct"<blob>="([^"]*)"', line):
                    current_account = match.group(1)
                # Reset on new keychain entry
                elif line.startswith("keychain:") or line.startswith("class:"):
                    if (
                        current_service == self.service
                        and current_account
                        and current_account != "__test__"
                    ):
                        # Get the actual value
                        val = self.get(current_account)
                        if val:
                            keys[current_account] = val
                    current_service = None
                    current_account = None

            # Check last entry
            if (
                current_service == self.service
                and current_account
                and current_account != "__test__"
            ):
                val = self.get(current_account)
                if val:
                    keys[current_account] = val

            return keys
        except subprocess.CalledProcessError:
            return {}

    def list_keys(self, key_names: list[str]) -> dict[str, str]:
        """List specific keys by name."""
        return {name: val for name in key_names if (val := self.get(name))}
