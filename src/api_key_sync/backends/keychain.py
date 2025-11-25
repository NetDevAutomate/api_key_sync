import subprocess
from ..models import APIKey


class KeychainStore:
    def __init__(self, service: str = "api-keys"):
        self.service = service

    def _run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["security"] + args, capture_output=True, text=True, check=check
        )

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

    def list_keys(self, key_names: list[str]) -> dict[str, str]:
        return {name: val for name in key_names if (val := self.get(name))}
