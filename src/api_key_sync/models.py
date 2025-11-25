from dataclasses import dataclass
from typing import Protocol
from abc import abstractmethod


@dataclass(frozen=True)
class APIKey:
    name: str
    value: str


class KeyStore(Protocol):
    @abstractmethod
    def get(self, name: str) -> str | None: ...

    @abstractmethod
    def put(self, key: APIKey) -> bool: ...

    @abstractmethod
    def delete(self, name: str) -> bool: ...

    @abstractmethod
    def list_keys(self, key_names: list[str]) -> dict[str, str]: ...
