from dataclasses import dataclass
from enum import Enum
from .models import APIKey, KeyStore


class SyncDirection(Enum):
    OP_TO_KEYCHAIN = "op-to-keychain"
    KEYCHAIN_TO_OP = "keychain-to-op"


@dataclass
class SyncResult:
    synced: list[str]
    deleted: list[str]
    skipped: list[str]
    errors: list[str]


class SyncEngine:
    def __init__(self, source: KeyStore, target: KeyStore, key_names: list[str]):
        self.source = source
        self.target = target
        self.key_names = key_names

    def sync(self, dry_run: bool = False, sync_deletions: bool = False) -> SyncResult:
        result = SyncResult(synced=[], deleted=[], skipped=[], errors=[])

        source_keys = self.source.list_keys(self.key_names)
        target_keys = self.target.list_keys(self.key_names)

        for name in self.key_names:
            src_val = source_keys.get(name)
            tgt_val = target_keys.get(name)

            if src_val:
                if src_val != tgt_val:
                    if not dry_run:
                        if self.target.put(APIKey(name, src_val)):
                            result.synced.append(name)
                        else:
                            result.errors.append(name)
                    else:
                        result.synced.append(name)
                else:
                    result.skipped.append(name)
            elif tgt_val and sync_deletions:
                if not dry_run:
                    if self.target.delete(name):
                        result.deleted.append(name)
                    else:
                        result.errors.append(name)
                else:
                    result.deleted.append(name)

        return result
