from dataclasses import dataclass
from enum import Enum
from .models import APIKey, KeyStore
from .config import filter_keys_by_pattern, DEFAULT_PATTERNS


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
    def __init__(
        self,
        source: KeyStore,
        target: KeyStore,
        patterns: list[str] | None = None,
        case_sensitive: bool = True,
    ):
        self.source = source
        self.target = target
        self.patterns = patterns or DEFAULT_PATTERNS
        self.case_sensitive = case_sensitive

    def sync(self, dry_run: bool = False, sync_deletions: bool = False) -> SyncResult:
        result = SyncResult(synced=[], deleted=[], skipped=[], errors=[])

        # Discover all keys from both stores
        all_source_keys = self.source.list_all_keys()
        all_target_keys = self.target.list_all_keys()

        # Filter by pattern
        source_names = filter_keys_by_pattern(
            list(all_source_keys.keys()), self.patterns, self.case_sensitive
        )
        target_names = filter_keys_by_pattern(
            list(all_target_keys.keys()), self.patterns, self.case_sensitive
        )

        # Union of all matching key names
        all_names = set(source_names) | set(target_names)

        for name in sorted(all_names):
            src_val = all_source_keys.get(name)
            tgt_val = all_target_keys.get(name)

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
                # Key exists in target but not in source
                if not dry_run:
                    if self.target.delete(name):
                        result.deleted.append(name)
                    else:
                        result.errors.append(name)
                else:
                    result.deleted.append(name)

        return result
