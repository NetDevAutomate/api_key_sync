import sys
from dataclasses import dataclass
from enum import Enum
from .models import APIKey, KeyStore
from .config import filter_keys_by_pattern, DEFAULT_PATTERNS


class SyncDirection(Enum):
    OP_TO_KEYCHAIN = "op-to-keychain"
    KEYCHAIN_TO_OP = "keychain-to-op"
    OP_TO_CHEZMOI = "op-to-chezmoi"
    CHEZMOI_TO_OP = "chezmoi-to-op"


class SyncSafetyError(Exception):
    """Raised when a sync operation would be dangerous."""
    pass


@dataclass
class SyncResult:
    synced: list[str]
    deleted: list[str]
    skipped: list[str]
    errors: list[str]


def _log(msg: str, verbose: bool) -> None:
    """Print a message to stderr if verbose mode is enabled."""
    if verbose:
        print(f"  → {msg}", file=sys.stderr)


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

    def sync(
        self, dry_run: bool = False, sync_deletions: bool = False, verbose: bool = False
    ) -> SyncResult:
        result = SyncResult(synced=[], deleted=[], skipped=[], errors=[])

        # Discover all keys from both stores
        _log("Listing source keys...", verbose)
        all_source_keys = self.source.list_all_keys()
        _log(f"Found {len(all_source_keys)} keys in source", verbose)

        _log("Listing target keys...", verbose)
        all_target_keys = self.target.list_all_keys()
        _log(f"Found {len(all_target_keys)} keys in target", verbose)

        # Safety check: refuse to delete all target keys if source returns 0
        if sync_deletions and len(all_source_keys) == 0 and len(all_target_keys) > 0:
            raise SyncSafetyError(
                f"SAFETY: Source returned 0 keys but target has {len(all_target_keys)} keys. "
                "This would delete all target keys. Check source authentication/connectivity. "
                "If this is intentional, manually delete target keys first."
            )

        # Filter by pattern
        _log("Filtering by patterns...", verbose)
        source_names = filter_keys_by_pattern(
            list(all_source_keys.keys()), self.patterns, self.case_sensitive
        )
        target_names = filter_keys_by_pattern(
            list(all_target_keys.keys()), self.patterns, self.case_sensitive
        )
        _log(f"Matched {len(source_names)} source, {len(target_names)} target", verbose)

        # Union of all matching key names
        all_names = set(source_names) | set(target_names)

        _log(f"Processing {len(all_names)} unique keys...", verbose)
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
