import typer
from typing import Annotated
from pathlib import Path

from .backends import OnePasswordStore, KeychainStore, ChezmoiStore
from .sync import SyncEngine, SyncDirection
from .config import load_patterns, filter_keys_by_pattern

app = typer.Typer(help="Sync API keys between 1Password, Apple Keychain, and Chezmoi")


@app.command()
def sync(
    direction: Annotated[
        SyncDirection, typer.Argument(help="Sync direction")
    ] = SyncDirection.OP_TO_KEYCHAIN,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview without changes")
    ] = False,
    sync_deletions: Annotated[
        bool, typer.Option("--sync-deletions", help="Delete missing keys")
    ] = False,
    unlock: Annotated[
        bool, typer.Option("--unlock", help="Prompt to unlock keychain before sync")
    ] = False,
    case_sensitive: Annotated[
        bool,
        typer.Option(
            "--case-sensitive/--no-case-sensitive",
            help="Case sensitive pattern matching",
        ),
    ] = True,
    vault: Annotated[str, typer.Option(help="1Password vault name")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service name")] = "api-keys",
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show progress messages")
    ] = False,
):
    """Sync API keys between stores.

    Discovers keys by pattern matching (_TOKEN, _API, _KEY, _PASSWORD, _SECRET, _CREDENTIAL).
    """
    patterns = load_patterns(config)

    if direction == SyncDirection.OP_TO_KEYCHAIN:
        typer.echo("Syncing: 1Password → Keychain")
        if verbose:
            typer.echo("  → Connecting to 1Password...", err=True)
        op_store = OnePasswordStore(vault)
        if verbose:
            typer.echo("  → Connecting to Keychain...", err=True)
        kc_store = KeychainStore(service)
        engine = SyncEngine(op_store, kc_store, patterns, case_sensitive)
    else:
        typer.echo("Syncing: Keychain → 1Password")
        if verbose:
            typer.echo("  → Connecting to Keychain...", err=True)
        kc_store = KeychainStore(service)
        if verbose:
            typer.echo("  → Connecting to 1Password...", err=True)
        op_store = OnePasswordStore(vault)
        engine = SyncEngine(kc_store, op_store, patterns, case_sensitive)

    if unlock or kc_store.is_locked():
        if not kc_store.unlock():
            typer.echo("Failed to unlock keychain", err=True)
            raise typer.Exit(1)

    if dry_run:
        typer.echo("[DRY RUN]")

    if verbose:
        typer.echo("  → Fetching keys from source...", err=True)

    result = engine.sync(dry_run=dry_run, sync_deletions=sync_deletions, verbose=verbose)

    for name in result.synced:
        typer.echo(f"  ✓ {'Would sync' if dry_run else 'Synced'}: {name}")
    for name in result.deleted:
        typer.echo(f"  ✗ {'Would delete' if dry_run else 'Deleted'}: {name}")
    for name in result.errors:
        typer.echo(f"  ⚠ Error: {name}", err=True)

    typer.echo(
        f"\nSummary: {len(result.synced)} synced, {len(result.deleted)} deleted, {len(result.skipped)} unchanged"
    )


@app.command()
def get(
    name: Annotated[str, typer.Argument(help="Key name")],
    source: Annotated[str, typer.Option(help="Source: op or keychain")] = "keychain",
    vault: Annotated[str, typer.Option(help="1Password vault")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
):
    """Get a single API key."""
    store = OnePasswordStore(vault) if source == "op" else KeychainStore(service)
    value = store.get(name)
    if value:
        typer.echo(value)
    else:
        typer.echo(f"Key not found: {name}", err=True)
        raise typer.Exit(1)


@app.command()
def put(
    name: Annotated[str, typer.Argument(help="Key name")],
    value: Annotated[str, typer.Argument(help="Key value")],
    target: Annotated[str, typer.Option(help="Target: op, keychain, or both")] = "both",
    unlock: Annotated[
        bool, typer.Option("--unlock", help="Prompt to unlock keychain before storing")
    ] = False,
    vault: Annotated[str, typer.Option(help="1Password vault")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
):
    """Store an API key."""
    from .models import APIKey

    key = APIKey(name, value)
    kc_store = KeychainStore(service)

    if unlock and target in ("keychain", "both"):
        if kc_store.is_locked():
            typer.echo("Keychain is locked. Unlocking...")
            if not kc_store.unlock():
                typer.echo("Failed to unlock keychain", err=True)
                raise typer.Exit(1)
            typer.echo("Keychain unlocked successfully")

    if target in ("op", "both"):
        if OnePasswordStore(vault).put(key):
            typer.echo(f"✓ Stored in 1Password: {name}")
        else:
            typer.echo(f"✗ Failed to store in 1Password: {name}", err=True)

    if target in ("keychain", "both"):
        if kc_store.put(key):
            typer.echo(f"✓ Stored in Keychain: {name}")
        else:
            typer.echo(f"✗ Failed to store in Keychain: {name}", err=True)


@app.command("list")
def list_keys(
    source: Annotated[str, typer.Option(help="Source: op or keychain")] = "keychain",
    case_sensitive: Annotated[
        bool,
        typer.Option(
            "--case-sensitive/--no-case-sensitive",
            help="Case sensitive pattern matching",
        ),
    ] = True,
    vault: Annotated[str, typer.Option(help="1Password vault")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
):
    """List all API keys matching configured patterns."""
    patterns = load_patterns(config)
    store = OnePasswordStore(vault) if source == "op" else KeychainStore(service)

    all_keys = store.list_all_keys()
    matching_names = filter_keys_by_pattern(
        list(all_keys.keys()), patterns, case_sensitive
    )

    for name in sorted(matching_names):
        typer.echo(f"✓ {name}")


def _is_valid_shell_var(name: str) -> bool:
    """Check if name is a valid shell variable name."""
    import re

    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


@app.command("export-env")
def export_env(
    case_sensitive: Annotated[
        bool,
        typer.Option(
            "--case-sensitive/--no-case-sensitive",
            help="Case sensitive pattern matching",
        ),
    ] = True,
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
):
    """Output export commands for all keys in Keychain (source in shell)."""
    patterns = load_patterns(config)
    store = KeychainStore(service)

    all_keys = store.list_all_keys()
    matching_names = filter_keys_by_pattern(
        list(all_keys.keys()), patterns, case_sensitive
    )

    for name in sorted(matching_names):
        # Sanitize key name (strip whitespace)
        clean_name = name.strip()
        if not _is_valid_shell_var(clean_name):
            typer.echo(f"# Skipping invalid variable name: {name!r}", err=True)
            continue
        value = all_keys[name]
        # Escape single quotes in value
        escaped = value.replace("'", "'\"'\"'")
        typer.echo(f"export {clean_name}='{escaped}'")


@app.command("chezmoi-sync")
def chezmoi_sync(
    direction: Annotated[
        str, typer.Argument(help="Direction: op-to-chezmoi or chezmoi-to-op")
    ] = "op-to-chezmoi",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview without changes")
    ] = False,
    sync_deletions: Annotated[
        bool, typer.Option("--sync-deletions", help="Delete missing keys")
    ] = False,
    case_sensitive: Annotated[
        bool,
        typer.Option(
            "--case-sensitive/--no-case-sensitive",
            help="Case sensitive pattern matching",
        ),
    ] = True,
    name_style: Annotated[
        str,
        typer.Option(
            help="Name style for chezmoi: upper, lower, or preserve"
        ),
    ] = "upper",
    vault: Annotated[str, typer.Option(help="1Password vault name")] = "API_KEYS",
    secrets_file: Annotated[
        Path | None, typer.Option(help="Chezmoi secrets.json.age path")
    ] = None,
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show progress messages")
    ] = False,
):
    """Sync API keys between 1Password and Chezmoi.

    Directions:
      op-to-chezmoi: Sync from 1Password to chezmoi secrets.json.age
      chezmoi-to-op: Sync from chezmoi secrets.json.age to 1Password
    """
    patterns = load_patterns(config)

    if direction == "op-to-chezmoi":
        typer.echo("Syncing: 1Password → Chezmoi")
        if verbose:
            typer.echo("  → Connecting to 1Password...", err=True)
        op_store = OnePasswordStore(vault)
        if verbose:
            typer.echo("  → Loading chezmoi secrets...", err=True)
        cz_store = ChezmoiStore(secrets_file=secrets_file, name_style=name_style)  # type: ignore
        engine = SyncEngine(op_store, cz_store, patterns, case_sensitive)
    elif direction == "chezmoi-to-op":
        typer.echo("Syncing: Chezmoi → 1Password")
        if verbose:
            typer.echo("  → Loading chezmoi secrets...", err=True)
        cz_store = ChezmoiStore(secrets_file=secrets_file, name_style=name_style)  # type: ignore
        if verbose:
            typer.echo("  → Connecting to 1Password...", err=True)
        op_store = OnePasswordStore(vault)
        engine = SyncEngine(cz_store, op_store, patterns, case_sensitive)
    else:
        typer.echo(f"Invalid direction: {direction}", err=True)
        typer.echo("Use: op-to-chezmoi or chezmoi-to-op", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo("[DRY RUN]")

    if verbose:
        typer.echo("  → Fetching keys from source...", err=True)

    result = engine.sync(dry_run=dry_run, sync_deletions=sync_deletions, verbose=verbose)

    for name in result.synced:
        typer.echo(f"  ✓ {'Would sync' if dry_run else 'Synced'}: {name}")
    for name in result.deleted:
        typer.echo(f"  ✗ {'Would delete' if dry_run else 'Deleted'}: {name}")
    for name in result.errors:
        typer.echo(f"  ⚠ Error: {name}", err=True)

    typer.echo(
        f"\nSummary: {len(result.synced)} synced, {len(result.deleted)} deleted, {len(result.skipped)} unchanged"
    )


@app.command("chezmoi-list")
def chezmoi_list(
    case_sensitive: Annotated[
        bool,
        typer.Option(
            "--case-sensitive/--no-case-sensitive",
            help="Case sensitive pattern matching",
        ),
    ] = True,
    name_style: Annotated[
        str,
        typer.Option(help="Name style: upper, lower, or preserve"),
    ] = "preserve",
    secrets_file: Annotated[
        Path | None, typer.Option(help="Chezmoi secrets.json.age path")
    ] = None,
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
):
    """List all API keys in chezmoi secrets.json.age matching configured patterns."""
    patterns = load_patterns(config)
    store = ChezmoiStore(secrets_file=secrets_file, name_style=name_style)  # type: ignore

    all_keys = store.list_all_keys()
    matching_names = filter_keys_by_pattern(
        list(all_keys.keys()), patterns, case_sensitive
    )

    for name in sorted(matching_names):
        typer.echo(f"✓ {name}")


if __name__ == "__main__":
    app()
