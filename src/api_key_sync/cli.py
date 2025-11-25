import typer
from typing import Annotated
from pathlib import Path

from .backends import OnePasswordStore, KeychainStore
from .sync import SyncEngine, SyncDirection
from .config import load_key_list

app = typer.Typer(help="Sync API keys between 1Password and Apple Keychain")


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
    vault: Annotated[str, typer.Option(help="1Password vault name")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service name")] = "api-keys",
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
):
    """Sync API keys between stores."""
    key_names = load_key_list(config)
    op_store = OnePasswordStore(vault)
    kc_store = KeychainStore(service)

    if direction == SyncDirection.OP_TO_KEYCHAIN:
        engine = SyncEngine(op_store, kc_store, key_names)
        typer.echo("Syncing: 1Password → Keychain")
    else:
        engine = SyncEngine(kc_store, op_store, key_names)
        typer.echo("Syncing: Keychain → 1Password")

    if dry_run:
        typer.echo("[DRY RUN]")

    result = engine.sync(dry_run=dry_run, sync_deletions=sync_deletions)

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
    vault: Annotated[str, typer.Option(help="1Password vault")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
):
    """Store an API key."""
    from .models import APIKey

    key = APIKey(name, value)

    if target in ("op", "both"):
        if OnePasswordStore(vault).put(key):
            typer.echo(f"✓ Stored in 1Password: {name}")
        else:
            typer.echo(f"✗ Failed to store in 1Password: {name}", err=True)

    if target in ("keychain", "both"):
        if KeychainStore(service).put(key):
            typer.echo(f"✓ Stored in Keychain: {name}")
        else:
            typer.echo(f"✗ Failed to store in Keychain: {name}", err=True)


@app.command("list")
def list_keys(
    source: Annotated[str, typer.Option(help="Source: op or keychain")] = "keychain",
    vault: Annotated[str, typer.Option(help="1Password vault")] = "API_KEYS",
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
):
    """List all configured API keys and their status."""
    key_names = load_key_list(config)
    store = OnePasswordStore(vault) if source == "op" else KeychainStore(service)

    for name in key_names:
        value = store.get(name)
        status = "✓" if value else "✗"
        typer.echo(f"{status} {name}")


@app.command("export-env")
def export_env(
    service: Annotated[str, typer.Option(help="Keychain service")] = "api-keys",
    config: Annotated[Path | None, typer.Option(help="Path to config file")] = None,
):
    """Output export commands for all keys in Keychain (source in shell)."""
    key_names = load_key_list(config)
    store = KeychainStore(service)

    for name in key_names:
        value = store.get(name)
        if value:
            # Escape single quotes in value
            escaped = value.replace("'", "'\"'\"'")
            typer.echo(f"export {name}='{escaped}'")


if __name__ == "__main__":
    app()
