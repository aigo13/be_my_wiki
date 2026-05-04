"""Command-line interface for be_my_wiki.

Subcommands:
- ``index``: walk the vault and update the vector store (incremental)
- ``stats``: show indexed counts
- ``search``: run a query against the index from the terminal (debugging)

Entry point registered in pyproject.toml as ``my-wiki``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from .config import Config, load_config
from .embedding.bge_m3 import BgeM3Embedder
from .indexer.pipeline import Indexer
from .indexer.watcher import VaultWatcher
from .store.chroma import ChromaStore

# Force UTF-8 on stdout/stderr so non-ASCII content (e.g. Korean snippets
# from `search`) renders correctly on Windows consoles that default to
# cp949. Best-effort: silently skip if the streams cannot be reconfigured
# (e.g. when piped to a non-text consumer).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

app = typer.Typer(
    help="be_my_wiki: Obsidian vault context-search MCP utilities.",
    no_args_is_help=True,
)


@app.command()
def index(
    config: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config.toml."
    ),
    prune: bool = typer.Option(
        True,
        "--prune/--no-prune",
        help="Remove store entries for files no longer on disk.",
    ),
) -> None:
    """Walk the vault and (re-)index every .md file."""
    cfg = load_config(config)
    indexer = _build_indexer(cfg)

    typer.echo(f"Indexing vault: {cfg.vault.path}")
    r = indexer.index_directory(prune_orphans=prune)

    typer.echo(
        f"Notes: total={r.notes_total} changed={r.notes_changed} "
        f"unchanged={r.notes_unchanged} failed={r.notes_failed} pruned={r.notes_pruned}"
    )
    typer.echo(
        f"Chunks: added={r.chunks_added} updated={r.chunks_updated} "
        f"skipped={r.chunks_skipped} deleted={r.chunks_deleted}"
    )


@app.command()
def stats(
    config: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config.toml."
    ),
) -> None:
    """Print vector-store statistics."""
    cfg = load_config(config)
    store = _build_store(cfg)
    s = store.stats()
    typer.echo(f"Total chunks: {s.total_chunks}")
    typer.echo(f"Total notes: {s.total_notes}")


@app.command()
def watch(
    config: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config.toml."
    ),
    initial: bool = typer.Option(
        True,
        "--initial/--no-initial",
        help="Run a full index_directory before starting the watcher.",
    ),
) -> None:
    """Watch the vault and incrementally re-index on changes (Ctrl+C to stop)."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(config)
    indexer = _build_indexer(cfg)

    if initial:
        typer.echo(f"Initial index: {cfg.vault.path}")
        r = indexer.index_directory()
        typer.echo(
            f"Initial: changed={r.notes_changed} unchanged={r.notes_unchanged} "
            f"failed={r.notes_failed} pruned={r.notes_pruned}"
        )

    watcher = VaultWatcher(
        indexer=indexer,
        debounce_seconds=cfg.indexer.debounce_seconds,
    )
    typer.echo(f"Watching {cfg.vault.path}  (Ctrl+C to stop)")
    watcher.run_forever()


@app.command(name="ssl-init")
def ssl_init(
    out_dir: Path = typer.Option(
        Path("./data/certs"),
        "--out",
        "-o",
        help="Directory to write certificate files into.",
    ),
    hostnames: list[str] = typer.Option(
        ["localhost", "127.0.0.1"],
        "--host",
        help="Hostnames or IPs to include in the server cert SAN. Repeatable.",
    ),
) -> None:
    """Generate a local root CA + server certificate for HTTPS MCP serving.

    Writes four files under ``--out``:

    \b
      be-my-wiki-ca.pem        - Root CA (install this in Windows trust store)
      be-my-wiki-ca-key.pem    - Root CA private key (keep safe; offline use)
      server.pem               - Server cert (pass to my-wiki-mcp --ssl-cert)
      server-key.pem           - Server private key (pass to --ssl-key)

    Then prints the PowerShell command to install the root CA in
    ``Cert:\\LocalMachine\\Root`` so Claude Desktop trusts the server.
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "be-my-wiki Local CA")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    san_entries: list = []
    for h in hostnames:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san_entries.append(x509.DNSName(h))

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        )
        .issuer_name(ca_subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName(san_entries), critical=False
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    pem = serialization.Encoding.PEM
    no_enc = serialization.NoEncryption()
    pkcs8 = serialization.PrivateFormat.PKCS8

    files = {
        "be-my-wiki-ca.pem": ca_cert.public_bytes(pem),
        "be-my-wiki-ca-key.pem": ca_key.private_bytes(pem, pkcs8, no_enc),
        "server.pem": server_cert.public_bytes(pem),
        "server-key.pem": server_key.private_bytes(pem, pkcs8, no_enc),
    }
    for name, data in files.items():
        (out_dir / name).write_bytes(data)

    ca_path = (out_dir / "be-my-wiki-ca.pem").resolve()
    server_cert_path = (out_dir / "server.pem").resolve()
    server_key_path = (out_dir / "server-key.pem").resolve()
    san_str = ", ".join(hostnames)

    typer.echo(f"Generated certs in: {out_dir.resolve()}")
    typer.echo(f"  Root CA   : be-my-wiki-ca.pem  (valid 10 years)")
    typer.echo(f"  Server cert: server.pem  (SAN: {san_str}, valid 825 days)")
    typer.echo("")
    typer.echo("Next steps")
    typer.echo("----------")
    typer.echo("1) Install the root CA in Windows trust store (PowerShell as admin):")
    typer.echo(
        f'   Import-Certificate -FilePath "{ca_path}" '
        '-CertStoreLocation Cert:\\LocalMachine\\Root'
    )
    typer.echo("")
    typer.echo("2) Start the MCP server over HTTPS:")
    typer.echo(
        '   my-wiki-mcp --transport streamable-http --port 7777 \\\n'
        '               --config config.toml \\\n'
        f'               --ssl-cert "{server_cert_path}" \\\n'
        f'               --ssl-key  "{server_key_path}"'
    )
    typer.echo("")
    typer.echo("3) In Claude Desktop, add a Custom Connector with URL:")
    typer.echo("   https://localhost:7777/mcp")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results."),
    config: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config.toml."
    ),
) -> None:
    """Run a semantic search and print top-k hits."""
    cfg = load_config(config)
    embedder = _build_embedder(cfg)
    store = _build_store(cfg)

    vec = embedder.embed_query(query)
    hits = store.search(vec, top_k=top_k)

    if not hits:
        typer.echo("No hits.")
        return

    for i, hit in enumerate(hits, 1):
        heading = " > ".join(hit.heading_path) if hit.heading_path else "(no heading)"
        typer.echo(
            f"\n[{i}] {hit.note_path}  ::  {heading}  (score={hit.score:.3f})"
        )
        snippet = hit.body if len(hit.body) <= 240 else hit.body[:240] + "..."
        typer.echo(snippet)


# --- builders ---


def _build_indexer(cfg: Config) -> Indexer:
    return Indexer(
        vault_root=cfg.vault.path,
        embedder=_build_embedder(cfg),
        store=_build_store(cfg),
        max_tokens=cfg.chunking.max_tokens,
        heading_level=cfg.chunking.heading_level,
        ignore_dirs=_extract_ignore_dirs(cfg.vault.ignore),
    )


def _build_embedder(cfg: Config) -> BgeM3Embedder:
    return BgeM3Embedder(
        model_name=cfg.embedding.model,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
    )


def _build_store(cfg: Config) -> ChromaStore:
    if cfg.storage.backend != "chroma":
        raise typer.BadParameter(
            f"Unsupported storage backend: {cfg.storage.backend!r}"
        )
    return ChromaStore(
        persist_path=str(cfg.storage.chroma_path),
        collection=cfg.storage.collection,
    )


def _extract_ignore_dirs(patterns: list[str]) -> set[str]:
    """Translate v1 simple glob patterns (``<dir>/**``) to directory names.

    Sensible defaults are always included; user patterns can extend the
    set but cannot disable a default.
    """
    out: set[str] = {".obsidian", ".trash", ".git", "node_modules"}
    for p in patterns:
        p = p.strip()
        if p.endswith("/**"):
            d = p[:-3].strip("/")
            if "/" not in d and "*" not in d:
                out.add(d)
    return out
