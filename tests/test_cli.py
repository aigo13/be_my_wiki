"""End-to-end tests for the CLI.

Uses the conftest ``fake_st`` fixture so BgeM3Embedder can be instantiated
without downloading the real bge-m3 model. The store, indexer, chunker,
and CLI plumbing are all real.
"""

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from be_my_wiki.cli import app


def _write_config(tmp_path: Path, vault: Path, chroma: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            [vault]
            path = "{vault.as_posix()}"

            [storage]
            chroma_path = "{chroma.as_posix()}"
            collection = "test_cli"
            """
        ),
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def runner():
    return CliRunner()


def test_index_creates_chunks(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("## A\nbody", encoding="utf-8")
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    result = runner.invoke(app, ["index", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "added=1" in result.output


def test_index_unchanged_skips(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("## A\nbody", encoding="utf-8")
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    runner.invoke(app, ["index", "--config", str(cfg)])
    result = runner.invoke(app, ["index", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "skipped=1" in result.output
    assert "added=0" in result.output


def test_stats_after_index(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("## A\nbody", encoding="utf-8")
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    runner.invoke(app, ["index", "--config", str(cfg)])
    result = runner.invoke(app, ["stats", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Total chunks: 1" in result.output
    assert "Total notes: 1" in result.output


def test_search_returns_indexed_note(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("## A\nalpha", encoding="utf-8")
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    runner.invoke(app, ["index", "--config", str(cfg)])
    result = runner.invoke(
        app, ["search", "alpha", "--config", str(cfg), "-k", "1"]
    )
    assert result.exit_code == 0
    assert "n.md" in result.output


def test_search_no_hits_in_empty_store(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    result = runner.invoke(app, ["search", "anything", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "No hits" in result.output


def test_index_prune_orphans(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("## A\nx", encoding="utf-8")
    b = vault / "b.md"
    b.write_text("## B\ny", encoding="utf-8")
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    runner.invoke(app, ["index", "--config", str(cfg)])
    b.unlink()
    result = runner.invoke(app, ["index", "--config", str(cfg)])
    assert "pruned=1" in result.output


def test_index_skips_obsidian_dir(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "good.md").write_text("## A\nx", encoding="utf-8")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "config.md").write_text("## ignored\nz", encoding="utf-8")
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    result = runner.invoke(app, ["index", "--config", str(cfg)])
    assert "total=1" in result.output


# --- multilingual + TeX coverage (per project policy) ---


def test_korean_note_full_pipeline(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "ko.md").write_text(
        "---\ntags: [한국어]\n---\n\n## 서론\n안녕하세요",
        encoding="utf-8",
    )
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    r1 = runner.invoke(app, ["index", "--config", str(cfg)])
    assert r1.exit_code == 0
    assert "added=1" in r1.output

    r2 = runner.invoke(
        app, ["search", "안녕", "--config", str(cfg), "-k", "1"]
    )
    assert r2.exit_code == 0
    assert "ko.md" in r2.output


def test_ssl_init_generates_ca_and_server_certs(tmp_path, runner):
    out = tmp_path / "certs"
    result = runner.invoke(app, ["ssl-init", "--out", str(out)])
    assert result.exit_code == 0, result.output

    expected = {"be-my-wiki-ca.pem", "be-my-wiki-ca-key.pem", "server.pem", "server-key.pem"}
    assert {p.name for p in out.iterdir()} == expected

    # Parse the certs and verify the server cert is signed by the CA.
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
    from cryptography.hazmat.primitives import hashes

    ca = x509.load_pem_x509_certificate((out / "be-my-wiki-ca.pem").read_bytes())
    srv = x509.load_pem_x509_certificate((out / "server.pem").read_bytes())
    assert srv.issuer == ca.subject
    # Verify signature.
    ca.public_key().verify(
        srv.signature,
        srv.tbs_certificate_bytes,
        PKCS1v15(),
        srv.signature_hash_algorithm,
    )
    # SAN includes localhost + 127.0.0.1.
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    dns_names = san.get_values_for_type(x509.DNSName)
    ip_addrs = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
    assert "localhost" in dns_names
    assert "127.0.0.1" in ip_addrs

    # Output mentions the install + start commands.
    assert "Import-Certificate" in result.output
    assert "https://localhost:7777/mcp" in result.output


def test_ssl_init_custom_hostnames(tmp_path, runner):
    out = tmp_path / "certs"
    result = runner.invoke(
        app,
        ["ssl-init", "--out", str(out), "--host", "wiki.local", "--host", "10.0.0.5"],
    )
    assert result.exit_code == 0, result.output

    from cryptography import x509

    srv = x509.load_pem_x509_certificate((out / "server.pem").read_bytes())
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "wiki.local" in san.get_values_for_type(x509.DNSName)
    assert "10.0.0.5" in [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]


def test_tex_note_full_pipeline(tmp_path, runner, fake_st):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "math.md").write_text(
        "## Integral\n\n$$\n\\int_0^1 x \\, dx = \\frac{1}{2}\n$$",
        encoding="utf-8",
    )
    cfg = _write_config(tmp_path, vault, tmp_path / "chroma")

    r1 = runner.invoke(app, ["index", "--config", str(cfg)])
    assert r1.exit_code == 0
    assert "added=1" in r1.output

    r2 = runner.invoke(
        app, ["search", "integral", "--config", str(cfg), "-k", "1"]
    )
    assert r2.exit_code == 0
    assert "math.md" in r2.output
    assert r"\int_0^1 x \, dx" in r2.output
