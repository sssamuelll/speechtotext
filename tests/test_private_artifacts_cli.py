from __future__ import annotations

import hashlib

import pytest

from speechtotext.security.__main__ import main
from speechtotext.security.artifacts import (
    FakePrivateArtifactFilesystem,
    PrivateArtifactStore,
)


def _factory(tmp_path, payload=b"new-artifact"):
    filesystem = FakePrivateArtifactFilesystem(
        known_local_app_data=tmp_path,
        acl_ok=True,
        encryption_ok=True,
    )
    source = filesystem.private_source("incoming/new.json", payload)
    store = PrivateArtifactStore.current_user(filesystem=filesystem)
    return lambda: store, source, filesystem


def _argv(source, *, digest, name="descriptors/provider-v1.json"):
    return [
        "promote",
        "--source",
        str(source),
        "--name",
        name,
        "--expected-sha256",
        digest,
        "--max-bytes",
        "1024",
    ]


def test_cli_promote_emite_solo_confirmacion_sanitizada(tmp_path, capsys):
    factory, source, _ = _factory(tmp_path)
    digest = hashlib.sha256(b"new-artifact").hexdigest()
    assert main(_argv(source, digest=digest), store_factory=factory) == 0
    captured = capsys.readouterr()
    assert captured.out == "OK artifact_promoted=true\n"
    assert captured.err == ""


@pytest.mark.parametrize("failure", ["collision", "hash", "exception"])
def test_cli_no_filtra_source_hash_nombre_ni_excepcion(tmp_path, capsys, failure):
    factory, source, filesystem = _factory(tmp_path)
    malicious_name = "descriptors/secret-do-not-leak.json"
    malicious_hash = "0" * 64
    if failure == "collision":
        filesystem.install(malicious_name, b"existing")
        digest = hashlib.sha256(b"new-artifact").hexdigest()
    elif failure == "hash":
        digest = malicious_hash
    else:
        digest = hashlib.sha256(b"new-artifact").hexdigest()
        filesystem.inject_source_fault(source, "exception:/private/leak")

    assert (
        main(
            _argv(source, digest=digest, name=malicious_name),
            store_factory=factory,
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "ERROR code=artifact_promotion_failed\n"
    combined = output.out + output.err
    assert str(source) not in combined
    assert malicious_name not in combined
    assert malicious_hash not in combined
    assert "/private/leak" not in combined


def test_cli_servicio_activo_falla_antes_de_abrir_source(tmp_path, capsys):
    factory, source, filesystem = _factory(tmp_path)
    digest = hashlib.sha256(b"new-artifact").hexdigest()
    with factory().runtime_session():
        assert main(_argv(source, digest=digest), store_factory=factory) == 1
    assert filesystem.source_open_count == 0
    assert capsys.readouterr().err == "ERROR code=artifact_promotion_failed\n"


def test_cli_argumentos_invalidos_emiten_un_solo_codigo(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["promote", "--source", "sensitive-path"])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "ERROR code=invalid_arguments\n"
    assert "sensitive-path" not in output.err
