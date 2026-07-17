from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from speechtotext.security.artifacts import (
    ArtifactIntegrityError,
    ArtifactLease,
    FakePrivateArtifactFilesystem,
    PrivateArtifactStore,
    default_private_artifact_filesystem,
)


def _store(tmp_path, payload=b"protected-calibrator"):
    filesystem = FakePrivateArtifactFilesystem(
        known_local_app_data=tmp_path,
        acl_ok=True,
        encryption_ok=True,
    )
    filesystem.install("calibrators/es-v1.json", payload)
    return PrivateArtifactStore.current_user(filesystem=filesystem), filesystem


def test_store_y_lease_solo_se_construyen_desde_factories(tmp_path):
    with pytest.raises(TypeError):
        PrivateArtifactStore(tmp_path)
    with pytest.raises(TypeError):
        ArtifactLease()


def test_import_security_no_carga_ctypes_ni_stacks_de_modelos():
    script = """
import sys
sys.path.insert(0, 'src')
import speechtotext.security
blocked = {'ctypes', 'faster_whisper', 'sklearn', 'scipy', 'torch', 'pyannote'}
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )


def test_lease_lee_una_vez_desde_handle_y_exige_sha_externo(tmp_path):
    payload = b"protected-calibrator"
    store, filesystem = _store(tmp_path, payload)
    expected = hashlib.sha256(payload).hexdigest()

    with store.lease(
        "calibrators/es-v1.json", expected_sha256=expected, max_bytes=1024
    ) as lease:
        assert lease.read_bytes_once() == payload
        with pytest.raises(ArtifactIntegrityError, match="consumido"):
            lease.read_bytes_once()
        with pytest.raises(PermissionError, match="sharing violation"):
            filesystem.replace("calibrators/es-v1.json", b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="activo"):
        lease.read_bytes_once()
    assert filesystem.open_counts["calibrators/es-v1.json"] == 1


def test_sha_incorrecto_y_limite_fallan_sin_segunda_apertura(tmp_path):
    store, filesystem = _store(tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="sha256"):
        with store.lease(
            "calibrators/es-v1.json", expected_sha256="0" * 64, max_bytes=1024
        ) as lease:
            lease.read_bytes_once()
    assert filesystem.open_counts["calibrators/es-v1.json"] == 1

    with pytest.raises(ArtifactIntegrityError, match="limite"):
        with store.lease(
            "calibrators/es-v1.json",
            expected_sha256=hashlib.sha256(b"protected-calibrator").hexdigest(),
            max_bytes=4,
        ) as lease:
            lease.read_bytes_once()


@pytest.mark.parametrize("fault", ["acl", "encryption", "reparse", "hardlink"])
def test_store_falla_cerrado_ante_evidencia_local_insegura(tmp_path, fault):
    store, filesystem = _store(tmp_path)
    filesystem.inject_fault("calibrators/es-v1.json", fault)

    with pytest.raises(ArtifactIntegrityError, match=fault):
        with store.lease(
            "calibrators/es-v1.json",
            expected_sha256=hashlib.sha256(b"protected-calibrator").hexdigest(),
            max_bytes=1024,
        ):
            pass


@pytest.mark.parametrize(
    "unsafe",
    [
        r"C:\outside.json",
        r"calibrators\es.json",
        "../outside.json",
        "calibrators/./es.json",
        "calibrators//es.json",
        "calibrators/es.json:zone.identifier",
        "calibrators/CON.json",
        "calibrators/com1.txt",
        "calibrators/es.json.",
        "calibrators/es.json ",
        "calibrators/es\x00.json",
        r"\\?\C:\outside.json",
        # Nombres internos reservados del store: temps de promocion, el lock
        # y cualquier componente con punto inicial.
        f".artifact-{'a' * 32}.tmp",
        ".runtime.lock",
        ".hidden/es.json",
        "calibrators/.hidden.json",
    ],
)
def test_store_rechaza_paths_ambiguos_antes_de_abrir(tmp_path, unsafe):
    store, filesystem = _store(tmp_path)
    with pytest.raises(ValueError, match="ruta relativa segura"):
        store.lease(unsafe, expected_sha256="0" * 64, max_bytes=1024)
    assert filesystem.total_open_count == 0


@pytest.mark.parametrize(
    "digest",
    ["0" * 63, "A" * 64, "g" * 64, "", None, True],
)
def test_store_rechaza_sha_no_canonico_antes_de_abrir(tmp_path, digest):
    store, filesystem = _store(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        store.lease("calibrators/es-v1.json", expected_sha256=digest, max_bytes=1024)
    assert filesystem.total_open_count == 0


@pytest.mark.parametrize("limit", [0, -1, True, 16 * 1024 * 1024 + 1])
def test_store_rechaza_limite_invalido_antes_de_abrir(tmp_path, limit):
    store, filesystem = _store(tmp_path)
    with pytest.raises(ValueError, match="max_bytes"):
        store.lease("calibrators/es-v1.json", expected_sha256="0" * 64, max_bytes=limit)
    assert filesystem.total_open_count == 0


def test_error_de_lectura_consume_el_lease(tmp_path):
    store, filesystem = _store(tmp_path)
    filesystem.inject_fault("calibrators/es-v1.json", "read")
    with store.lease(
        "calibrators/es-v1.json",
        expected_sha256=hashlib.sha256(b"protected-calibrator").hexdigest(),
        max_bytes=1024,
    ) as lease:
        with pytest.raises(OSError, match="lectura"):
            lease.read_bytes_once()
        with pytest.raises(ArtifactIntegrityError, match="consumido"):
            lease.read_bytes_once()


def test_limite_se_aplica_durante_lectura_aunque_stat_sea_menor(tmp_path):
    store, filesystem = _store(tmp_path, payload=b"1234")
    filesystem.inject_fault("calibrators/es-v1.json", "grow_after_stat")
    with store.lease(
        "calibrators/es-v1.json",
        expected_sha256=hashlib.sha256(b"12345").hexdigest(),
        max_bytes=4,
    ) as lease:
        with pytest.raises(ArtifactIntegrityError, match="limite"):
            lease.read_bytes_once()


def test_lectura_rechaza_tamano_distinto_al_observado_en_handle(tmp_path):
    store, filesystem = _store(tmp_path, payload=b"1234")
    filesystem.inject_fault("calibrators/es-v1.json", "grow_after_stat")
    with store.lease(
        "calibrators/es-v1.json",
        expected_sha256=hashlib.sha256(b"12345").hexdigest(),
        max_bytes=1024,
    ) as lease:
        with pytest.raises(ArtifactIntegrityError, match="tamano"):
            lease.read_bytes_once()


@pytest.mark.parametrize(
    "component", ["known_local_app_data", "speechtotext", "artifacts"]
)
@pytest.mark.parametrize("fault", ["acl", "encryption", "reparse", "identity"])
def test_cadena_completa_del_root_falla_cerrado(tmp_path, component, fault):
    store, filesystem = _store(tmp_path)
    filesystem.inject_root_fault(component, fault)
    with pytest.raises(ArtifactIntegrityError, match=f"{component}.*{fault}"):
        with store.lease(
            "calibrators/es-v1.json",
            expected_sha256=hashlib.sha256(b"protected-calibrator").hexdigest(),
            max_bytes=1024,
        ):
            pass
    assert filesystem.total_open_count == 0


def test_default_falla_cerrado_fuera_de_windows(monkeypatch):
    monkeypatch.setattr("speechtotext.security.artifacts.sys.platform", "linux")
    with pytest.raises(ArtifactIntegrityError, match="Windows"):
        default_private_artifact_filesystem()


def test_promote_es_create_new_y_verifica_reopen(tmp_path):
    store, filesystem = _store(tmp_path)
    source = filesystem.private_source("incoming/new.json", b"new-artifact")
    expected = hashlib.sha256(b"new-artifact").hexdigest()

    store.promote_from_path(
        source,
        "descriptors/provider-v1.json",
        expected_sha256=expected,
        max_bytes=1024,
    )

    assert filesystem.promotion_events == [
        "offline_lock",
        "source_leased",
        "temp_secured",
        "temp_flushed",
        "create_new_committed",
        "destination_reopened",
        "destination_verified",
    ]
    with store.runtime_session():
        with store.lease(
            "descriptors/provider-v1.json",
            expected_sha256=expected,
            max_bytes=1024,
        ) as lease:
            assert lease.read_bytes_once() == b"new-artifact"


def test_promote_falla_ante_colision_tamper_o_servicio_activo(tmp_path):
    store, filesystem = _store(tmp_path)
    source = filesystem.private_source("incoming/new.json", b"new-artifact")
    expected = hashlib.sha256(b"new-artifact").hexdigest()
    filesystem.install("descriptors/provider-v1.json", b"existing")

    with pytest.raises(ArtifactIntegrityError, match="colision"):
        store.promote_from_path(
            source,
            "descriptors/provider-v1.json",
            expected_sha256=expected,
            max_bytes=1024,
        )
    assert filesystem.read("descriptors/provider-v1.json") == b"existing"

    filesystem.clear_promotion_events()
    with store.runtime_session():
        with pytest.raises(ArtifactIntegrityError, match="servicio activo"):
            store.promote_from_path(
                source,
                "descriptors/other.json",
                expected_sha256=expected,
                max_bytes=1024,
            )
    assert "source_leased" not in filesystem.promotion_events

    filesystem.inject_source_fault(source, "tamper")
    with pytest.raises(ArtifactIntegrityError, match="sha256|identidad"):
        store.promote_from_path(
            source,
            "descriptors/other.json",
            expected_sha256=expected,
            max_bytes=1024,
        )
    assert not filesystem.exists("descriptors/other.json")


def test_promotion_source_inseguro_falla_antes_de_publicar(tmp_path):
    store, filesystem = _store(tmp_path)
    source = filesystem.private_source("incoming/new.json", b"new-artifact")
    expected = hashlib.sha256(b"new-artifact").hexdigest()

    for fault in ("acl", "encryption", "reparse", "hardlink"):
        filesystem.inject_source_fault(source, fault)
        with pytest.raises(ArtifactIntegrityError, match=fault):
            store.promote_from_path(
                source,
                f"descriptors/{fault}.json",
                expected_sha256=expected,
                max_bytes=1024,
            )
        assert not filesystem.exists(f"descriptors/{fault}.json")
        filesystem.clear_source_faults(source)


@pytest.mark.parametrize("fault", ["crash_before_commit", "destination_tamper"])
def test_promocion_no_publica_parcial_y_limpia_solo_su_temp(tmp_path, fault):
    store, filesystem = _store(tmp_path)
    source = filesystem.private_source("incoming/new.json", b"new-artifact")
    expected = hashlib.sha256(b"new-artifact").hexdigest()
    filesystem.install_temp(".artifact-owned-by-other.tmp", b"unrelated")
    filesystem.inject_promotion_fault(fault)

    with pytest.raises(ArtifactIntegrityError, match="promocion|destino"):
        store.promote_from_path(
            source,
            "descriptors/provider-v1.json",
            expected_sha256=expected,
            max_bytes=1024,
        )

    assert not filesystem.exists("descriptors/provider-v1.json")
    assert filesystem.temp_names == (".artifact-owned-by-other.tmp",)


def test_lock_offline_reconoce_y_limpia_temp_reservado_de_caida(tmp_path):
    _, filesystem = _store(tmp_path)
    stale = f".artifact-{'a' * 32}.tmp"
    filesystem.install_temp(stale, b"incomplete")
    with filesystem.offline_promotion():
        assert filesystem.temp_names == ()
