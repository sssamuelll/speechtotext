from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import os
import subprocess
import sys

import pytest

from speechtotext.security.artifacts import (
    ArtifactFileHandle,
    ArtifactIntegrityError,
    ArtifactSourceHandle,
    PrivateArtifactStore,
    WindowsPrivateArtifactFilesystem,
)


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="requiere Win32")


def _adapter(tmp_path, *, acl_ok=True, encryption_ok=True):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    return WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: acl_ok,
        encryption_probe=lambda handle, path: (
            "test-encryption" if encryption_ok else None
        ),
        secure_new_file=lambda handle, path: None,
    ), root


def test_adapter_windows_lee_un_handle_y_bloquea_replace_hasta_teardown(tmp_path):
    adapter, root = _adapter(tmp_path)
    payload = b"private"
    artifact = root / "calibrators" / "es.json"
    artifact.parent.mkdir()
    artifact.write_bytes(payload)
    store = PrivateArtifactStore.current_user(filesystem=adapter)

    with store.lease(
        "calibrators/es.json",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        max_bytes=1024,
    ) as lease:
        with pytest.raises(PermissionError):
            artifact.write_bytes(b"tamper")
        with pytest.raises(PermissionError):
            artifact.rename(root / "renamed.json")
        with pytest.raises(PermissionError):
            artifact.unlink()
        assert lease.read_bytes_once() == payload

    artifact.write_bytes(b"after")
    assert artifact.read_bytes() == b"after"


def test_adapter_windows_falla_si_acl_o_cifrado_no_se_demuestran(tmp_path):
    for field in ("acl", "encryption"):
        case = tmp_path / field
        adapter, root = _adapter(
            case,
            acl_ok=field != "acl",
            encryption_ok=field != "encryption",
        )
        artifact = root / "private.json"
        artifact.write_bytes(b"private")
        store = PrivateArtifactStore.current_user(filesystem=adapter)
        with pytest.raises(ArtifactIntegrityError, match=field):
            with store.lease(
                "private.json",
                expected_sha256=hashlib.sha256(b"private").hexdigest(),
                max_bytes=1024,
            ):
                pass


@pytest.mark.parametrize(
    "component", ["speechtotext", "artifacts", "calibrators", "es.json"]
)
def test_adapter_windows_valida_cada_componente_propio_de_la_cadena(
    tmp_path, component
):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    artifact = root / "calibrators" / "es.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"private")
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: path.name != component,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="acl"):
        with store.lease(
            "calibrators/es.json",
            expected_sha256=hashlib.sha256(b"private").hexdigest(),
            max_bytes=1024,
        ):
            pass


def test_adapter_windows_no_prueba_acl_ni_cifrado_en_el_known_folder(tmp_path):
    # Probe scoping: %LOCALAPPDATA% es el ancla confiable del OS. Una probe de
    # ACL/cifrado que lo rechaza no puede bloquear el lease; los componentes
    # propios del store si se prueban (test anterior).
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    (root / "private.json").write_bytes(b"private")
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: path.name != "LocalAppData",
        encryption_probe=lambda handle, path: (
            None if path.name == "LocalAppData" else "test-encryption"
        ),
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with store.lease(
        "private.json",
        expected_sha256=hashlib.sha256(b"private").hexdigest(),
        max_bytes=1024,
    ) as lease:
        assert lease.read_bytes_once() == b"private"


def test_adapter_windows_defaults_no_son_probes_permisivas(tmp_path):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    (root / "private.json").write_bytes(b"private")
    adapter = WindowsPrivateArtifactFilesystem(known_folder_probe=lambda: local)
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="acl|encryption"):
        with store.lease(
            "private.json",
            expected_sha256=hashlib.sha256(b"private").hexdigest(),
            max_bytes=1024,
        ):
            pass


@pytest.mark.parametrize("field", ["acl", "encryption"])
def test_adapter_windows_rechaza_evidencia_de_probe_con_tipo_ambiguo(tmp_path, field):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    (root / "private.json").write_bytes(b"private")
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: 1 if field == "acl" else True,
        encryption_probe=lambda handle, path: (
            1 if field == "encryption" else "test-encryption"
        ),
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match=field):
        with store.lease(
            "private.json",
            expected_sha256=hashlib.sha256(b"private").hexdigest(),
            max_bytes=1024,
        ):
            pass


def test_adapter_windows_rechaza_hardlink_interno(tmp_path):
    adapter, root = _adapter(tmp_path)
    artifact = root / "private.json"
    artifact.write_bytes(b"private")
    hardlink = root / "other.json"
    os.link(artifact, hardlink)
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="hardlink"):
        with store.lease(
            "private.json",
            expected_sha256=hashlib.sha256(b"private").hexdigest(),
            max_bytes=1024,
        ):
            pass


def test_adapter_windows_rechaza_hardlink_externo(tmp_path):
    adapter, root = _adapter(tmp_path)
    artifact = root / "private.json"
    artifact.write_bytes(b"private")
    # Segundo nombre FUERA del root: solo NumberOfLinks sobre el handle
    # leased puede detectarlo.
    os.link(artifact, tmp_path / "outside.json")
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="hardlink"):
        with store.lease(
            "private.json",
            expected_sha256=hashlib.sha256(b"private").hexdigest(),
            max_bytes=1024,
        ):
            pass


def test_adapter_windows_rechaza_symlink_de_directorio(tmp_path):
    adapter, root = _adapter(tmp_path)
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    target = root / "real"
    target.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("el entorno no permite crear symlinks")
    (target / "private.json").write_bytes(b"private")
    with pytest.raises(ArtifactIntegrityError, match="reparse"):
        with store.lease(
            "linked/private.json",
            expected_sha256=hashlib.sha256(b"private").hexdigest(),
            max_bytes=1024,
        ):
            pass


@pytest.mark.parametrize(
    "component", ["LocalAppData", "speechtotext", "artifacts", "calibrators"]
)
def test_adapter_windows_rechaza_junction_en_cada_componente(tmp_path, component):
    # Las junctions no requieren privilegios: este negativo de reparse corre
    # SIEMPRE; el skip del test de symlinks no puede enmascararlo.
    import _winapi

    chain_names = ["LocalAppData", "speechtotext", "artifacts", "calibrators"]
    index = chain_names.index(component)
    local = tmp_path / "LocalAppData"
    paths = [
        local,
        local / "speechtotext",
        local / "speechtotext" / "artifacts",
        local / "speechtotext" / "artifacts" / "calibrators",
    ]
    if index > 0:
        paths[index - 1].mkdir(parents=True)
    target = tmp_path / "junction-target"
    deepest = target.joinpath(*chain_names[index + 1 :])
    deepest.mkdir(parents=True)
    (deepest / "es.json").write_bytes(b"private")
    _winapi.CreateJunction(str(target), str(paths[index]))
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    try:
        with pytest.raises(ArtifactIntegrityError, match="reparse"):
            with store.lease(
                "calibrators/es.json",
                expected_sha256=hashlib.sha256(b"private").hexdigest(),
                max_bytes=1024,
            ):
                pass
    finally:
        os.rmdir(paths[index])


def test_adapter_windows_promocion_create_new_sin_overwrite(tmp_path):
    adapter, root = _adapter(tmp_path)
    (root / "descriptors").mkdir()
    source_dir = tmp_path / "private-source"
    source_dir.mkdir()
    source = source_dir / "provider.json"
    source.write_bytes(b"provider")
    expected = hashlib.sha256(b"provider").hexdigest()
    store = PrivateArtifactStore.current_user(filesystem=adapter)

    store.promote_from_path(
        source,
        "descriptors/provider.json",
        expected_sha256=expected,
        max_bytes=1024,
    )
    destination = root / "descriptors" / "provider.json"
    assert destination.read_bytes() == b"provider"

    with pytest.raises(ArtifactIntegrityError, match="colision"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=expected,
            max_bytes=1024,
        )
    assert destination.read_bytes() == b"provider"


def test_adapter_windows_lock_coordina_instancias_y_se_libera_en_teardown(tmp_path):
    first, root = _adapter(tmp_path)
    second = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: root.parent.parent,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: None,
    )
    (root / "descriptors").mkdir()
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    expected = hashlib.sha256(b"provider").hexdigest()
    runtime_store = PrivateArtifactStore.current_user(filesystem=first)
    promotion_store = PrivateArtifactStore.current_user(filesystem=second)

    with pytest.raises(RuntimeError):
        with runtime_store.runtime_session():
            raise RuntimeError("simulated crash")

    with runtime_store.runtime_session():
        with pytest.raises(ArtifactIntegrityError, match="servicio activo"):
            promotion_store.promote_from_path(
                source,
                "descriptors/provider.json",
                expected_sha256=expected,
                max_bytes=1024,
            )

    promotion_store.promote_from_path(
        source,
        "descriptors/provider.json",
        expected_sha256=expected,
        max_bytes=1024,
    )
    assert (root / "descriptors" / "provider.json").read_bytes() == b"provider"


def test_adapter_windows_no_reasegura_lock_existente_entre_instancias(tmp_path):
    secure_calls = []
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)

    def secure(handle, path):
        secure_calls.append(path.name)

    kwargs = {
        "known_folder_probe": lambda: local,
        "acl_probe": lambda handle, path: True,
        "encryption_probe": lambda handle, path: "test-encryption",
        "secure_new_file": secure,
    }
    first = WindowsPrivateArtifactFilesystem(**kwargs)
    second = WindowsPrivateArtifactFilesystem(**kwargs)
    with first.runtime_session():
        with second.runtime_session():
            pass
    assert secure_calls == [".runtime.lock"]


def test_adapter_windows_lock_offline_limpia_temp_reservado_de_caida(tmp_path):
    adapter, root = _adapter(tmp_path)
    stale = root / f".artifact-{'a' * 32}.tmp"
    stale.write_bytes(b"incomplete")
    with adapter.offline_promotion():
        assert not stale.exists()


def test_adapter_windows_fallo_antes_de_rename_no_publica_y_limpia_su_temp(
    tmp_path, monkeypatch
):
    adapter, root = _adapter(tmp_path)
    (root / "descriptors").mkdir()
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    monkeypatch.setattr(
        adapter._api,
        "move_create_new",
        lambda source_path, destination_path: (_ for _ in ()).throw(
            ArtifactIntegrityError("promocion interrumpida")
        ),
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="interrumpida"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not (root / "descriptors" / "provider.json").exists()
    assert not tuple(root.glob(".artifact-*.tmp"))


def test_adapter_windows_fallo_al_asegurar_temp_no_deja_path(tmp_path):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    (root / "descriptors").mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")

    def secure(handle, path):
        if path.name.startswith(".artifact-"):
            raise ArtifactIntegrityError("temp no pudo asegurarse")

    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=secure,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="asegurarse"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not tuple(root.glob(".artifact-*.tmp"))
    assert not (root / "descriptors" / "provider.json").exists()


def test_adapter_windows_fallo_al_asegurar_lock_no_lo_deja_persistente(tmp_path):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: (_ for _ in ()).throw(
            ArtifactIntegrityError("lock no pudo asegurarse")
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="asegurarse"):
        with adapter.runtime_session():
            pass
    assert not (root.parent / ".runtime.lock").exists()
    assert not (root / ".runtime.lock").exists()


def test_adapter_windows_lock_vive_en_el_parent_fijo_speechtotext(tmp_path):
    adapter, root = _adapter(tmp_path)
    with adapter.runtime_session():
        # Brief 1A: el lock OS vive bajo %LOCALAPPDATA%\speechtotext, no
        # dentro del artifacts root junto a los artefactos promovidos.
        assert (root.parent / ".runtime.lock").exists()
        assert not (root / ".runtime.lock").exists()


def test_adapter_windows_reopen_inseguro_retira_solo_destino_nuevo(tmp_path):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    (root / "descriptors").mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    destination = root / "descriptors" / "provider.json"
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: path != destination,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="acl"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not destination.exists()


def test_adapter_windows_rechaza_ads_en_lease_por_el_adapter(tmp_path):
    adapter, root = _adapter(tmp_path)
    (root / "private.json").write_bytes(b"private")
    with adapter.lease_current_user_root() as lease:
        with pytest.raises(ValueError, match="ruta relativa segura"):
            with adapter.lease_file("private.json:Zone.Identifier", lease):
                pass


def test_adapter_windows_abre_el_artefacto_una_sola_vez_por_lease(
    tmp_path, monkeypatch
):
    adapter, root = _adapter(tmp_path)
    (root / "private.json").write_bytes(b"private")
    opened: list[str] = []
    original_open = adapter._api.open

    def counting_open(path, **kwargs):
        opened.append(os.path.normcase(str(path)))
        return original_open(path, **kwargs)

    monkeypatch.setattr(adapter._api, "open", counting_open)
    target = os.path.normcase(str(root / "private.json"))
    with adapter.lease_current_user_root() as lease:
        with adapter.lease_file("private.json", lease) as handle:
            assert opened.count(target) == 1
            # NumberOfLinks proviene de GetFileInformationByHandle sobre ese
            # unico handle abierto.
            assert handle.identity.link_count == 1
            assert handle.stream.read() == b"private"
    assert opened.count(target) == 1


def test_adapter_windows_source_con_junction_en_directorio_falla(tmp_path):
    import _winapi

    adapter, root = _adapter(tmp_path)
    real_dir = tmp_path / "real-sources"
    real_dir.mkdir()
    (real_dir / "provider.json").write_bytes(b"provider")
    linked_dir = tmp_path / "linked-sources"
    _winapi.CreateJunction(str(real_dir), str(linked_dir))
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    try:
        with pytest.raises(ArtifactIntegrityError, match="reparse"):
            store.promote_from_path(
                linked_dir / "provider.json",
                "descriptors/provider.json",
                expected_sha256=hashlib.sha256(b"provider").hexdigest(),
                max_bytes=1024,
            )
    finally:
        os.rmdir(linked_dir)
    assert not (root / "descriptors" / "provider.json").exists()


def test_adapter_windows_source_symlink_falla(tmp_path):
    adapter, root = _adapter(tmp_path)
    real = tmp_path / "real.json"
    real.write_bytes(b"provider")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("el entorno no permite crear symlinks")
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="reparse"):
        store.promote_from_path(
            link,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not (root / "descriptors" / "provider.json").exists()


def test_adapter_windows_source_hardlink_externo_falla(tmp_path):
    adapter, root = _adapter(tmp_path)
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    os.link(source, tmp_path / "second-name.json")
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="hardlink"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not (root / "descriptors" / "provider.json").exists()


@pytest.mark.parametrize("field", ["acl", "encryption"])
def test_adapter_windows_source_sin_acl_o_cifrado_demostrado_falla(tmp_path, field):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    target = os.path.normcase(str(source))

    def acl_probe(handle, path):
        return not (field == "acl" and os.path.normcase(str(path)) == target)

    def encryption_probe(handle, path):
        if field == "encryption" and os.path.normcase(str(path)) == target:
            return None
        return "test-encryption"

    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=acl_probe,
        encryption_probe=encryption_probe,
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match=field):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not (root / "descriptors" / "provider.json").exists()
    assert not tuple(root.glob(".artifact-*.tmp"))


def test_adapter_windows_source_bajo_directorio_con_punto_promociona(tmp_path):
    # La regla de punto inicial aplica a nombres DESTINO relativos al store
    # (temps y .runtime.lock reservados), no a los ancestros del source:
    # promover desde .cache\cal.json debe funcionar.
    adapter, root = _adapter(tmp_path)
    dot_dir = tmp_path / ".cache"
    dot_dir.mkdir()
    source = dot_dir / "cal.json"
    source.write_bytes(b"provider")
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    store.promote_from_path(
        source,
        "calibrators/cal.json",
        expected_sha256=hashlib.sha256(b"provider").hexdigest(),
        max_bytes=1024,
    )
    assert (root / "calibrators" / "cal.json").read_bytes() == b"provider"


def test_adapter_windows_source_leased_bloquea_tamper_y_rename(tmp_path):
    adapter, _ = _adapter(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    source = source_dir / "provider.json"
    source.write_bytes(b"provider")
    with adapter.lease_private_source(source) as handle:
        with pytest.raises(PermissionError):
            source.write_bytes(b"tamper")
        with pytest.raises(PermissionError):
            source.rename(source_dir / "renamed.json")
        assert handle.identity.link_count == 1
    source.write_bytes(b"after")
    assert source.read_bytes() == b"after"


def test_adapter_windows_source_cambiado_entre_open_y_copia_falla(
    tmp_path, monkeypatch
):
    adapter, root = _adapter(tmp_path)
    (root / "descriptors").mkdir()
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    original = adapter.lease_private_source

    @contextmanager
    def inflated_identity(source_path):
        with original(source_path) as handle:
            yield ArtifactSourceHandle(
                stream=handle.stream,
                identity=replace(handle.identity, size=handle.identity.size + 1),
                encryption_provider=handle.encryption_provider,
            )

    monkeypatch.setattr(adapter, "lease_private_source", inflated_identity)
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="identidad de source"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not (root / "descriptors" / "provider.json").exists()
    assert not tuple(root.glob(".artifact-*.tmp"))


def test_adapter_windows_probes_estrictas_solo_en_store_y_source_file(tmp_path):
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    source = source_dir / "provider.json"
    source.write_bytes(b"provider")
    probed: list[str] = []

    def acl_probe(handle, path):
        probed.append(os.path.normcase(str(path)))
        return True

    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=acl_probe,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    store.promote_from_path(
        source,
        "descriptors/provider.json",
        expected_sha256=hashlib.sha256(b"provider").hexdigest(),
        max_bytes=1024,
    )
    probed_set = set(probed)
    normcase = lambda value: os.path.normcase(str(value))  # noqa: E731
    # Nunca se prueban el Known Folder, el volumen del source ni sus ancestros.
    assert normcase(local) not in probed_set
    assert normcase(source.anchor) not in probed_set
    assert normcase(source_dir) not in probed_set
    # Si se prueban los componentes propios del store y el source FILE.
    assert normcase(source) in probed_set
    assert normcase(root) in probed_set
    assert normcase(root.parent) in probed_set
    assert normcase(root / "descriptors") in probed_set
    assert normcase(root / "descriptors" / "provider.json") in probed_set


def test_adapter_windows_promocion_provisiona_root_y_subdirectorios(tmp_path):
    # Maquina fresca: solo existe el Known Folder. offline_promotion debe
    # provisionar speechtotext/artifacts y el subdirectorio destino.
    local = tmp_path / "LocalAppData"
    local.mkdir()
    secured: list[str] = []
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: secured.append(path.name),
    )
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    store.promote_from_path(
        source,
        "calibrators/es.json",
        expected_sha256=hashlib.sha256(b"provider").hexdigest(),
        max_bytes=1024,
    )
    root = local / "speechtotext" / "artifacts"
    assert (root / "calibrators" / "es.json").read_bytes() == b"provider"
    assert secured.count("speechtotext") == 1
    assert secured.count("artifacts") == 1
    assert secured.count("calibrators") == 1

    # Convergencia: una segunda promocion abre+verifica sin re-crear.
    source_two = tmp_path / "source2.json"
    source_two.write_bytes(b"other")
    store.promote_from_path(
        source_two,
        "calibrators/other.json",
        expected_sha256=hashlib.sha256(b"other").hexdigest(),
        max_bytes=1024,
    )
    assert secured.count("speechtotext") == 1
    assert secured.count("artifacts") == 1
    assert secured.count("calibrators") == 1


def test_adapter_windows_reader_no_provisiona_root_faltante(tmp_path):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: None,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(OSError):
        with store.lease(
            "private.json", expected_sha256="0" * 64, max_bytes=1024
        ):
            pass
    with pytest.raises(OSError):
        with adapter.runtime_session():
            pass
    assert not (local / "speechtotext").exists()


def test_adapter_windows_cleanup_no_se_wedgea_con_temp_inseguro(tmp_path):
    # Un crash entre CREATE_NEW y secure_new_file deja un temp con DACL
    # heredada y sin cifrado; la limpieza no puede abortar por sus probes.
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    stale = root / f".artifact-{'b' * 32}.tmp"
    stale.write_bytes(b"half-provisioned")
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: not path.name.endswith(".tmp"),
        encryption_probe=lambda handle, path: (
            None if path.name.endswith(".tmp") else "test-encryption"
        ),
        secure_new_file=lambda handle, path: None,
    )
    with adapter.offline_promotion():
        assert not stale.exists()


def test_adapter_windows_promocion_converge_dir_vacio_y_lock_inseguros(tmp_path):
    # Crash entre CREATE_NEW y secure_new_file: queda un directorio VACIO sin
    # asegurar (artifacts) y un .runtime.lock sin asegurar. Solo la ruta
    # exclusiva de promocion converge (re-asegura sobre el objeto retenido y
    # re-verifica); los readers NO se auto-curan.
    local = tmp_path / "LocalAppData"
    speech = local / "speechtotext"
    root = speech / "artifacts"
    root.mkdir(parents=True)
    lock = speech / ".runtime.lock"
    lock.write_bytes(b"")
    normcase = lambda value: os.path.normcase(str(value))  # noqa: E731
    unsecured = {normcase(root), normcase(lock)}
    secured: list[str] = []

    def secure(handle, path):
        secured.append(path.name)
        unsecured.discard(normcase(path))

    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: normcase(path) not in unsecured,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=secure,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)

    # Reader primero: falla cerrado y no re-asegura nada.
    with pytest.raises(ArtifactIntegrityError, match="acl"):
        with adapter.runtime_session():
            pass
    assert secured == []

    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    digest = hashlib.sha256(b"provider").hexdigest()
    store.promote_from_path(
        source, "calibrators/es.json", expected_sha256=digest, max_bytes=1024
    )
    assert secured.count("artifacts") == 1
    assert secured.count(".runtime.lock") == 1
    assert not unsecured
    # Un lease posterior (reader) funciona sobre el store convergido.
    with store.lease(
        "calibrators/es.json", expected_sha256=digest, max_bytes=1024
    ) as lease:
        assert lease.read_bytes_once() == b"provider"


def test_adapter_windows_promocion_no_converge_dir_inseguro_no_vacio(tmp_path):
    # Un directorio sin asegurar pero NO vacio no es el residuo de un crash
    # entre CREATE_NEW y secure_new_file: sigue fallando cerrado y no se
    # re-asegura ni se toca su contenido.
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    (root / "planted.json").write_bytes(b"planted")
    normcase = lambda value: os.path.normcase(str(value))  # noqa: E731
    unsecured = {normcase(root)}
    secured: list[str] = []

    def secure(handle, path):
        secured.append(path.name)
        unsecured.discard(normcase(path))

    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: normcase(path) not in unsecured,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=secure,
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    with pytest.raises(ArtifactIntegrityError, match="acl"):
        store.promote_from_path(
            source,
            "calibrators/es.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert "artifacts" not in secured
    assert (root / "planted.json").read_bytes() == b"planted"


def test_adapter_windows_promocion_no_converge_junction_en_subdir_destino(tmp_path):
    # La convergencia de la ruta exclusiva re-verifica tampering (owned=False)
    # sobre el handle retenido ANTES de re-asegurar: una junction plantada con
    # el nombre del subdirectorio destino debe fallar 'reparse' y
    # secure_new_file jamas puede tocar ese componente (no se "cura" el
    # target de una junction).
    import _winapi

    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    target = tmp_path / "junction-target"
    target.mkdir()
    _winapi.CreateJunction(str(target), str(root / "calibrators"))
    secured: list[str] = []
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        acl_probe=lambda handle, path: True,
        encryption_probe=lambda handle, path: "test-encryption",
        secure_new_file=lambda handle, path: secured.append(path.name),
    )
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    try:
        with pytest.raises(ArtifactIntegrityError, match="reparse"):
            store.promote_from_path(
                source,
                "calibrators/es.json",
                expected_sha256=hashlib.sha256(b"provider").hexdigest(),
                max_bytes=1024,
            )
    finally:
        os.rmdir(root / "calibrators")
    assert "calibrators" not in secured
    assert not (target / "es.json").exists()


def _current_user_string_sid() -> str:
    output = subprocess.run(
        ["whoami", "/user", "/fo", "csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return output.strip().splitlines()[-1].rsplit(",", 1)[1].strip().strip('"')


def _efs_supported(tmp_path, api) -> bool:
    probe = tmp_path / "efs-probe.bin"
    probe.write_bytes(b"probe")
    return bool(api.advapi32.EncryptFileW(str(probe)))


def test_adapter_windows_lease_con_acl_real_y_solo_encryption_inyectado(tmp_path):
    # Unica probe sustituida y documentada: encryption (EFS puede no existir
    # en el volumen). La probe de ACL es la REAL por defecto y debe aceptar
    # un arbol provisionado con DACL protegida current-user-only.
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    payload = b"private-real-acl"
    artifact = root / "private.json"
    artifact.write_bytes(payload)
    sid = _current_user_string_sid()
    for path in (root.parent, root, artifact):
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"*{sid}:(F)",
            ],
            check=True,
            capture_output=True,
        )
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        encryption_probe=lambda handle, path: "efs-sustituido-solo-en-test",
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with store.lease(
        "private.json",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        max_bytes=1024,
    ) as lease:
        assert lease.read_bytes_once() == payload


def test_adapter_windows_lease_con_acl_real_rechaza_arbol_sin_asegurar(tmp_path):
    # Gemelo negativo del test anterior: el MISMO arbol pero SIN el paso de
    # icacls que lo asegura (DACL heredada, sin proteger). La probe REAL de
    # ACL por defecto debe rechazarlo; solo encryption esta inyectada, asi
    # que el fallo tiene que ser exactamente 'acl'.
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    payload = b"private-real-acl"
    (root / "private.json").write_bytes(payload)
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        encryption_probe=lambda handle, path: "efs-sustituido-solo-en-test",
    )
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="acl"):
        with store.lease(
            "private.json",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            max_bytes=1024,
        ):
            pass


def test_adapter_windows_reader_con_acl_real_no_converge_lock_inseguro(
    tmp_path, monkeypatch
):
    # Gate del reader sobre el lock: un .runtime.lock preexistente SIN
    # asegurar (DACL heredada) en el parent fijo speechtotext debe fallar
    # cerrado con la probe REAL de ACL en runtime_session, sin re-asegurarse
    # jamas (la convergencia del lock es exclusiva de la promocion).
    local = tmp_path / "LocalAppData"
    root = local / "speechtotext" / "artifacts"
    root.mkdir(parents=True)
    # Lock inseguro creado ANTES de cualquier llamada al store.
    lock = root.parent / ".runtime.lock"
    lock.write_bytes(b"")
    sid = _current_user_string_sid()
    for path in (root.parent, root):
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"*{sid}:(F)",
            ],
            check=True,
            capture_output=True,
        )
    # DACL del lock: abrible (usuario F) pero INSEGURA para la probe real —
    # sin SE_DACL_PROTECTED y con un SID ajeno (SYSTEM), como la deja un
    # crash antes de secure_new_file.
    subprocess.run(
        [
            "icacls",
            str(lock),
            "/grant",
            f"*{sid}:(F)",
            "/grant",
            "*S-1-5-18:(F)",
        ],
        check=True,
        capture_output=True,
    )
    adapter = WindowsPrivateArtifactFilesystem(
        known_folder_probe=lambda: local,
        encryption_probe=lambda handle, path: "efs-sustituido-solo-en-test",
    )
    secured: list[str] = []
    original_secure = adapter._secure_new_file

    def counting_secure(handle, path):
        secured.append(path.name)
        original_secure(handle, path)

    monkeypatch.setattr(adapter, "_secure_new_file", counting_secure)
    with pytest.raises(ArtifactIntegrityError, match="acl"):
        with adapter.runtime_session():
            pass
    assert secured == []


def test_adapter_windows_promote_y_lease_con_probes_reales_por_defecto(tmp_path):
    # End-to-end con TODAS las probes por defecto: provisioning real
    # (CREATE_NEW + DACL protegida + EFS), promocion y lease.
    local = tmp_path / "LocalAppData"
    local.mkdir()
    adapter = WindowsPrivateArtifactFilesystem(known_folder_probe=lambda: local)
    if not _efs_supported(tmp_path, adapter._api):
        pytest.skip("EFS unavailable: encryption enforcement unverified")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "es.json"
    payload = b"calibrator-real"
    source.write_bytes(payload)
    # El brief exige source con DACL current-user-only y cifrado demostrado.
    api = adapter._api
    handle = api.open(
        source,
        access=(
            api.GENERIC_READ
            | api.GENERIC_WRITE
            | api.WRITE_DAC
            | api.READ_CONTROL
            | api.FILE_READ_ATTRIBUTES
        ),
        share=(
            api.FILE_SHARE_READ | api.FILE_SHARE_WRITE | api.FILE_SHARE_DELETE
        ),
        creation=api.OPEN_EXISTING,
    )
    try:
        api.secure_new_file(handle, source)
    finally:
        api.close(handle)
    digest = hashlib.sha256(payload).hexdigest()
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    store.promote_from_path(
        source,
        "calibrators/es.json",
        expected_sha256=digest,
        max_bytes=1024,
    )
    with store.runtime_session():
        with store.lease(
            "calibrators/es.json", expected_sha256=digest, max_bytes=1024
        ) as lease:
            assert lease.read_bytes_once() == payload


def test_adapter_windows_reopen_exige_misma_identidad_del_temp(tmp_path, monkeypatch):
    adapter, root = _adapter(tmp_path)
    (root / "descriptors").mkdir()
    source = tmp_path / "source.json"
    source.write_bytes(b"provider")
    original_lease_file = adapter.lease_file

    @contextmanager
    def changed_identity(relative_name, root_lease):
        with original_lease_file(relative_name, root_lease) as handle:
            yield ArtifactFileHandle(
                relative_name=handle.relative_name,
                stream=handle.stream,
                identity=replace(handle.identity, file_id=b"changed!"),
                encryption_provider=handle.encryption_provider,
            )

    monkeypatch.setattr(adapter, "lease_file", changed_identity)
    store = PrivateArtifactStore.current_user(filesystem=adapter)
    with pytest.raises(ArtifactIntegrityError, match="identidad"):
        store.promote_from_path(
            source,
            "descriptors/provider.json",
            expected_sha256=hashlib.sha256(b"provider").hexdigest(),
            max_bytes=1024,
        )
    assert not (root / "descriptors" / "provider.json").exists()
