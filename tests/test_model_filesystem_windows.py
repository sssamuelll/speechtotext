from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from speechtotext.models.filesystem import WindowsModelFilesystem
from speechtotext.models.manifest import (
    ModelIntegrityError,
    load_model_manifest,
    verify_model_files,
)


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="requiere Win32")


def _current_user_string_sid() -> str:
    output = subprocess.run(
        ["whoami", "/user", "/fo", "csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return output.strip().splitlines()[-1].rsplit(",", 1)[1].strip().strip('"')


def _data(digest: str) -> dict[str, object]:
    return {
        "schema_version": "speechtotext.model/v1",
        "model_id": "fixture",
        "source": "https://example.invalid/model",
        "revision_kind": "content_digest",
        "revision": "sha256:" + "a" * 64,
        "license": "MIT",
        "format": "ctranslate2",
        "sample_rate": 16000,
        "preprocessing": {"mono": True},
        "files": [{"path": "model.bin", "sha256": digest}],
    }


def _fingerprint(data: dict[str, object]) -> str:
    canonical = json.dumps(
        data,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _fixture(tmp_path, *, acl_ok=True):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.bin").write_bytes(b"weights")
    data = _data(hashlib.sha256(b"weights").hexdigest())
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    filesystem = WindowsModelFilesystem(
        read_only_acl_probe=lambda handle, path: acl_ok
    )
    return root, manifest_path, data, filesystem


def test_adapter_real_falla_si_dacl_read_only_no_se_demuestra(tmp_path):
    root, path, data, filesystem = _fixture(tmp_path, acl_ok=False)
    with pytest.raises(ModelIntegrityError, match="read-only|DACL"):
        load_model_manifest(
            path,
            model_root=root,
            expected_fingerprint=_fingerprint(data),
            filesystem=filesystem,
        )


def test_adapter_real_lesea_ids_y_bloquea_peso_root_y_ancestros(tmp_path):
    root, path, data, filesystem = _fixture(tmp_path)
    manifest = load_model_manifest(
        path,
        model_root=root,
        expected_fingerprint=_fingerprint(data),
        filesystem=filesystem,
    )
    parent = root.parent
    sibling = parent / "sibling.bin"

    with verify_model_files(manifest, root, filesystem=filesystem) as artifact:
        identities = artifact.file_identities
        assert set(identities) == {"manifest.json", "model.bin"}
        with pytest.raises(PermissionError):
            (root / "model.bin").write_bytes(b"tamper")
        with pytest.raises(PermissionError):
            (root / "model.bin").replace(root / "renamed.bin")
        with pytest.raises(PermissionError):
            (root / "model.bin").unlink()
        with pytest.raises(PermissionError):
            root.rename(parent / "renamed-root")
        with pytest.raises(PermissionError):
            parent.rename(parent.parent / "renamed-parent")
        sibling.write_bytes(b"allowed")
        assert sibling.read_bytes() == b"allowed"
        # Re-derive identities from FRESH handles at the end of the lease and
        # compare against the initially recorded pairs (no self-comparison).
        api = filesystem._api
        for name, (volume_serial, file_id) in identities.items():
            fresh = api.open(
                root / name,
                access=api.FILE_READ_ATTRIBUTES,
                share=api.FILE_SHARE_READ | api.FILE_SHARE_WRITE,
                directory=False,
            )
            try:
                fresh_identity, _ = api.identity(fresh)
            finally:
                api.close(fresh)
            assert (fresh_identity.volume_serial, fresh_identity.file_id) == (
                volume_serial,
                file_id,
            )

    (root / "model.bin").write_bytes(b"after")
    root.rename(parent / "renamed-root")


def test_adapter_real_rechaza_extra_y_hardlink_interno(tmp_path):
    root, path, data, filesystem = _fixture(tmp_path)
    manifest = load_model_manifest(
        path,
        model_root=root,
        expected_fingerprint=_fingerprint(data),
        filesystem=filesystem,
    )
    (root / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ModelIntegrityError, match="inventario"):
        with verify_model_files(manifest, root, filesystem=filesystem):
            pass
    (root / "extra.bin").unlink()

    # An internal second name is caught by the inventory comparison alone.
    hardlink = root / "other.bin"
    os.link(root / "model.bin", hardlink)
    with pytest.raises(ModelIntegrityError, match="inventario"):
        with verify_model_files(manifest, root, filesystem=filesystem):
            pass
    hardlink.unlink()


def test_adapter_real_rechaza_hardlink_externo(tmp_path):
    root, path, data, filesystem = _fixture(tmp_path)
    manifest = load_model_manifest(
        path,
        model_root=root,
        expected_fingerprint=_fingerprint(data),
        filesystem=filesystem,
    )
    # A second name OUTSIDE the root leaves the inventory intact; only the
    # NumberOfLinks check on the leased handle can reject it.
    os.link(root / "model.bin", tmp_path / "outside.bin")
    with pytest.raises(ModelIntegrityError, match="hardlink"):
        with verify_model_files(manifest, root, filesystem=filesystem):
            pass


def test_adapter_real_rechaza_directorio_extra_vacio(tmp_path):
    root, path, data, filesystem = _fixture(tmp_path)
    manifest = load_model_manifest(
        path,
        model_root=root,
        expected_fingerprint=_fingerprint(data),
        filesystem=filesystem,
    )
    (root / "empty-dir").mkdir()
    with pytest.raises(ModelIntegrityError, match="inventario"):
        with verify_model_files(manifest, root, filesystem=filesystem):
            pass


def test_adapter_real_rechaza_symlink_de_archivo(tmp_path):
    root, path, data, filesystem = _fixture(tmp_path)
    manifest = load_model_manifest(
        path,
        model_root=root,
        expected_fingerprint=_fingerprint(data),
        filesystem=filesystem,
    )
    target = tmp_path / "target.bin"
    target.write_bytes(b"weights")
    (root / "model.bin").unlink()
    try:
        (root / "model.bin").symlink_to(target)
    except OSError:
        pytest.skip("el entorno no permite crear symlinks")
    with pytest.raises(ModelIntegrityError, match="reparse"):
        with verify_model_files(manifest, root, filesystem=filesystem):
            pass


@pytest.mark.parametrize("position", ["ancestor", "root", "inside"])
def test_adapter_real_rechaza_junction_en_cualquier_componente(tmp_path, position):
    # Junctions need no privilege, so this reparse negative ALWAYS runs; the
    # symlink variant above may skip without masking this coverage.
    import _winapi

    real_parent = tmp_path / "real"
    real_root = real_parent / "model"
    real_root.mkdir(parents=True)
    (real_root / "model.bin").write_bytes(b"weights")
    data = _data(hashlib.sha256(b"weights").hexdigest())
    (real_root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    filesystem = WindowsModelFilesystem(
        read_only_acl_probe=lambda handle, path: True
    )
    junction: Path | None = None
    try:
        if position == "ancestor":
            junction = tmp_path / "linked-parent"
            _winapi.CreateJunction(str(real_parent), str(junction))
            root = junction / "model"
        elif position == "root":
            junction = tmp_path / "linked-root"
            _winapi.CreateJunction(str(real_root), str(junction))
            root = junction
        else:
            root = real_root
            target = tmp_path / "elsewhere"
            target.mkdir()
            junction = root / "sub"
            _winapi.CreateJunction(str(target), str(junction))
        if position == "inside":
            manifest = load_model_manifest(
                root / "manifest.json",
                model_root=root,
                expected_fingerprint=_fingerprint(data),
                filesystem=filesystem,
            )
            with pytest.raises(ModelIntegrityError, match="reparse"):
                with verify_model_files(manifest, root, filesystem=filesystem):
                    pass
        else:
            with pytest.raises(ModelIntegrityError, match="reparse"):
                load_model_manifest(
                    root / "manifest.json",
                    model_root=root,
                    expected_fingerprint=_fingerprint(data),
                    filesystem=filesystem,
                )
    finally:
        if junction is not None:
            os.rmdir(junction)


def test_adapter_real_dacl_por_defecto_rechaza_root_escribible(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.bin").write_bytes(b"weights")
    data = _data(hashlib.sha256(b"weights").hexdigest())
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    filesystem = WindowsModelFilesystem()  # probe DACL real por defecto
    with pytest.raises(ModelIntegrityError, match="read-only|DACL"):
        load_model_manifest(
            manifest_path,
            model_root=root,
            expected_fingerprint=_fingerprint(data),
            filesystem=filesystem,
        )


def test_adapter_real_acepta_dacl_read_only_real_y_bloquea_creacion(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.bin").write_bytes(b"weights")
    data = _data(hashlib.sha256(b"weights").hexdigest())
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    sid = _current_user_string_sid()
    # Protected read-only DACL: standard "Read & execute" grant, inheritance
    # removed. The DEFAULT probe must accept exactly this shape.
    subprocess.run(
        [
            "icacls",
            str(root),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:(OI)(CI)(RX)",
        ],
        check=True,
        capture_output=True,
    )
    try:
        filesystem = WindowsModelFilesystem()  # probe DACL real por defecto
        manifest = load_model_manifest(
            manifest_path,
            model_root=root,
            expected_fingerprint=_fingerprint(data),
            filesystem=filesystem,
        )
        with verify_model_files(manifest, root, filesystem=filesystem) as artifact:
            artifact.require_active()
            with pytest.raises(PermissionError):
                (root / "extra.bin").write_bytes(b"extra")
            with pytest.raises(PermissionError):
                (root / "subdir").mkdir()
    finally:
        # Always restore, or pytest's tmp cleanup breaks on the RX-only tree.
        subprocess.run(
            ["icacls", str(root), "/reset", "/t"],
            check=True,
            capture_output=True,
        )
    (root / "extra.bin").write_bytes(b"extra")
    (root / "subdir").mkdir()


def test_adapter_real_rechaza_dacl_rx_heredada_sin_proteccion(tmp_path):
    # Negativo real de la politica de owner: un root con DACL RX SOLO
    # heredada (owner = usuario actual, SIN SE_DACL_PROTECTED) no prueba
    # nada durable: el owner conserva WRITE_DAC implicito y puede
    # re-ACLear a mitad de lease. La probe por defecto debe rechazarlo.
    parent = tmp_path / "parent"
    root = parent / "model"
    root.mkdir(parents=True)
    (root / "model.bin").write_bytes(b"weights")
    data = _data(hashlib.sha256(b"weights").hexdigest())
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    sid = _current_user_string_sid()
    # icacls sobre el PARENT: el root hijo queda con una DACL RX heredada
    # unicamente (propagada), owner usuario actual y sin bit protected.
    subprocess.run(
        [
            "icacls",
            str(parent),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:(OI)(CI)(RX)",
        ],
        check=True,
        capture_output=True,
    )
    try:
        filesystem = WindowsModelFilesystem()  # probe DACL real por defecto
        with pytest.raises(ModelIntegrityError, match="read-only|DACL"):
            load_model_manifest(
                manifest_path,
                model_root=root,
                expected_fingerprint=_fingerprint(data),
                filesystem=filesystem,
            )
    finally:
        # Always restore, or pytest's tmp cleanup breaks on the RX-only tree.
        subprocess.run(
            ["icacls", str(parent), "/reset", "/t"],
            check=True,
            capture_output=True,
        )


@pytest.mark.parametrize(
    "path",
    [r"C:\outside.bin", r"\\?\C:\outside.bin", r"model.bin:Zone.Identifier"],
)
def test_adapter_real_rechaza_device_ads_y_paths_absolutos(tmp_path, path):
    filesystem = WindowsModelFilesystem(read_only_acl_probe=lambda handle, path: True)
    root = tmp_path / "model"
    root.mkdir()
    with filesystem.lease_read_only_root(root) as lease:
        with pytest.raises((ValueError, ModelIntegrityError), match="ruta|ADS|device"):
            with filesystem.lease_file(path, lease):
                pass
