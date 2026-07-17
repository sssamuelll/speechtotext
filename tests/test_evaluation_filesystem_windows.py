import hashlib
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="requiere Win32")

from speechtotext.evaluation.filesystem import (  # noqa: E402
    CorpusFilesystemError,
    WindowsCorpusFilesystem,
)
from speechtotext.evaluation.manifest import CorpusAsset  # noqa: E402


def _asset(path: str, payload: bytes) -> CorpusAsset:
    return CorpusAsset("primary_audio", path, hashlib.sha256(payload).hexdigest())


def _always(value):
    return lambda handle, path: value


def test_asset_lease_verifica_sha_identidad_y_un_solo_link(tmp_path):
    root = tmp_path / "private"
    (root / "clips").mkdir(parents=True)
    payload = b"hello-audio"
    (root / "clips" / "a.wav").write_bytes(payload)
    fs = WindowsCorpusFilesystem()
    asset = _asset("clips/a.wav", payload)
    with fs.lease_asset(asset, root) as lease:
        assert lease.verified_sha256 == asset.sha256
        assert lease.identity.link_count == 1
        assert lease.stream.read() == payload


def test_asset_lease_rechaza_hardlink(tmp_path):
    root = tmp_path / "private"
    (root / "clips").mkdir(parents=True)
    payload = b"linked-audio"
    original = root / "clips" / "a.wav"
    original.write_bytes(payload)
    os.link(original, root / "clips" / "b.wav")
    fs = WindowsCorpusFilesystem()
    asset = _asset("clips/a.wav", payload)
    with pytest.raises(CorpusFilesystemError, match="hardlink"):
        with fs.lease_asset(asset, root):
            pass


def test_asset_lease_rechaza_componente_reparse(tmp_path):
    import _winapi

    root = tmp_path / "private"
    real = root / "real"
    real.mkdir(parents=True)
    payload = b"reparse-audio"
    (real / "a.wav").write_bytes(payload)
    junction = root / "link"
    _winapi.CreateJunction(str(real), str(junction))
    fs = WindowsCorpusFilesystem()
    asset = _asset("link/a.wav", payload)
    with pytest.raises(CorpusFilesystemError, match="reparse"):
        with fs.lease_asset(asset, root):
            pass


def test_lease_activo_bloquea_delete_concurrente(tmp_path):
    root = tmp_path / "private"
    (root / "clips").mkdir(parents=True)
    payload = b"guarded-audio"
    path = root / "clips" / "a.wav"
    path.write_bytes(payload)
    fs = WindowsCorpusFilesystem()
    asset = _asset("clips/a.wav", payload)
    with fs.lease_asset(asset, root):
        with pytest.raises(PermissionError):
            os.remove(path)
    assert path.exists()


def test_delete_leased_borra_via_handle(tmp_path):
    root = tmp_path / "private"
    (root / "clips").mkdir(parents=True)
    payload = b"doomed-audio"
    path = root / "clips" / "a.wav"
    path.write_bytes(payload)
    fs = WindowsCorpusFilesystem()
    asset = _asset("clips/a.wav", payload)
    with fs.lease_asset(asset, root) as lease:
        fs.delete_leased(lease)
    assert not path.exists()


def test_probes_inyectados_gobiernan_dacl_y_cifrado(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    denied = WindowsCorpusFilesystem(
        acl_probe=_always(False),
        encryption_probe=lambda h, p: (False, "none"),
    )
    assert denied.current_user_only_acl(root) is False
    ok, provider = denied.encryption_at_rest(root)
    assert ok is False
    granted = WindowsCorpusFilesystem(
        acl_probe=_always(True),
        encryption_probe=lambda h, p: (True, "efs"),
    )
    assert granted.current_user_only_acl(root) is True
    assert granted.encryption_at_rest(root) == (True, "efs")


def test_write_privado_solo_nuevo_y_exige_cifrado(tmp_path):
    root = tmp_path / "private"
    reports = root / "reports"
    reports.mkdir(parents=True)
    fs = WindowsCorpusFilesystem(
        acl_probe=_always(True),
        encryption_probe=lambda h, p: (True, "efs"),
    )
    target = reports / "r.json"
    fs.atomic_write_private(target, reports, b"{}")
    assert target.read_bytes() == b"{}"
    with pytest.raises(CorpusFilesystemError, match="destino nuevo"):
        fs.atomic_write_private(target, reports, b"{}")

    blocked = WindowsCorpusFilesystem(
        acl_probe=_always(True),
        encryption_probe=lambda h, p: (False, "none"),
    )
    other = reports / "r2.json"
    with pytest.raises(CorpusFilesystemError, match="cifrado"):
        blocked.atomic_write_private(other, reports, b"{}")
    assert not other.exists()


def test_create_private_secret_es_new_only(tmp_path):
    root = tmp_path / "private"
    secrets = root / "secrets"
    secrets.mkdir(parents=True)
    fs = WindowsCorpusFilesystem(
        acl_probe=_always(True),
        encryption_probe=lambda h, p: (True, "efs"),
    )
    key = secrets / "ref.key"
    identity = fs.create_private_secret(key, secrets, size=32)
    assert identity.size == 32
    assert len(key.read_bytes()) == 32
    with pytest.raises(CorpusFilesystemError, match="secreto existente"):
        fs.create_private_secret(key, secrets, size=32)
